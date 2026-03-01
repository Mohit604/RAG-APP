import logging
from fastapi import FastAPI
from groq import Groq
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import os 
import uuid
import datetime
from data_loader import load_and_chunk,embed_texts
from vector_db import QdrantStorage
from custom_types import RAGUPsertResult,RAGSearchResult,RAGQueryResult,RAGCHUNKANDSrc

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
inngest_client = inngest.Inngest(
app_id="mohit-rag-app",
logger=logging.getLogger("uvicorn"),
is_production=False,
serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
)
async def rag_inngest_pdf(ctx:inngest.Context):#event triggered function to ingest pdf and upsert to vector db
   #ctx event context
     def _load(ctx: inngest.Context) -> RAGCHUNKANDSrc:
         pdf_path = ctx.event.data["pdf_path"]
         source_id = ctx.event.data.get("source_id", pdf_path)
         chunks = load_and_chunk(pdf_path)
         return RAGCHUNKANDSrc(chunks=chunks, source_id=source_id)
     def _upsert(chunks_and_src: RAGCHUNKANDSrc) -> RAGUPsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUPsertResult(ingested=len(chunks))
     chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGCHUNKANDSrc)
     ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUPsertResult)
     return ingested.model_dump()#Pydantic model object ko Python dictionary me convert karta hai.

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf")
    
)

async def rag_query_pdf(ctx:inngest.Context):
    def _search(question:str,top_k:int=5) -> RAGSearchResult:
       query_vec=embed_texts([question])[0]
       found=QdrantStorage().search(query_vec,top_k)
       return RAGSearchResult(contexts=found["contexts"],sources=found["sources"])

    question=ctx.event.data["question"]
    top_k=ctx.event.data.get("top_k",5)
    found=await ctx.step.run("embed-and-search",lambda:_search(question=question,top_k=top_k),output_type=RAGSearchResult)
    
    context_block="\n\n".join(found.contexts)
    user_content=(f"Answer the question based on the following retrieved contexts:\n\n{context_block}\n\n"
                  f"Question: {question}\n\n"
                  f"Answer:"
    )
   
    def _generate(prompt: str) -> RAGQueryResult:
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant. Answer strictly based on the provided contexts."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile", 
        )
        answer = chat_completion.choices[0].message.content
        return RAGQueryResult(answer=answer, contexts=found.contexts, sources=found.sources, num_contexts=len(found.contexts))

    
    final_result = await ctx.step.run(
        "generate-answer-groq", 
        lambda: _generate(user_content), 
        output_type=RAGQueryResult
    )

    return final_result.model_dump()


app = FastAPI()
#Endpoint
inngest.fast_api.serve(app,inngest_client,[rag_inngest_pdf, rag_query_pdf])

#inngest dev server npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery  ##
#uv run uvicorn main:app


#qdrant satup (docker run -d --name qdrantRagDb -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant)