from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams,Distance,PointStruct

class QdrantStorage:
    def __init__(self,url="http://localhost:6333", collection = "docs_gemini",dim=768):
        self.client=QdrantClient(url=url,timeout=30)
        self.collection=collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim,distance=Distance.COSINE)
            )
            print(f"Collection '{self.collection}' created.")
    def upsert(self,ids,vectors,payloads):
        points=[
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i])
                for i in range(len(ids))]
        self.client.upsert(self.collection,points=points,wait=True)
        print(f"Upserted {len(points)} points to collection '{self.collection}'.")
    
    
    def search(self, query_vector, top_k: int = 5):
        response = self.client.query_points(
            self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )
        
        contexts = []
        sources = set()

        for i in response.points:
            payload = getattr(i, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)
                
        return {"contexts": contexts, "sources": list(sources)}     



         