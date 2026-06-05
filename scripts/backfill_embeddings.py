import asyncio
import os
import json
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from sentence_transformers import SentenceTransformer

async def backfill():
    load_dotenv(".env.local")
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    print("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    async with driver.session() as session:
        print("Finding nodes missing embeddings...")
        # Check Entities first
        res = await session.run("MATCH (e:Entity) WHERE e.embedding IS NULL RETURN e.id as id, e.name as name, e.description as description")
        entities = await res.data()
        print(f"Found {len(entities)} Entities missing embeddings.")
        
        for ent in entities:
            # For entities, we can use description or name
            text = f"{ent['name']}: {ent['description']}" if ent['description'] else ent['name']
            emb = model.encode(text).tolist()
            await session.run("MATCH (e:Entity {id: $id}) SET e.embedding = $emb", id=ent['id'], emb=emb)
            print(f"Updated Entity: {ent['name']}")

        # Check Chunks
        res = await session.run("MATCH (c:Chunk) WHERE c.embedding IS NULL RETURN c.id as id, c.content as content")
        chunks = await res.data()
        print(f"Found {len(chunks)} Chunks missing embeddings.")
        
        for chunk in chunks:
            emb = model.encode(chunk['content']).tolist()
            await session.run("MATCH (c:Chunk {id: $id}) SET c.embedding = $emb", id=chunk['id'], emb=emb)
            print(f"Updated Chunk: {chunk['id']}")

    await driver.close()
    print("Backfill complete!")

if __name__ == "__main__":
    asyncio.run(backfill())
