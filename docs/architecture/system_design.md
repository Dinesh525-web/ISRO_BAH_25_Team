## Knowledge-Graph Layer

| Component | Purpose | Technology |
|-----------|---------|------------|
| Neo4j DB  | Property graph storage | Neo4j 5 |
| KG Builder | Extracts entities & relations from processed docs | spaCy + custom rules |
| Cypher API | Query interface for RAG service | Neo4j Python driver |

### Sequence – Chat Flow

1. **User UI** sends message → FastAPI `/chat/`
2. **RAG Service**  
   a. Validates + enriches query via KG  
   b. Retrieves relevant chunks from FAISS & DB  
   c. Generates answer with GPT-4  
3. Stores conversation, returns response to UI.
