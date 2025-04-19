from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

class ImprovedLocalRAG:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.sentences = []
        
    def preprocess_text(self, text):
        """Split text into clean sentences"""
        # Split by periods, question marks, or exclamation marks followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Remove empty sentences and extra whitespace
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def create_index(self, text):
        """Create FAISS index from text"""
        self.sentences = self.preprocess_text(text)
        embeddings = self.embedding_model.encode(self.sentences)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def search(self, query, k=1):
        """Search for most relevant sentences"""
        if self.index is None:
            raise ValueError("Index not initialized.")
        
        query_embedding = self.embedding_model.encode(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        distances, indices = self.index.search(query_embedding, k)
        return [self.sentences[i] for i in indices[0]]

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = ImprovedLocalRAG()
    
    # Your input text
    text = """
    Quantum computing is the use of quantum-mechanical phenomena to perform computation.
    A quantum computer uses qubits which can be in superposition of states.
    Python is a popular programming language for quantum computing with libraries like Qiskit.
    The Bloch sphere is a representation of a qubit's quantum state.
    Riya Singh is a web developer.
    """
    
    # Create index
    rag.create_index(text)
    
    # Perform a search
    query = "What does riya do?"
    results = rag.search(query)
    
    print(f"Question: {query}")
    print("\nMost relevant answer:")
    print(results[0])