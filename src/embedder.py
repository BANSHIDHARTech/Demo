"""
Embedding generation and ChromaDB storage module.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class EmbeddingGenerator:
    """Generate embeddings and store in ChromaDB."""
    
    def __init__(self, db_dir: Path, model_name: str = "all-MiniLM-L6-v2", collection_name: str = "pdf_embeddings"):
        """
        Initialize embedding generator.
        
        Args:
            db_dir: Directory for ChromaDB persistence
            model_name: Sentence transformer model name
            collection_name: ChromaDB collection name
        """
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        
        # Ensure db directory exists
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.client = self._initialize_chromadb()
        self.collection = self._get_or_create_collection()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client with persistence."""
        try:
            settings = Settings(
                persist_directory=str(self.db_dir),
                anonymized_telemetry=False
            )
            
            client = chromadb.PersistentClient(
                path=str(self.db_dir),
                settings=settings
            )
            
            self.logger.info(f"Initialized ChromaDB with persistence at: {self.db_dir}")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            self.logger.info(f"Using existing collection: {self.collection_name}")
            
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "PDF text chunks with embeddings"}
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def embed_and_store_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate embeddings for chunks and store in ChromaDB.
        
        Args:
            chunks: List of chunk dictionaries from SemanticChunker
            
        Returns:
            Storage statistics
        """
        if not chunks:
            self.logger.warning("No chunks provided for embedding")
            return {"stored_count": 0, "failed_count": 0}
        
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts for batch embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batch for efficiency
        try:
            embeddings = self.model.encode(
                texts, 
                show_progress_bar=True,
                batch_size=32
            )
            self.logger.info(f"Generated {len(embeddings)} embeddings")
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents = []
        embedding_list = []
        
        stored_count = 0
        failed_count = 0
        
        for chunk, embedding in zip(chunks, embeddings):
            try:
                # Generate unique ID
                doc_name = chunk["metadata"]["doc_name"]
                page_num = chunk["metadata"]["page_number"]
                chunk_idx = chunk["metadata"]["chunk_index"]
                unique_id = f"{doc_name}_page{page_num}_chunk{chunk_idx}_{str(uuid.uuid4())[:8]}"
                
                ids.append(unique_id)
                documents.append(chunk["text"])
                metadatas.append(chunk["metadata"])
                embedding_list.append(embedding.tolist())
                
                stored_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to prepare chunk for storage: {str(e)}")
                failed_count += 1
                continue
        
        # Store in ChromaDB
        if ids:
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embedding_list
                )
                
                self.logger.info(f"Successfully stored {stored_count} chunks in ChromaDB")
                
            except Exception as e:
                self.logger.error(f"Failed to store chunks in ChromaDB: {str(e)}")
                raise
        
        return {
            "stored_count": stored_count,
            "failed_count": failed_count,
            "total_chunks_in_db": self.collection.count()
        }
    
    def search_similar(self, query: str, n_results: int = 5, 
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the database.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar chunks with metadata and similarity scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.model.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            count = self.collection.count()
            
            stats = {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "model_name": self.model_name,
                "db_directory": str(self.db_dir)
            }
            
            # Get sample metadata if collection is not empty
            if count > 0:
                sample = self.collection.get(limit=1, include=["metadatas"])
                if sample["metadatas"]:
                    stats["sample_metadata"] = sample["metadatas"][0]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_document_chunks(self, doc_name: str) -> int:
        """
        Delete all chunks for a specific document.
        
        Args:
            doc_name: Name of the document to delete
            
        Returns:
            Number of deleted chunks
        """
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"doc_name": doc_name},
                include=["metadatas"]
            )
            
            if results["ids"]:
                # Delete the chunks
                self.collection.delete(ids=results["ids"])
                deleted_count = len(results["ids"])
                
                self.logger.info(f"Deleted {deleted_count} chunks for document: {doc_name}")
                return deleted_count
            else:
                self.logger.info(f"No chunks found for document: {doc_name}")
                return 0
                
        except Exception as e:
            self.logger.error(f"Failed to delete chunks for {doc_name}: {str(e)}")
            raise
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all data)."""
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate it
            self.collection = self._get_or_create_collection()
            
            self.logger.info(f"Reset collection: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {str(e)}")
            raise