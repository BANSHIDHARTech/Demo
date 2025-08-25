"""
Main pipeline orchestrating PDF processing workflow.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from tqdm import tqdm

from extract import PDFExtractor
from chunker import SemanticChunker
from embedder import EmbeddingGenerator
from utils import validate_pdf_file, load_config


class PDFEmbeddingPipeline:
    """Main pipeline for PDF text extraction, chunking, and embedding storage."""
    
    def __init__(self, input_dir: str = "input", output_dir: str = "output", 
                 db_dir: str = "db", chunk_size: int = 300):
        """
        Initialize the PDF embedding pipeline.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for output JSON files
            db_dir: Directory for ChromaDB persistence
            chunk_size: Target chunk size in tokens
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.db_dir = Path(db_dir)
        self.chunk_size = chunk_size
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure directories exist
        for directory in [self.input_dir, self.output_dir, self.db_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = PDFExtractor(output_dir=self.output_dir)
        self.chunker = SemanticChunker(chunk_size=chunk_size)
        self.embedder = EmbeddingGenerator(db_dir=self.db_dir)
        
        self.logger.info("PDF Embedding Pipeline initialized")
    
    def process_single_pdf(self, pdf_filename: str) -> Dict[str, Any]:
        """
        Process a single PDF file through the complete pipeline.
        
        Args:
            pdf_filename: Name of the PDF file in input directory
            
        Returns:
            Processing results and statistics
        """
        pdf_path = self.input_dir / pdf_filename
        
        # Validate PDF file
        if not validate_pdf_file(pdf_path):
            raise ValueError(f"Invalid or missing PDF file: {pdf_path}")
        
        self.logger.info(f"Processing PDF: {pdf_filename}")
        
        results = {
            "pdf_file": pdf_filename,
            "status": "failed",
            "extracted_pages": 0,
            "created_chunks": 0,
            "stored_embeddings": 0,
            "error": None
        }
        
        try:
            # Step 1: Extract content
            self.logger.info("Step 1: Extracting content...")
            with tqdm(desc="Extracting", unit="pages") as pbar:
                extracted_content = self.extractor.extract_pdf_content(pdf_path)
                pbar.update(extracted_content["total_pages"])
            
            results["extracted_pages"] = extracted_content["total_pages"]
            
            # Save extracted content
            extraction_file = self.extractor.save_extracted_content(
                extracted_content, self.output_dir
            )
            
            # Step 2: Create semantic chunks
            self.logger.info("Step 2: Creating semantic chunks...")
            with tqdm(desc="Chunking", unit="pages") as pbar:
                chunks = self.chunker.chunk_extracted_content(extracted_content)
                pbar.update(len(extracted_content["pages"]))
            
            results["created_chunks"] = len(chunks)
            
            # Save chunks for debugging/inspection
            chunks_file = self._save_chunks(chunks, pdf_filename)
            
            # Step 3: Generate embeddings and store
            self.logger.info("Step 3: Generating embeddings and storing...")
            storage_results = self.embedder.embed_and_store_chunks(chunks)
            
            results["stored_embeddings"] = storage_results["stored_count"]
            results["failed_embeddings"] = storage_results["failed_count"]
            results["status"] = "completed"
            
            # Save processing summary
            self._save_processing_summary(results, pdf_filename)
            
            self.logger.info(f"Successfully processed {pdf_filename}")
            return results
            
        except Exception as e:
            error_msg = f"Error processing {pdf_filename}: {str(e)}"
            self.logger.error(error_msg)
            results["error"] = error_msg
            results["status"] = "failed"
            return results
    
    def process_all_pdfs(self) -> Dict[str, Any]:
        """
        Process all PDF files in the input directory.
        
        Returns:
            Overall processing results and statistics
        """
        # Find all PDF files
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {self.input_dir}")
            return {
                "total_files": 0,
                "successful": 0,
                "failed": 0,
                "results": []
            }
        
        self.logger.info(f"Processing {len(pdf_files)} PDF files")
        
        overall_results = {
            "total_files": len(pdf_files),
            "successful": 0,
            "failed": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "results": []
        }
        
        # Process each PDF with progress bar
        with tqdm(total=len(pdf_files), desc="Processing PDFs", unit="file") as pbar:
            for pdf_file in pdf_files:
                try:
                    result = self.process_single_pdf(pdf_file.name)
                    overall_results["results"].append(result)
                    
                    if result["status"] == "completed":
                        overall_results["successful"] += 1
                        overall_results["total_pages"] += result["extracted_pages"]
                        overall_results["total_chunks"] += result["created_chunks"]
                        overall_results["total_embeddings"] += result["stored_embeddings"]
                    else:
                        overall_results["failed"] += 1
                
                except Exception as e:
                    self.logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                    overall_results["failed"] += 1
                    overall_results["results"].append({
                        "pdf_file": pdf_file.name,
                        "status": "failed",
                        "error": str(e)
                    })
                
                pbar.update(1)
        
        # Save overall summary
        self._save_overall_summary(overall_results)
        
        self.logger.info(f"Batch processing completed: {overall_results['successful']} successful, "
                        f"{overall_results['failed']} failed")
        
        return overall_results
    
    def _save_chunks(self, chunks: List[Dict[str, Any]], pdf_filename: str) -> Path:
        """Save chunks to JSON file for inspection."""
        chunks_file = self.output_dir / f"{Path(pdf_filename).stem}_chunks.json"
        
        try:
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "pdf_file": pdf_filename,
                    "total_chunks": len(chunks),
                    "chunks": chunks
                }, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved chunks to: {chunks_file}")
            return chunks_file
            
        except Exception as e:
            self.logger.warning(f"Failed to save chunks: {str(e)}")
            return None
    
    def _save_processing_summary(self, results: Dict[str, Any], pdf_filename: str) -> Path:
        """Save processing summary for a single PDF."""
        summary_file = self.output_dir / f"{Path(pdf_filename).stem}_summary.json"
        
        try:
            # Add collection stats
            collection_stats = self.embedder.get_collection_stats()
            results["collection_stats"] = collection_stats
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved processing summary to: {summary_file}")
            return summary_file
            
        except Exception as e:
            self.logger.warning(f"Failed to save processing summary: {str(e)}")
            return None
    
    def _save_overall_summary(self, results: Dict[str, Any]) -> Path:
        """Save overall batch processing summary."""
        summary_file = self.output_dir / "batch_processing_summary.json"
        
        try:
            # Add collection stats and pipeline info
            results["collection_stats"] = self.embedder.get_collection_stats()
            results["pipeline_config"] = {
                "input_dir": str(self.input_dir),
                "output_dir": str(self.output_dir),
                "db_dir": str(self.db_dir),
                "chunk_size": self.chunk_size
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved batch summary to: {summary_file}")
            return summary_file
            
        except Exception as e:
            self.logger.warning(f"Failed to save batch summary: {str(e)}")
            return None
    
    def search_documents(self, query: str, n_results: int = 5, 
                        doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content across all processed documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            doc_filter: Optional document name filter
            
        Returns:
            List of similar chunks with metadata
        """
        filter_metadata = None
        if doc_filter:
            filter_metadata = {"doc_name": doc_filter}
        
        return self.embedder.search_similar(
            query=query, 
            n_results=n_results, 
            filter_metadata=filter_metadata
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get overall pipeline statistics."""
        try:
            # Count PDFs in input directory
            pdf_count = len(list(self.input_dir.glob("*.pdf")))
            
            # Get output file counts
            json_files = len(list(self.output_dir.glob("*.json")))
            
            # Get collection stats
            collection_stats = self.embedder.get_collection_stats()
            
            return {
                "input_pdfs": pdf_count,
                "output_files": json_files,
                "database": collection_stats,
                "directories": {
                    "input": str(self.input_dir),
                    "output": str(self.output_dir),
                    "database": str(self.db_dir)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get pipeline stats: {str(e)}")
            return {"error": str(e)}
    
    def cleanup_document(self, doc_name: str) -> Dict[str, Any]:
        """
        Remove all data for a specific document.
        
        Args:
            doc_name: Name of document to clean up
            
        Returns:
            Cleanup results
        """
        results = {
            "document": doc_name,
            "deleted_chunks": 0,
            "deleted_files": []
        }
        
        try:
            # Delete from ChromaDB
            deleted_chunks = self.embedder.delete_document_chunks(doc_name)
            results["deleted_chunks"] = deleted_chunks
            
            # Delete output files
            patterns = [
                f"{doc_name}_extracted.json",
                f"{doc_name}_chunks.json", 
                f"{doc_name}_summary.json"
            ]
            
            for pattern in patterns:
                file_path = self.output_dir / pattern
                if file_path.exists():
                    file_path.unlink()
                    results["deleted_files"].append(str(file_path))
            
            self.logger.info(f"Cleaned up document: {doc_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup {doc_name}: {str(e)}")
            results["error"] = str(e)
            return results