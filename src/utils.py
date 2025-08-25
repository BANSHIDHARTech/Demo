"""
Utility functions for the PDF embedding pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create logger for this module
    logger = logging.getLogger("pdf_embedder")
    
    # Suppress verbose logs from external libraries unless in debug mode
    if not verbose:
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger


def validate_pdf_file(pdf_path: Path) -> bool:
    """
    Validate that a PDF file exists and is readable.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if valid PDF file
    """
    if not pdf_path.exists():
        return False
    
    if not pdf_path.is_file():
        return False
    
    if pdf_path.suffix.lower() != '.pdf':
        return False
    
    # Check if file is readable
    try:
        with open(pdf_path, 'rb') as f:
            # Read first few bytes to check if it looks like a PDF
            header = f.read(8)
            return header.startswith(b'%PDF-')
    except (IOError, OSError):
        return False


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        "chunk_size": 300,
        "overlap": 50,
        "model_name": "all-MiniLM-L6-v2",
        "collection_name": "pdf_embeddings",
        "batch_size": 32
    }
    
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not load config from {config_path}: {str(e)}")
    
    return default_config


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted file size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    i = 0
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def get_pdf_info(pdf_path: Path) -> Dict[str, Any]:
    """
    Get basic information about a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with PDF information
    """
    info = {
        "filename": pdf_path.name,
        "file_size": 0,
        "file_size_formatted": "0 B",
        "exists": False,
        "readable": False
    }
    
    try:
        if pdf_path.exists():
            info["exists"] = True
            info["file_size"] = pdf_path.stat().st_size
            info["file_size_formatted"] = format_file_size(info["file_size"])
            info["readable"] = validate_pdf_file(pdf_path)
            
    except (OSError, IOError) as e:
        info["error"] = str(e)
    
    return info


def print_processing_stats(results: Dict[str, Any]) -> None:
    """
    Print formatted processing statistics.
    
    Args:
        results: Processing results dictionary
    """
    print("\n" + "="*60)
    print("PDF PROCESSING RESULTS")
    print("="*60)
    
    if "total_files" in results:
        # Batch processing results
        print(f"Total PDFs processed: {results['total_files']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Total pages extracted: {results.get('total_pages', 0)}")
        print(f"Total chunks created: {results.get('total_chunks', 0)}")
        print(f"Total embeddings stored: {results.get('total_embeddings', 0)}")
        
        if results.get('results'):
            print("\nFile Details:")
            for result in results['results']:
                status_symbol = "✓" if result['status'] == 'completed' else "✗"
                print(f"  {status_symbol} {result['pdf_file']}")
                if result['status'] == 'completed':
                    print(f"    Pages: {result.get('extracted_pages', 0)}, "
                          f"Chunks: {result.get('created_chunks', 0)}, "
                          f"Embeddings: {result.get('stored_embeddings', 0)}")
                else:
                    print(f"    Error: {result.get('error', 'Unknown error')}")
    else:
        # Single file processing results
        status_symbol = "✓" if results['status'] == 'completed' else "✗"
        print(f"{status_symbol} {results['pdf_file']}")
        print(f"Pages extracted: {results.get('extracted_pages', 0)}")
        print(f"Chunks created: {results.get('created_chunks', 0)}")
        print(f"Embeddings stored: {results.get('stored_embeddings', 0)}")
        
        if results['status'] != 'completed':
            print(f"Error: {results.get('error', 'Unknown error')}")
    
    print("="*60)


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary with dependency availability status
    """
    dependencies = {}
    
    try:
        import pypdf
        dependencies['pypdf'] = True
    except ImportError:
        dependencies['pypdf'] = False
    
    try:
        import pdfplumber
        dependencies['pdfplumber'] = True
    except ImportError:
        dependencies['pdfplumber'] = False
    
    try:
        import sentence_transformers
        dependencies['sentence_transformers'] = True
    except ImportError:
        dependencies['sentence_transformers'] = False
    
    try:
        import chromadb
        dependencies['chromadb'] = True
    except ImportError:
        dependencies['chromadb'] = False
    
    try:
        import tqdm
        dependencies['tqdm'] = True
    except ImportError:
        dependencies['tqdm'] = False
    
    try:
        import nltk
        dependencies['nltk'] = True
    except ImportError:
        dependencies['nltk'] = False
    
    return dependencies


def ensure_dependencies() -> bool:
    """
    Ensure all dependencies are available and exit if not.
    
    Returns:
        True if all dependencies are available
    """
    deps = check_dependencies()
    missing = [name for name, available in deps.items() if not available]
    
    if missing:
        print("Missing required dependencies:", file=sys.stderr)
        for dep in missing:
            print(f"  - {dep}", file=sys.stderr)
        print("\nPlease install missing dependencies with:", file=sys.stderr)
        print("  pip install -r requirements.txt", file=sys.stderr)
        return False
    
    return True


def create_sample_search_script() -> str:
    """
    Create a sample search script for users.
    
    Returns:
        Sample script content
    """
    script_content = '''#!/usr/bin/env python3
"""
Sample search script for querying the PDF embeddings database.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from embedder import EmbeddingGenerator

def search_pdfs(query, n_results=5, doc_filter=None):
    """Search the PDF embeddings database."""
    embedder = EmbeddingGenerator(db_dir="db")
    
    results = embedder.search_similar(
        query=query,
        n_results=n_results,
        filter_metadata={"doc_name": doc_filter} if doc_filter else None
    )
    
    print(f"Search Results for: '{query}'")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        print(f"Result {i}:")
        print(f"  Document: {metadata['doc_name']}")
        print(f"  Page: {metadata['page_number']}")
        print(f"  Section: {metadata['section']}")
        print(f"  Similarity: {result['similarity_score']:.3f}")
        print(f"  Text: {result['text'][:200]}...")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_example.py 'your search query'")
        sys.exit(1)
    
    query = sys.argv[1]
    search_pdfs(query)
'''
    
    return script_content