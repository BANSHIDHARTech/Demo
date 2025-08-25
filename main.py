#!/usr/bin/env python3
"""
PDF Embedder - CLI entrypoint for PDF text extraction and embeddings pipeline.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from pipeline import PDFEmbeddingPipeline
from utils import setup_logging


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs, chunk semantically, and store embeddings in ChromaDB"
    )
    parser.add_argument(
        "pdf_file", 
        nargs="?", 
        help="Specific PDF file to process (optional, processes all PDFs in input/ if not specified)"
    )
    parser.add_argument(
        "--input-dir", 
        default="input", 
        help="Input directory containing PDFs (default: input)"
    )
    parser.add_argument(
        "--output-dir", 
        default="output", 
        help="Output directory for JSON files (default: output)"
    )
    parser.add_argument(
        "--db-dir", 
        default="db", 
        help="ChromaDB persistence directory (default: db)"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=300, 
        help="Target chunk size in tokens (default: 300)"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(verbose=args.verbose)
    
    # Create directories if they don't exist
    for dir_path in [args.input_dir, args.output_dir, args.db_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = PDFEmbeddingPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        db_dir=args.db_dir,
        chunk_size=args.chunk_size
    )
    
    try:
        if args.pdf_file:
            # Process single PDF
            pdf_path = Path(args.input_dir) / args.pdf_file
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                sys.exit(1)
            
            logger.info(f"Processing single PDF: {args.pdf_file}")
            pipeline.process_single_pdf(args.pdf_file)
        else:
            # Process all PDFs in input directory
            pdf_files = list(Path(args.input_dir).glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in {args.input_dir}")
                sys.exit(0)
            
            logger.info(f"Processing {len(pdf_files)} PDFs from {args.input_dir}")
            pipeline.process_all_pdfs()
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()