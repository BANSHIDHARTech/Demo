"""
Semantic text chunking module for optimal embedding generation.
"""

import logging
from typing import List, Dict, Any, Tuple
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class SemanticChunker:
    """Semantic text chunker that creates meaningful chunks for embeddings."""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target chunk size in tokens (200-400 recommended)
            overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)
        
        # Download required NLTK data
        self._ensure_nltk_data()
        
        # Initialize tokenizer
        self.tokenizer = self._get_tokenizer()
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is available."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
    
    def _get_tokenizer(self):
        """Get appropriate tokenizer for token counting."""
        if TIKTOKEN_AVAILABLE:
            return tiktoken.get_encoding("cl100k_base")
        return None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer and TIKTOKEN_AVAILABLE:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate token count (1 token ≈ 0.75 words)
            return int(len(word_tokenize(text)) / 0.75)
    
    def chunk_extracted_content(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk extracted PDF content into semantic chunks.
        
        Args:
            content: Extracted PDF content from PDFExtractor
            
        Returns:
            List of chunk dictionaries with metadata
        """
        self.logger.info(f"Chunking content from: {content['doc_name']}")
        
        all_chunks = []
        
        for page in content["pages"]:
            page_chunks = self._chunk_page_content(
                page, 
                content["doc_name"]
            )
            all_chunks.extend(page_chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {content['doc_name']}")
        return all_chunks
    
    def _chunk_page_content(self, page: Dict[str, Any], doc_name: str) -> List[Dict[str, Any]]:
        """Chunk content from a single page."""
        chunks = []
        page_num = page["page_number"]
        
        # Include image information in chunks
        has_images = page.get("image_flag", False)
        image_paths = page.get("extracted_images", [])
        
        # Process sections if available, otherwise process main text
        if page["sections"]:
            for section in page["sections"]:
                section_chunks = self._chunk_section(
                    section, 
                    doc_name, 
                    page_num,
                    page["table_flag"],
                    has_images,
                    image_paths
                )
                chunks.extend(section_chunks)
        else:
            # Process main text if no sections identified
            text = page.get("text", "")
            if text.strip():
                text_chunks = self._create_semantic_chunks(text)
                
                for i, chunk_text in enumerate(text_chunks):
                    chunks.append(self._create_chunk_metadata(
                        chunk_text,
                        doc_name,
                        page_num,
                        "Main Content",
                        page["table_flag"],
                        has_images,
                        i,
                        image_paths
                    ))
        
        # Process tables separately if they exist
        if page["tables"]:
            table_chunks = self._chunk_tables(
                page["tables"], 
                doc_name, 
                page_num,
                has_images,
                image_paths
            )
            chunks.extend(table_chunks)
        
        return chunks
    
    def _chunk_section(self, section: Dict[str, Any], doc_name: str, page_num: int, 
                      table_flag: bool, image_flag: bool, image_paths: List[str] = None) -> List[Dict[str, Any]]:
        """Chunk a single section."""
        section_text = section.get("text", "")
        section_title = section.get("title", "Unknown Section")
        
        if image_paths is None:
            image_paths = []
        
        if not section_text.strip():
            return []
        
        # Create semantic chunks for this section
        text_chunks = self._create_semantic_chunks(section_text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunks.append(self._create_chunk_metadata(
                chunk_text,
                doc_name,
                page_num,
                section_title,
                table_flag,
                image_flag,
                i,
                image_paths
            ))
        
        return chunks
    
    def _create_semantic_chunks(self, text: str) -> List[str]:
        """Create semantic chunks from text."""
        if not text.strip():
            return []
        
        # First, try to split by sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds chunk size, split it further
            if sentence_tokens > self.chunk_size:
                # Add current chunk if it has content
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence
                sub_chunks = self._split_long_sentence(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if self.overlap > 0:
                    overlap_text = self._create_overlap(current_chunk)
                    overlap_tokens = self.count_tokens(overlap_text)
                    
                    if overlap_tokens + sentence_tokens <= self.chunk_size:
                        current_chunk = [overlap_text, sentence]
                        current_tokens = overlap_tokens + sentence_tokens
                    else:
                        current_chunk = [sentence]
                        current_tokens = sentence_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split a long sentence into smaller chunks."""
        # Try splitting by common delimiters
        parts = re.split(r'[,;:]', sentence)
        
        if len(parts) == 1:
            # If no delimiters, split by words
            words = sentence.split()
            parts = []
            
            # Create chunks of words that fit within token limit
            current_part = []
            current_tokens = 0
            
            for word in words:
                word_tokens = self.count_tokens(word)
                
                if current_tokens + word_tokens > self.chunk_size and current_part:
                    parts.append(' '.join(current_part))
                    current_part = [word]
                    current_tokens = word_tokens
                else:
                    current_part.append(word)
                    current_tokens += word_tokens
            
            if current_part:
                parts.append(' '.join(current_part))
        
        # Filter and clean parts
        valid_parts = []
        for part in parts:
            part = part.strip()
            if part and self.count_tokens(part) <= self.chunk_size:
                valid_parts.append(part)
        
        return valid_parts if valid_parts else [sentence[:self.chunk_size * 4]]  # Fallback
    
    def _create_overlap(self, chunk_sentences: List[str]) -> str:
        """Create overlap text from the end of current chunk."""
        if not chunk_sentences or self.overlap <= 0:
            return ""
        
        # Take last few sentences that fit within overlap limit
        overlap_text = ""
        overlap_tokens = 0
        
        for sentence in reversed(chunk_sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_text = sentence + " " + overlap_text
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_text.strip()
    
    def _chunk_tables(self, tables: List[Dict[str, Any]], doc_name: str, 
                     page_num: int, image_flag: bool, image_paths: List[str] = None) -> List[Dict[str, Any]]:
        """Create chunks from table data."""
        chunks = []
        
        if image_paths is None:
            image_paths = []
        
        for table_idx, table in enumerate(tables):
            # Convert table to text representation
            table_text = self._table_to_text(table)
            
            if self.count_tokens(table_text) <= self.chunk_size:
                # Table fits in one chunk
                chunks.append(self._create_chunk_metadata(
                    table_text,
                    doc_name,
                    page_num,
                    f"Table {table_idx + 1}",
                    True,  # table_flag is True for table chunks
                    image_flag,
                    0,
                    image_paths,
                    table_data=table["data"]
                ))
            else:
                # Split large table into multiple chunks
                table_chunks = self._split_table(table, table_idx)
                for i, chunk_text in enumerate(table_chunks):
                    chunks.append(self._create_chunk_metadata(
                        chunk_text,
                        doc_name,
                        page_num,
                        f"Table {table_idx + 1} (Part {i + 1})",
                        True,
                        image_flag,
                        i,
                        image_paths,
                        table_data=table["data"] if i == 0 else None
                    ))
        
        return chunks
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table data to text representation."""
        if not table.get("data"):
            return ""
        
        text_parts = [f"Table with {table['rows']} rows and {table['columns']} columns:"]
        
        for row_idx, row in enumerate(table["data"]):
            if row_idx == 0 and table.get("header_row"):
                text_parts.append("Headers: " + " | ".join(row))
            else:
                text_parts.append("Row: " + " | ".join(row))
        
        return "\n".join(text_parts)
    
    def _split_table(self, table: Dict[str, Any], table_idx: int) -> List[str]:
        """Split large table into smaller chunks."""
        chunks = []
        data = table.get("data", [])
        
        if not data:
            return chunks
        
        header = data[0] if table.get("header_row") else []
        rows = data[1:] if header else data
        
        current_chunk_rows = []
        current_tokens = 0
        
        # Always include header in token count
        header_text = "Headers: " + " | ".join(header) if header else ""
        header_tokens = self.count_tokens(header_text)
        
        for row in rows:
            row_text = "Row: " + " | ".join(row)
            row_tokens = self.count_tokens(row_text)
            
            if current_tokens + row_tokens + header_tokens > self.chunk_size and current_chunk_rows:
                # Create chunk with current rows
                chunk_text = self._format_table_chunk(header, current_chunk_rows, table_idx)
                chunks.append(chunk_text)
                
                # Start new chunk
                current_chunk_rows = [row]
                current_tokens = row_tokens
            else:
                current_chunk_rows.append(row)
                current_tokens += row_tokens
        
        # Add remaining rows
        if current_chunk_rows:
            chunk_text = self._format_table_chunk(header, current_chunk_rows, table_idx)
            chunks.append(chunk_text)
        
        return chunks
    
    def _format_table_chunk(self, header: List[str], rows: List[List[str]], table_idx: int) -> str:
        """Format table chunk with header and rows."""
        parts = [f"Table {table_idx + 1}:"]
        
        if header:
            parts.append("Headers: " + " | ".join(header))
        
        for row in rows:
            parts.append("Row: " + " | ".join(row))
        
        return "\n".join(parts)
    
    def _create_chunk_metadata(self, text: str, doc_name: str, page_num: int, 
                             section: str, table_flag: bool, image_flag: bool, 
                             chunk_idx: int, image_paths: List[str] = None,
                             table_data: List[List[str]] = None) -> Dict[str, Any]:
        """Create metadata for a chunk."""
        if image_paths is None:
            image_paths = []
        
        # Convert image_paths list to string for ChromaDB compatibility
        image_paths_str = "|".join(image_paths) if image_paths else ""
        
        # Convert table_data to string if present for ChromaDB compatibility
        table_data_str = ""
        if table_data:
            # Convert table data to JSON string
            import json
            table_data_str = json.dumps(table_data)
            
        return {
            "text": text,
            "metadata": {
                "doc_name": doc_name,
                "page_number": page_num,
                "section": section,
                "table_flag": table_flag,
                "image_flag": image_flag,
                "chunk_index": chunk_idx,
                "token_count": self.count_tokens(text),
                "char_count": len(text),
                "image_paths": image_paths_str,
                "table_data": table_data_str
            }
        }