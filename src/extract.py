"""
PDF text, table, image, and metadata extraction module.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import io
import hashlib

import pypdf
import pdfplumber
from pdfplumber.page import Page
from PIL import Image


class PDFExtractor:
    """Extract text, tables, images, and metadata from PDF files."""
    
    def __init__(self, output_dir: Path):
        """Initialize PDF extractor."""
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        
        # Create images directory
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_pdf_content(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        self.logger.info(f"Extracting content from: {pdf_path.name}")
        
        content = {
            "doc_name": pdf_path.stem,
            "file_path": str(pdf_path),
            "pages": [],
            "total_pages": 0,
            "metadata": {}
        }
        
        try:
            # Extract metadata using pypdf
            content["metadata"] = self._extract_metadata(pdf_path)
            
            # Extract text and tables using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                content["total_pages"] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_content = self._extract_page_content(page, page_num)
                    content["pages"].append(page_content)
            
            # Extract images using pypdf
            self._extract_images_from_pdf(pdf_path, content)
                    
        except Exception as e:
            self.logger.error(f"Error extracting from {pdf_path.name}: {str(e)}")
            raise
        
        self.logger.info(f"Extracted {len(content['pages'])} pages from {pdf_path.name}")
        return content
    
    def _extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata using pypdf."""
        metadata = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                if pdf_reader.metadata:
                    metadata.update({
                        "title": pdf_reader.metadata.get('/Title', ''),
                        "author": pdf_reader.metadata.get('/Author', ''),
                        "subject": pdf_reader.metadata.get('/Subject', ''),
                        "creator": pdf_reader.metadata.get('/Creator', ''),
                        "producer": pdf_reader.metadata.get('/Producer', ''),
                        "creation_date": str(pdf_reader.metadata.get('/CreationDate', '')),
                        "modification_date": str(pdf_reader.metadata.get('/ModDate', ''))
                    })
                
                metadata["page_count"] = len(pdf_reader.pages)
                metadata["encrypted"] = pdf_reader.is_encrypted
                
        except Exception as e:
            self.logger.warning(f"Could not extract metadata: {str(e)}")
        
        return metadata
    
    def _extract_page_content(self, page: Page, page_num: int) -> Dict[str, Any]:
        """Extract content from a single page."""
        page_content = {
            "page_number": page_num,
            "text": "",
            "sections": [],
            "tables": [],
            "image_objects": [],
            "extracted_images": [],
            "table_flag": False,
            "image_flag": False
        }
        
        try:
            # Extract text
            text = page.extract_text() or ""
            page_content["text"] = text.strip()
            
            # Extract tables
            tables = page.extract_tables()
            if tables:
                page_content["table_flag"] = True
                page_content["tables"] = self._process_tables(tables)
            
            # Detect images (basic detection based on page objects)
            if hasattr(page, 'images') and page.images:
                page_content["image_flag"] = True
                page_content["image_objects"] = [
                    {
                        "x0": img.get("x0", 0),
                        "y0": img.get("y0", 0), 
                        "x1": img.get("x1", 0),
                        "y1": img.get("y1", 0),
                        "width": img.get("width", 0),
                        "height": img.get("height", 0)
                    }
                    for img in page.images
                ]
            
            # Identify sections based on text formatting
            page_content["sections"] = self._identify_sections(text, page)
            
        except Exception as e:
            self.logger.warning(f"Error extracting page {page_num}: {str(e)}")
        
        return page_content
    
    def _extract_images_from_pdf(self, pdf_path: Path, content: Dict[str, Any]) -> None:
        """Extract images from PDF using pypdf and save them."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                doc_name = content["doc_name"]
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    if '/XObject' in page['/Resources']:
                        xObject = page['/Resources']['/XObject'].get_object()
                        
                        image_count = 0
                        for obj in xObject:
                            if xObject[obj]['/Subtype'] == '/Image':
                                try:
                                    image_path = self._save_image_from_pdf_object(
                                        xObject[obj], doc_name, page_num, image_count
                                    )
                                    
                                    if image_path:
                                        # Add to corresponding page content
                                        if page_num <= len(content["pages"]):
                                            page_content = content["pages"][page_num - 1]
                                            page_content["extracted_images"].append(str(image_path))
                                            page_content["image_flag"] = True
                                        
                                        image_count += 1
                                        
                                except Exception as e:
                                    self.logger.warning(f"Failed to extract image from page {page_num}: {str(e)}")
                                    continue
                                    
        except Exception as e:
            self.logger.warning(f"Failed to extract images from {pdf_path.name}: {str(e)}")
    
    def _save_image_from_pdf_object(self, image_obj, doc_name: str, page_num: int, image_count: int) -> Optional[Path]:
        """Save an image object from PDF to file."""
        try:
            # Get image data
            data = image_obj.get_data()
            
            # Create unique filename
            image_hash = hashlib.md5(data).hexdigest()[:8]
            filename = f"{doc_name}_page{page_num}_img{image_count}_{image_hash}"
            
            # Determine image format and save accordingly
            filter_type = image_obj.get('/Filter')
            
            if filter_type == '/DCTDecode':
                # JPEG image
                image_path = self.images_dir / f"{filename}.jpg"
                with open(image_path, 'wb') as img_file:
                    img_file.write(data)
                    
            elif filter_type == '/FlateDecode':
                # PNG or other compressed format
                try:
                    width = image_obj.get('/Width')
                    height = image_obj.get('/Height')
                    color_space = image_obj.get('/ColorSpace')
                    
                    if width and height:
                        # Determine color mode
                        if color_space == '/DeviceRGB' or str(color_space).endswith('RGB'):
                            mode = 'RGB'
                            channels = 3
                        elif color_space == '/DeviceGray':
                            mode = 'L'
                            channels = 1
                        elif color_space == '/DeviceCMYK':
                            mode = 'CMYK'
                            channels = 4
                        else:
                            mode = 'RGB'
                            channels = 3
                        
                        # Calculate expected data length
                        expected_length = width * height * channels
                        
                        if len(data) >= expected_length:
                            # Try to create PIL Image
                            try:
                                # Reshape data to proper dimensions
                                if len(data) > expected_length:
                                    data = data[:expected_length]
                                
                                img = Image.frombytes(mode, (width, height), data)
                                image_path = self.images_dir / f"{filename}.png"
                                img.save(image_path, 'PNG')
                                
                            except Exception as e:
                                # If PIL fails, try converting to RGB and saving
                                try:
                                    import numpy as np
                                    # Reshape as numpy array and convert
                                    arr = np.frombuffer(data[:expected_length], dtype=np.uint8)
                                    arr = arr.reshape((height, width, channels))
                                    
                                    if channels == 4:  # CMYK
                                        # Convert CMYK to RGB (simple approximation)
                                        rgb_arr = 255 - arr[:, :, :3]  # Invert CMY, ignore K
                                        img = Image.fromarray(rgb_arr, 'RGB')
                                    elif channels == 1:  # Grayscale
                                        img = Image.fromarray(arr.squeeze(), 'L')
                                    else:  # RGB
                                        img = Image.fromarray(arr, 'RGB')
                                    
                                    image_path = self.images_dir / f"{filename}.png"
                                    img.save(image_path, 'PNG')
                                    
                                except Exception:
                                    # Final fallback: save as binary
                                    image_path = self.images_dir / f"{filename}.bin"
                                    with open(image_path, 'wb') as img_file:
                                        img_file.write(data)
                        else:
                            # Data length mismatch, save as binary
                            image_path = self.images_dir / f"{filename}.bin"
                            with open(image_path, 'wb') as img_file:
                                img_file.write(data)
                    else:
                        # Missing dimensions, save as binary
                        image_path = self.images_dir / f"{filename}.bin"
                        with open(image_path, 'wb') as img_file:
                            img_file.write(data)
                            
                except Exception as e:
                    # Fallback: save raw data as binary
                    image_path = self.images_dir / f"{filename}.bin"
                    with open(image_path, 'wb') as img_file:
                        img_file.write(data)
                        
            elif filter_type == '/JPXDecode':
                # JPEG 2000 format
                try:
                    # Try to save as JPEG 2000 or convert to PNG
                    image_path = self.images_dir / f"{filename}.jp2"
                    with open(image_path, 'wb') as img_file:
                        img_file.write(data)
                except Exception:
                    # Fallback to binary
                    image_path = self.images_dir / f"{filename}.bin"
                    with open(image_path, 'wb') as img_file:
                        img_file.write(data)
                        
            else:
                # Unknown format or no filter, try to detect format from data
                try:
                    # Check for common image signatures
                    if data.startswith(b'\xff\xd8\xff'):
                        # JPEG signature
                        image_path = self.images_dir / f"{filename}.jpg"
                        with open(image_path, 'wb') as img_file:
                            img_file.write(data)
                    elif data.startswith(b'\x89PNG\r\n\x1a\n'):
                        # PNG signature
                        image_path = self.images_dir / f"{filename}.png"
                        with open(image_path, 'wb') as img_file:
                            img_file.write(data)
                    elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
                        # GIF signature
                        image_path = self.images_dir / f"{filename}.gif"
                        with open(image_path, 'wb') as img_file:
                            img_file.write(data)
                    else:
                        # Try to open with PIL to detect format
                        try:
                            from io import BytesIO
                            img = Image.open(BytesIO(data))
                            format_ext = img.format.lower() if img.format else 'png'
                            image_path = self.images_dir / f"{filename}.{format_ext}"
                            img.save(image_path)
                        except Exception:
                            # Final fallback: save as binary
                            image_path = self.images_dir / f"{filename}.bin"
                            with open(image_path, 'wb') as img_file:
                                img_file.write(data)
                except Exception:
                    # Ultimate fallback: save as binary
                    image_path = self.images_dir / f"{filename}.bin"
                    with open(image_path, 'wb') as img_file:
                        img_file.write(data)
            
            self.logger.debug(f"Saved image: {image_path}")
            return image_path
            
        except Exception as e:
            self.logger.warning(f"Failed to save image: {str(e)}")
            return None
    
    def _process_tables(self, tables: List[List[List[str]]]) -> List[Dict[str, Any]]:
        """Process extracted tables into structured format."""
        processed_tables = []
        
        for i, table in enumerate(tables):
            if table and len(table) > 0:
                # Remove None values and clean data
                clean_table = []
                for row in table:
                    clean_row = [cell.strip() if cell else "" for cell in row]
                    clean_table.append(clean_row)
                
                processed_tables.append({
                    "table_id": i,
                    "rows": len(clean_table),
                    "columns": len(clean_table[0]) if clean_table else 0,
                    "data": clean_table,
                    "header_row": clean_table[0] if clean_table else []
                })
        
        return processed_tables
    
    def _identify_sections(self, text: str, page: Page) -> List[Dict[str, Any]]:
        """Identify sections based on text structure and formatting."""
        sections = []
        
        if not text.strip():
            return sections
        
        # Simple section identification based on common patterns
        lines = text.split('\n')
        current_section = {
            "title": "Main Content",
            "start_line": 0,
            "content": []
        }
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristics for section headers
            if self._is_likely_header(line):
                # Save previous section if it has content
                if current_section["content"]:
                    current_section["text"] = '\n'.join(current_section["content"])
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    "title": line,
                    "start_line": i,
                    "content": []
                }
            else:
                current_section["content"].append(line)
        
        # Add the last section
        if current_section["content"]:
            current_section["text"] = '\n'.join(current_section["content"])
            sections.append(current_section)
        
        # If no sections were identified, create one main section
        if not sections:
            sections.append({
                "title": "Main Content",
                "start_line": 0,
                "text": text,
                "content": lines
            })
        
        return sections
    
    def _is_likely_header(self, line: str) -> bool:
        """Determine if a line is likely a section header."""
        # Simple heuristics for headers
        line = line.strip()
        
        if len(line) == 0:
            return False
        
        # Check for common header patterns
        header_indicators = [
            line.isupper() and len(line) > 2 and len(line) < 100,  # ALL CAPS
            line.endswith(':') and len(line.split()) <= 10,  # Ends with colon
            line.startswith(('Chapter', 'Section', 'Part', 'Introduction', 'Conclusion', 'Abstract')),
            len(line) < 80 and not line.endswith('.') and len(line.split()) <= 15  # Short without period
        ]
        
        return any(header_indicators)
    
    def save_extracted_content(self, content: Dict[str, Any], output_dir: Path) -> Path:
        """Save extracted content to JSON file."""
        output_file = output_dir / f"{content['doc_name']}_extracted.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved extracted content to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving extracted content: {str(e)}")
            raise