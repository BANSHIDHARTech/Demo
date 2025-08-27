"""
Simple Adobe PDF Extract - Clean and Simple
Gives exactly the JSON format requested by the user
"""
import os
import zipfile
import json
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class SimpleAdobeExtractor:
    def __init__(self, input_dir: str = "inputs", output_dir: str = "outputs"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.adobe_creds = 'pdfservices-api-credentials.json'
        
        # Ensure directories exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Import Adobe SDK
        self._import_adobe_sdk()
    
    def _import_adobe_sdk(self):
        """Import Adobe PDF Services SDK v4.x"""
        try:
            from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
            from adobe.pdfservices.operation.pdf_services import PDFServices
            from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
            from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
            from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
            from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
            from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
            
            self.ServicePrincipalCredentials = ServicePrincipalCredentials
            self.PDFServices = PDFServices
            self.PDFServicesMediaType = PDFServicesMediaType
            self.ExtractPDFJob = ExtractPDFJob
            self.ExtractElementType = ExtractElementType
            self.ExtractPDFParams = ExtractPDFParams
            self.ExtractPDFResult = ExtractPDFResult
            
            logger.info("✅ Adobe PDF Services SDK imported successfully")
        except ImportError as e:
            raise SystemExit(f"❌ Missing pdfservices-sdk. Run: pip install pdfservices-sdk\nError: {e}")
    
    def _extract_credentials(self, creds_path: Path) -> tuple:
        """Extract client_id and client_secret from credentials file"""
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
        
        # Handle the specific credential format
        if 'project' in creds_data:
            project = creds_data['project']
            if 'workspace' in project and 'details' in project['workspace']:
                details = project['workspace']['details']
                if 'credentials' in details and isinstance(details['credentials'], list) and len(details['credentials']) > 0:
                    first_cred = details['credentials'][0]
                    if 'oauth_server_to_server' in first_cred:
                        oauth_creds = first_cred['oauth_server_to_server']
                        client_id = oauth_creds.get('client_id')
                        client_secrets = oauth_creds.get('client_secrets', [])
                        client_secret = client_secrets[0] if client_secrets else None
                        return client_id, client_secret
        
        raise ValueError("Could not find valid credentials in the credentials file")
    
    def _create_clean_json(self, adobe_data: Dict, pdf_filename: str) -> Dict:
        """Create the EXACT JSON format requested by the user"""
        page_count = adobe_data.get('extended_metadata', {}).get('page_count', 0)
        elements = adobe_data.get('elements', [])
        
        # Initialize the clean structure
        clean_output = {
            "document_id": Path(pdf_filename).stem,
            "metadata": {
                "source": pdf_filename,
                "page_count": page_count
            },
            "pages": []
        }
        
        # Process each page
        for page_num in range(page_count):
            page_data = {
                "page_number": page_num + 1,
                "text": "",
                "tables": [],
                "images": []
            }
            
            # Extract text for this page
            page_text = []
            for element in elements:
                if (element.get('Page') == page_num and 
                    element.get('Path', '').startswith('//Document/P')):
                    text_content = element.get('Text', '').strip()
                    if text_content:
                        page_text.append(text_content)
            
            page_data["text"] = " ".join(page_text)
            
            # Extract tables for this page
            table_count = 0
            for element in elements:
                if (element.get('Page') == page_num and 
                    element.get('Path', '').startswith('//Document/Table')):
                    table_count += 1
                    table_id = f"t{table_count}"
                    
                    # Try to get table data
                    table_data = []
                    if 'attributes' in element and 'NumRow' in element['attributes'] and 'NumCol' in element['attributes']:
                        # Look for table cells (TD elements)
                        table_cells = []
                        for cell_element in elements:
                            if (cell_element.get('Page') == page_num and 
                                cell_element.get('Path', '').startswith('//Document/TD')):
                                cell_text = cell_element.get('Text', '').strip()
                                if cell_text:
                                    table_cells.append(cell_text)
                        
                        # Simple table reconstruction
                        if table_cells:
                            # Group into rows (very basic)
                            rows = []
                            current_row = []
                            for i, cell in enumerate(table_cells):
                                current_row.append(cell)
                                if (i + 1) % element['attributes']['NumCol'] == 0:
                                    rows.append(current_row)
                                    current_row = []
                            
                            if current_row:  # Add any remaining cells
                                rows.append(current_row)
                            
                            table_data = rows
                    
                    clean_table = {
                        "table_id": table_id,
                        "data": table_data
                    }
                    page_data["tables"].append(clean_table)
            
            # Extract images for this page
            image_count = 0
            for element in elements:
                if (element.get('Page') == page_num and 
                    element.get('Path', '').startswith('//Document/Figure')):
                    image_count += 1
                    image_id = f"i{image_count}"
                    
                    # Try to get caption from nearby text
                    caption = ""
                    image_bounds = element.get('Bounds', [])
                    if image_bounds and len(image_bounds) >= 4:
                        # Look for text below the image
                        for text_element in elements:
                            if (text_element.get('Page') == page_num and 
                                text_element.get('Path', '').startswith('//Document/P') and
                                'Bounds' in text_element):
                                
                                text_bounds = text_element['Bounds']
                                if len(text_bounds) >= 4:
                                    # Check if text is below and near the image
                                    if (text_bounds[1] < image_bounds[1] and  # Below image
                                        abs(text_bounds[0] - image_bounds[0]) < 50):  # Roughly aligned
                                        caption = text_element.get('Text', '').strip()
                                        break
                    
                    clean_image = {
                        "image_id": image_id,
                        "path": f"images/page{page_num + 1}_{image_id}.png",
                        "caption": caption
                    }
                    page_data["images"].append(clean_image)
            
            clean_output["pages"].append(page_data)
        
        return clean_output
    
    def extract_pdf(self, pdf_path: Path) -> Dict:
        """Extract content from a single PDF file"""
        logger.info(f"Starting Adobe PDF extraction for: {pdf_path.name}")
        
        # Verify input file exists
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {pdf_path}")
        
        # Find and validate credentials
        creds_path = Path(self.adobe_creds)
        if not creds_path.exists():
            raise FileNotFoundError(f"Adobe credentials file not found: {creds_path}")
        
        client_id, client_secret = self._extract_credentials(creds_path)
        
        try:
            # Setup credentials
            credentials = self.ServicePrincipalCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            
            # Create PDF Services instance
            pdf_services = self.PDFServices(credentials=credentials)
            
            # Create asset from input file
            logger.info("Creating input asset from PDF file")
            with open(pdf_path, 'rb') as file:
                input_stream = file.read()
            
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=self.PDFServicesMediaType.PDF)
            
            # Create extract parameters
            logger.info("Configuring extraction parameters")
            extract_pdf_params = self.ExtractPDFParams(
                elements_to_extract=[
                    self.ExtractElementType.TEXT,
                    self.ExtractElementType.TABLES
                ]
            )
            
            # Create and submit extract job
            extract_pdf_job = self.ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)
            logger.info("Executing PDF extraction operation")
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, self.ExtractPDFResult)
            
            # Get result asset
            result_asset = pdf_services_response.get_result().get_resource()
            stream_asset = pdf_services.get_content(result_asset)
            
            # Save result as zip
            result_zip = self.output_dir / f"{pdf_path.stem}_adobe_result.zip"
            with open(result_zip, "wb") as file:
                file.write(stream_asset.get_input_stream())
            
            logger.info(f"Extraction result saved to: {result_zip}")
            
            # Extract JSON from zip
            with zipfile.ZipFile(result_zip, 'r') as z:
                json_files = [n for n in z.namelist() if n.lower().endswith('.json')]
                if not json_files:
                    raise SystemExit("❌ No JSON found in Adobe result zip")
                
                json_name = json_files[0]
                logger.info(f"Extracting JSON file: {json_name}")
                
                with z.open(json_name) as f:
                    adobe_data = json.load(f)
                
                # Log extraction summary
                if 'elements' in adobe_data:
                    element_types = {}
                    for element in adobe_data['elements']:
                        if 'Path' in element:
                            path = element['Path']
                            element_type = path.split('/')[-1] if '/' in path else 'Unknown'
                            element_types[element_type] = element_types.get(element_type, 0) + 1
                    
                    logger.info("📊 Extraction Summary:")
                    for elem_type, count in element_types.items():
                        logger.info(f"   {elem_type}: {count} elements")
                    
                    if 'extended_metadata' in adobe_data and 'page_count' in adobe_data['extended_metadata']:
                        logger.info(f"   Total Pages: {adobe_data['extended_metadata']['page_count']}")
                
                # Create clean output in the EXACT format requested
                clean_data = self._create_clean_json(adobe_data, pdf_path.name)
                
                # Save clean output
                clean_json = self.output_dir / f"{pdf_path.stem}_clean.json"
                with open(clean_json, 'w', encoding='utf-8') as out:
                    json.dump(clean_data, out, indent=2)
                
                logger.info(f"✅ Adobe extraction completed successfully")
                logger.info(f"📁 Clean results: {clean_json}")
                logger.info(f"📁 Full results: {result_zip}")
                
                return clean_data
                
        except Exception as e:
            logger.error(f"❌ Adobe extraction failed: {str(e)}")
            raise
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process all PDF files in the input directory"""
        pdf_files = list(self.input_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files to process")
            return []
        
        results = []
        for pdf_file in pdf_files:
            try:
                result = self.extract_pdf(pdf_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Simple Adobe PDF Extract - Clean Output')
    parser.add_argument('--input-dir', default='inputs', help='Input directory containing PDF files')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for results')
    parser.add_argument('--pdf', help='Specific PDF file to process (optional)')
    
    args = parser.parse_args()
    
    try:
        extractor = SimpleAdobeExtractor(args.input_dir, args.output_dir)
        
        if args.pdf:
            # Process specific PDF
            pdf_path = Path(args.pdf)
            if not pdf_path.exists():
                pdf_path = extractor.input_dir / args.pdf
            
            result = extractor.extract_pdf(pdf_path)
        else:
            # Process all PDFs in input directory
            results = extractor.process_all_pdfs()
            logger.info(f"Processed {len(results)} PDF file(s)")
            
            # Save combined results
            if results:
                combined_file = extractor.output_dir / "combined_clean.json"
                with open(combined_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Combined results saved to: {combined_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {str(e)}")
        return 1

if __name__ == '__main__':
    exit(main())
