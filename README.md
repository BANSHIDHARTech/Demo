# PDF Image & Table Extraction

This repository provides utilities for extracting **images** (in JP2 format for human-readable quality) and **tables** (via Adobe credentials) from PDF documents.

## Repository Structure

- **Main Branch**  
  Contains all the files for **image extraction**. Extracted images are stored in **JP2 format**, making them easily viewable.

- **`table/` Directory**  
  Handles **table extraction** using **Adobe credentials** for structured data extraction.

---

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
Create and activate a virtual environment

bash
Copy code
python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
.\venv\Scripts\activate
Install dependencies

 ``` bash

pip install -r requirements.txt
Run Instructions
Table Extraction (Adobe)
Navigate to the table/ directory and run the extraction scripts with your Adobe credentials configured.

Image Extraction (JP2 format)
Example command to extract images from a sample PDF:
 ```

 ``` bash

python scripts/adobe_extract_simple.py --pdf inputs/bray_sample.pdf
