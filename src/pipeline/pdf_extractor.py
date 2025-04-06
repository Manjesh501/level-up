import pdfplumber
from typing import Dict, Optional
import json
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from SSR PDF (handles multi-column layouts)."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Adjust for multi-column (use `vertical_strategy="text"`)
                text += page.extract_text(layout=True, x_tolerance=2)
    except Exception as e:
        raise ValueError(f"Failed to extract text: {e}")
    return text

def save_text_to_json(text: str, output_path: str):
    """Save extracted text to JSON for later processing."""
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"raw_text": text}, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Use correct relative path from the pipeline directory
    pdf_path =r"C:\Users\DAMODAR\level-up\data\raw\sample_ssr.pdf"
    output_path = r"C:\Users\DAMODAR\level-up\data\raw\sample_ssr.json"
    
    text = extract_text_from_pdf(pdf_path)
    save_text_to_json(text, output_path)
    print(f"Extracted text saved to {output_path}")