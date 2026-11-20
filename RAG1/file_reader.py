from pymupdf import open
import re
from typing import List
from pathlib import Path


def extract_paragraphs_from_pdf(file_path: str) -> List[str]:
    """
    Extracts paragraphs from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        List[str]: A list of extracted paragraphs.
    """
    docs = open(file_path)
    paragraphs = []
    for page in docs:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b:(b[1], b[0]))
        
        for block in blocks:
            text = block[4]
            text = re.sub(r"\r+", " ", text).strip()
            
            
            if text:
                paragraphs.append(text)
    
    
    
    
    return paragraphs


paragraphs = extract_paragraphs_from_pdf("../PDFs/Penguins_ACL.pdf")

for para in paragraphs:
    print("*" * 50)
    print(para)
    print("*" * 50)