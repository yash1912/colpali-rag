import glob
import os
from typing import List, Literal
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import re

def parse_pdf(pdf_paths: List[str], dpi: int = 100, mode: Literal["format", "ocr"] = 'ocr'):

    """
    Parse a list of PDF files and return a list of the OCR output
    using the GOT-OCR2_0 model.

    Args:
        pdf_paths (List[str]): List of paths to the PDF files to parse.
        dpi (int, optional): DPI setting for the PDF to image conversion. Defaults to 100.
        mode (str, optional): Method to use for parsing the PDF. "format" will use the
            layout model to get bounding boxes and find titles, while "ocr" will use
            the GOT-OCR2_0 model to perform OCR on the PDF. Defaults to 'ocr'.

    Returns:
        List[str]: A list of the OCR output from the PDF files.
    """
    repo = "ucaslcl/GOT-OCR2_0"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(repo, 
                                    device_map=device,
                                    torch_dtype=torch.bfloat16, 
                                    trust_remote_code=True,
                                    quantization_config=quantization_config,
                                    low_cpu_mem_usage=True,
                                    use_safetensors=True, 
                                    pad_token_id=tokenizer.eos_token_id 
                                    )
    # Ensure the model is in evaluation mode
    model.eval()
    ocr_results = []
    for file_path in pdf_paths:
        try:
            from pdf2image import convert_from_path
            from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
            # Get the number of pages in the PDF
            from pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(file_path)
            max_pages = info["Pages"]
        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            print(f"Error processing PDF file: {e}")
            return

    
        images = convert_from_path(
            file_path, 
            first_page=1, 
            last_page=max_pages, 
            dpi=dpi  # Adjust DPI for quality vs. memory trade-off
        )
        for page_number, image in enumerate(images):
                outputs = model.chat(tokenizer, image, ocr_type=mode, gradio_input=True)
                print("Processed page:", page_number+1)
                ocr_results.append(outputs)
                
                # Clean up to free memory
                del outputs, images
                torch.cuda.empty_cache()  # If using GPU

    return ocr_results
