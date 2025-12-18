# resume_parser.py
import PyPDF2

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Simple PDF text extraction. Works for most text-based PDFs.
    Note: Scanned PDFs may need OCR (e.g., Tesseract) — not covered here.
    """
    text = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                t = page.extract_text() or ''
            except Exception:
                t = ''
            text.append(t)
    return '\n'.join(text)
