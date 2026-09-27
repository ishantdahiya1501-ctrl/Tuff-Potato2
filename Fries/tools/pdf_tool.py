import os
import fitz
def create_pdf(text, output_path):
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 545, 800)
    page.insert_textbox(rect, text, fontsize=12, fontname="helv", color=(0, 0, 0))
    doc.save(output_path)
    doc.close()
    return f"PDF created successfully: {output_path}"
def merge_pdfs(input_files, output_path):
    if not input_files:
        return "No PDF files were provided."
    output = fitz.open()
    for file_path in input_files:
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        if not file_path.lower().endswith(".pdf"):
            return f"Not a PDF file: {file_path}"
        pdf = fitz.open(file_path)
        output.insert_pdf(pdf)
        pdf.close()
    output.save(output_path)
    output.close()
    return f"PDFs merged successfully: {output_path}"
def split_pdf(input_file, output_folder):
    if not os.path.exists(input_file):
        return f"File not found: {input_file}"
    if not input_file.lower().endswith(".pdf"):
        return "Input file must be a PDF."
    os.makedirs(output_folder, exist_ok=True)
    pdf = fitz.open(input_file)
    created_files = []
    for page_number in range(len(pdf)):
        new_pdf = fitz.open()
        new_pdf.insert_pdf(pdf, from_page=page_number, to_page=page_number)
        output_path = os.path.join(output_folder, f"page_{page_number + 1}.pdf")
        new_pdf.save(output_path)
        new_pdf.close()
        created_files.append(output_path)
    pdf.close()
    return {"message": "PDF split successfully.", "files": created_files}
def extract_pages(input_file, start_page, end_page, output_path):
    if not os.path.exists(input_file):
        return f"File not found: {input_file}"
    pdf = fitz.open(input_file)
    start_page = int(start_page) - 1
    end_page = int(end_page) - 1
    if start_page < 0 or end_page >= len(pdf):
        pdf.close()
        return "Page range is outside the PDF."
    if start_page > end_page:
        pdf.close()
        return "Start page cannot be greater than end page."
    output = fitz.open()
    output.insert_pdf(pdf, from_page=start_page, to_page=end_page)
    output.save(output_path)
    output.close()
    pdf.close()
    return f"Pages extracted successfully: {output_path}"
def extract_text(input_file):
    if not os.path.exists(input_file):
        return f"File not found: {input_file}"
    pdf = fitz.open(input_file)
    text = ""
    for page_number, page in enumerate(pdf, start=1):
        text += f"\n--- Page {page_number} ---\n"
        text += page.get_text()
    pdf.close()
    return text
def delete_pages(input_file, pages, output_path):
    if not os.path.exists(input_file):
        return f"File not found: {input_file}"
    pdf = fitz.open(input_file)
    pages = [int(page) - 1 for page in pages]
    for page in sorted(pages, reverse=True):
        if page < 0 or page >= len(pdf):
            pdf.close()
            return f"Invalid page number: {page + 1}"
        pdf.delete_page(page)
    pdf.save(output_path)
    pdf.close()
    return f"Pages deleted successfully: {output_path}"
def rotate_pages(input_file, pages, rotation, output_path):
    if not os.path.exists(input_file):
        return f"File not found: {input_file}"
    pdf = fitz.open(input_file)
    pages = [int(page) - 1 for page in pages]
    for page_number in pages:
        if page_number < 0 or page_number >= len(pdf):
            pdf.close()
            return f"Invalid page number: {page_number + 1}"
        page = pdf[page_number]
        page.set_rotation((page.rotation + int(rotation)) % 360)
    pdf.save(output_path)
    pdf.close()
    return f"Pages rotated successfully: {output_path}"
def get_pdf_info(input_file):
    if not os.path.exists(input_file):
        return f"File not found: {input_file}"
    pdf = fitz.open(input_file)
    info = {"file": input_file, "pages": len(pdf), "metadata": pdf.metadata}
    pdf.close()
    return info