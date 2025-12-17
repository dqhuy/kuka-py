"""
PDF Page Extractor for Training Dataset Generation

Extracts pages from PDF files at different DPI resolutions for training and testing.
Supports batch processing of multiple PDFs.

Author: KUKA-LIB Team
Date: 2024-12-17
"""

import cv2
import numpy as np
import fitz  # PyMuPDF
import os
from pathlib import Path
from typing import List, Tuple
import glob


def extract_pdf_pages_multi_dpi(
    pdf_path: str,
    output_dir: str,
    dpi_values: List[int] = [50, 100],
    prefix: str = None
) -> List[Tuple[str, int, int]]:
    """
    Extract all pages from a PDF at multiple DPI resolutions.
    
    Parameters:
    -----------
    pdf_path : str
        Path to input PDF file
    output_dir : str
        Directory to save extracted images
    dpi_values : List[int]
        List of DPI values to extract (default: [50, 100])
    prefix : str
        Optional prefix for output filenames (if None, uses PDF filename)
        
    Returns:
    --------
    List[Tuple[str, int, int]]: List of (image_path, dpi, page_number)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get PDF filename without extension
    if prefix is None:
        prefix = Path(pdf_path).stem
    
    extracted_files = []
    
    try:
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        print(f"📄 Processing PDF: {pdf_path}")
        print(f"   Total pages: {total_pages}")
        
        # Extract each page at each DPI
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
            for dpi in dpi_values:
                # Calculate zoom factor (72 DPI is default)
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                
                # Render page to image
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to numpy array
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                
                # Convert RGB to BGR for OpenCV
                if pix.n == 3:  # RGB
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif pix.n == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
                # Save image
                output_filename = f"{prefix}_page{page_num+1:03d}_{dpi}dpi.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, img)
                
                extracted_files.append((output_path, dpi, page_num + 1))
                print(f"   ✅ Saved: {output_filename} ({img.shape[1]}x{img.shape[0]})")
        
        pdf_document.close()
        print(f"✨ Successfully extracted {len(extracted_files)} images")
        
    except Exception as e:
        print(f"❌ Error processing PDF {pdf_path}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return extracted_files


def batch_extract_pdfs(
    input_dir: str,
    output_dir: str,
    pattern: str = "Vais_*.pdf",
    dpi_values: List[int] = [50, 100]
) -> dict:
    """
    Batch extract all PDFs matching a pattern.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing PDF files
    output_dir : str
        Directory to save extracted images
    pattern : str
        Glob pattern to match PDF files (default: "Vais_*.pdf")
    dpi_values : List[int]
        List of DPI values to extract
        
    Returns:
    --------
    dict: Statistics about extraction
    """
    # Find all matching PDFs
    search_pattern = os.path.join(input_dir, pattern)
    pdf_files = glob.glob(search_pattern)
    
    if not pdf_files:
        print(f"⚠️  No PDF files found matching pattern: {pattern}")
        print(f"   Search path: {search_pattern}")
        return {"pdfs_found": 0, "images_extracted": 0, "files": []}
    
    print(f"🔍 Found {len(pdf_files)} PDF files:")
    for pdf_path in pdf_files:
        print(f"   - {os.path.basename(pdf_path)}")
    print()
    
    all_extracted = []
    
    # Process each PDF
    for pdf_path in sorted(pdf_files):
        extracted = extract_pdf_pages_multi_dpi(
            pdf_path,
            output_dir,
            dpi_values=dpi_values
        )
        all_extracted.extend(extracted)
        print()
    
    stats = {
        "pdfs_found": len(pdf_files),
        "images_extracted": len(all_extracted),
        "files": all_extracted,
        "dpi_values": dpi_values
    }
    
    return stats


def create_dataset_summary(output_dir: str, stats: dict) -> str:
    """
    Create a summary file of the extracted dataset.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing extracted images
    stats : dict
        Statistics from batch_extract_pdfs()
        
    Returns:
    --------
    str: Path to summary file
    """
    summary_path = os.path.join(output_dir, "dataset_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DATASET EXTRACTION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Total PDFs processed: {stats['pdfs_found']}\n")
        f.write(f"Total images extracted: {stats['images_extracted']}\n")
        f.write(f"DPI values: {stats['dpi_values']}\n\n")
        
        # Group by DPI
        by_dpi = {}
        for file_path, dpi, page_num in stats['files']:
            if dpi not in by_dpi:
                by_dpi[dpi] = []
            by_dpi[dpi].append(file_path)
        
        f.write("Images per DPI:\n")
        for dpi in sorted(by_dpi.keys()):
            f.write(f"  {dpi} DPI: {len(by_dpi[dpi])} images\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("IMAGE LIST\n")
        f.write("=" * 70 + "\n\n")
        
        for file_path, dpi, page_num in sorted(stats['files']):
            filename = os.path.basename(file_path)
            f.write(f"{filename} (Page {page_num}, {dpi} DPI)\n")
    
    print(f"📊 Dataset summary saved to: {summary_path}")
    return summary_path


if __name__ == "__main__":
    """
    Example usage:
    python kukalib/pdf_extractor.py
    """
    import sys
    
    # Default paths
    input_dir = "docs/samples/input"
    output_dir = "docs/samples/extracted_dataset"
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print("="*70)
    print("PDF DATASET EXTRACTOR")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Extract all Vais_*.pdf files
    stats = batch_extract_pdfs(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern="Vais_*.pdf",
        dpi_values=[50, 100]
    )
    
    # Create summary
    if stats['images_extracted'] > 0:
        create_dataset_summary(output_dir, stats)
        print("\n✅ Dataset extraction complete!")
    else:
        print("\n⚠️  No images were extracted. Please check:")
        print("   1. PDF files exist in the input directory")
        print("   2. PDF files match the pattern 'Vais_*.pdf'")
        print("   3. PDF files are not corrupted")
