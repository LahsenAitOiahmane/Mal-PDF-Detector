"""
PDF Static Analysis Script for Machine Learning
Extracts features from PDF files for classification purposes.
"""

import os
import csv
import re
import math
from pathlib import Path


def calculate_shannon_entropy(data):
    """
    Calculate Shannon entropy of binary data.
    
    Args:
        data: bytes object containing file data
        
    Returns:
        float: Shannon entropy value
    """
    if not data:
        return 0.0
    
    # Count frequency of each byte value
    byte_counts = {}
    for byte in data:
        byte_counts[byte] = byte_counts.get(byte, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    data_length = len(data)
    
    for count in byte_counts.values():
        probability = count / data_length
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy


def extract_pdf_version(data):
    """
    Extract PDF version from header (e.g., %PDF-1.4).
    
    Args:
        data: bytes object containing file data
        
    Returns:
        str: PDF version string (e.g., "1.4") or "Unknown"
    """
    try:
        # Read first 1024 bytes to find PDF header
        header = data[:1024].decode('latin-1', errors='ignore')
        match = re.search(r'%PDF-(\d+\.\d+)', header)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "Unknown"


def count_keywords(data, keywords):
    """
    Count occurrences of keywords in binary data.
    
    Args:
        data: bytes object containing file data
        keywords: list of keyword strings to search for
        
    Returns:
        dict: Dictionary with keyword counts
    """
    counts = {}
    # Convert data to string for regex search (using latin-1 to preserve bytes)
    try:
        text = data.decode('latin-1', errors='ignore')
    except Exception:
        text = ""
    
    for keyword in keywords:
        # Count occurrences using regex (case-sensitive)
        pattern = re.escape(keyword)
        matches = len(re.findall(pattern, text))
        counts[keyword] = matches
    
    return counts


def analyze_pdf_file(file_path):
    """
    Analyze a single PDF file and extract features.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        dict: Dictionary containing extracted features or None if error
    """
    try:
        # Read file in binary mode
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Get file size
        file_size = len(data)
        
        # Extract PDF version
        pdf_version = extract_pdf_version(data)
        
        # Calculate Shannon entropy
        entropy = calculate_shannon_entropy(data)
        
        # Count keywords
        keywords = ['/JS', '/JavaScript', '/AA', '/OpenAction', '/Launch', 
                   '/EmbeddedFile', '/URI', '/ObjStm']
        keyword_counts = count_keywords(data, keywords)
        
        # Prepare feature dictionary
        features = {
            'file_name': os.path.basename(file_path),
            'file_size': file_size,
            'pdf_version': pdf_version,
            'entropy': round(entropy, 6),
            'keyword_JS': keyword_counts['/JS'],
            'keyword_JavaScript': keyword_counts['/JavaScript'],
            'keyword_AA': keyword_counts['/AA'],
            'keyword_OpenAction': keyword_counts['/OpenAction'],
            'keyword_Launch': keyword_counts['/Launch'],
            'keyword_EmbeddedFile': keyword_counts['/EmbeddedFile'],
            'keyword_URI': keyword_counts['/URI'],
            'keyword_ObjStm': keyword_counts['/ObjStm']
        }
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def process_pdf_files(benign_folder, malicious_folder, output_csv):
    """
    Process all PDF files from both folders and save features to CSV.
    
    Args:
        benign_folder: Path to folder containing benign PDFs
        malicious_folder: Path to folder containing malicious PDFs
        output_csv: Path to output CSV file
    """
    all_features = []
    
    # Process benign PDFs (files ending with .pdf)
    print(f"Processing benign PDFs from: {benign_folder}")
    benign_path = Path(benign_folder)
    if benign_path.exists():
        benign_files = list(benign_path.glob('*.pdf'))
        print(f"Found {len(benign_files)} benign PDF files")
        
        for file_path in benign_files:
            features = analyze_pdf_file(file_path)
            if features:
                features['class'] = 0  # 0 for benign
                all_features.append(features)
    else:
        print(f"Warning: Benign folder not found: {benign_folder}")
    
    # Process malicious PDFs (files without extension or with unusual extensions)
    print(f"\nProcessing malicious PDFs from: {malicious_folder}")
    malicious_path = Path(malicious_folder)
    if malicious_path.exists():
        # Get all files (no extension filter, as malicious files don't have .pdf extension)
        all_files = [f for f in malicious_path.iterdir() if f.is_file()]
        print(f"Found {len(all_files)} malicious PDF files")
        
        for file_path in all_files:
            features = analyze_pdf_file(file_path)
            if features:
                features['class'] = 1  # 1 for malicious
                all_features.append(features)
    else:
        print(f"Warning: Malicious folder not found: {malicious_folder}")
    
    # Write to CSV
    if all_features:
        fieldnames = ['file_name', 'file_size', 'pdf_version', 'entropy',
                     'keyword_JS', 'keyword_JavaScript', 'keyword_AA',
                     'keyword_OpenAction', 'keyword_Launch', 'keyword_EmbeddedFile',
                     'keyword_URI', 'keyword_ObjStm', 'class']
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_features)
        
        print(f"\nSuccessfully processed {len(all_features)} files")
        print(f"Features saved to: {output_csv}")
    else:
        print("\nNo features extracted. Please check the input folders.")


def main():
    """Main function to run the PDF analysis."""
    # Define folder paths
    benign_folder = 'Data/Benign'
    malicious_folder = 'Data/Malicious'
    
    # Output CSV file (in the same directory as the script)
    script_dir = Path(__file__).parent
    output_csv = script_dir / 'pdf_features.csv'
    
    print("=" * 60)
    print("PDF Static Analysis for Machine Learning")
    print("=" * 60)
    print(f"Benign folder: {benign_folder}")
    print(f"Malicious folder: {malicious_folder}")
    print(f"Output file: {output_csv}")
    print("=" * 60)
    print()
    
    # Process files
    process_pdf_files(benign_folder, malicious_folder, output_csv)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

