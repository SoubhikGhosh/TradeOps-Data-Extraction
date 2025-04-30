# utils.py
import logging
import sys
from config import LOG_FILE, LOG_LEVEL

def setup_logger():
    """Configures and returns a logger."""
    logger = logging.getLogger("DocProcessor")
    logger.setLevel(LOG_LEVEL)

    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(LOG_LEVEL)

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, mode='a') # Append mode
    file_handler.setLevel(LOG_LEVEL)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(threadName)s - %(message)s'
    )
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add Handlers
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    return logger

# Get the logger instance
log = setup_logger()

def clean_filename(filename):
    """Removes problematic characters for file paths."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '-', '_')).rstrip()

def get_document_type_from_filename(filename):
    """Attempts to identify document type from filename patterns."""
    name_lower = filename.lower().split('.')[0] # Remove extension and lower case
    # Look for keywords, handle variations like INV, Invoice, CRL etc.
    if "crl" in name_lower:
        return "CRL"
    if "inv" in name_lower: # Catches Invoice, INV, inv3 etc.
        return "INVOICE"
    if "pack" in name_lower or "pkg" in name_lower or "packing" in name_lower:
        return "PACKING_LIST"
    if "bl" in name_lower or "bill of lading" in name_lower:
         return "BL"
    # Add more rules for other types (Insurance, COO, etc.)
    # Fallback or raise error if type cannot be determined
    log.warning(f"Could not determine document type for filename: {filename}")
    # Option 1: Return a default/unknown type
    # return "UNKNOWN"
    # Option 2: Raise an error
    raise ValueError(f"Could not determine document type for filename: {filename}")