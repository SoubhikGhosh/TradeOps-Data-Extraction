import logging
import sys
import re # Keep re for filename parsing
from config import LOG_FILE, LOG_LEVEL

def setup_logger():
    """Configures and returns a logger."""
    logger = logging.getLogger("DocProcessor")
    logger.setLevel(LOG_LEVEL)
    if logger.hasHandlers():
        logger.handlers.clear()
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(LOG_LEVEL)
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    file_handler.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(threadName)s - %(message)s'
    )
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger

log = setup_logger()

def clean_filename(filename):
    """Removes problematic characters for file paths."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '-', '_')).rstrip()

# REMOVED: get_document_type_from_filename function is no longer used.

def parse_filename_for_grouping(filename):
    """
    Parses filename to extract a base name for grouping and a page number.
    Handles patterns like 'Name 1.pdf', 'Name_1.pdf', 'NamePage1.pdf', 'Name.pdf'
    Returns: (base_name, page_number)
    """
    name_no_ext = filename.rsplit('.', 1)[0]
    page_number = 1 # Default page number

    # Try to find common page number patterns at the end
    # Pattern: (space or underscore or 'Page' optionally) followed by digits
    match = re.search(r'([ _]|Page)(\d+)$', name_no_ext, re.IGNORECASE)
    if match:
        page_number = int(match.group(2))
        # Base name is the part before the matched pattern
        base_name = name_no_ext[:match.start()]
    else:
        # No clear page number pattern found, assume page 1 and use the whole name (without ext) as base
        base_name = name_no_ext

    # Basic cleaning of base name (remove trailing spaces/underscores)
    base_name = base_name.strip(' _')

    # Handle case where filename might *only* be a number (e.g., "1.pdf") - less likely but possible
    if not base_name and name_no_ext.isdigit():
         base_name = f"doc_{name_no_ext}" # Assign a generic base name
         page_number = int(name_no_ext)
    elif not base_name: # If somehow base_name is empty
         base_name = "unknown_doc"

    log.debug(f"Parsed '{filename}': base_name='{base_name}', page_number={page_number}")
    return base_name, page_number