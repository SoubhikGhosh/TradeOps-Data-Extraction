# utils.py
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

def parse_filename_for_grouping(filename):
    """
    Parses filename to extract a base name for grouping and a page number.
    Handles patterns like 'Name 1.pdf', 'Name_1.pdf', 'NamePage1.pdf', 'Name1.pdf', 'Name.pdf'
    Returns: (base_name, page_number)
    """
    name_no_ext = filename.rsplit('.', 1)[0]
    page_number = 1 # Default page number
    base_name = name_no_ext # Default base name

    # Regex Explanation:
    # (?:[ _]|Page)? : Optionally matches a separator (space, underscore, or 'Page'). Non-capturing group.
    # (\d+)          : Captures one or more digits (the page number).
    # $              : Anchors the match to the end of the string.
    match = re.search(r'(?:[ _]|Page)?(\d+)$', name_no_ext, re.IGNORECASE)

    if match:
        potential_page_number_str = match.group(1)
        potential_base_name = name_no_ext[:match.start()]

        # Check if the part before the number is non-empty. Avoids classifying "1.pdf" as base="" page=1.
        # Also check if the base name itself ends with a number, which might indicate name1 vs name 1 pattern.
        if potential_base_name: # Ensure base name is not empty
             page_number = int(potential_page_number_str)
             base_name = potential_base_name.strip(' _') # Clean trailing separators
        # If potential_base_name is empty, it means the filename was just digits (e.g., "1.pdf")
        # Keep the default base_name = name_no_ext and page_number = 1 in this edge case,
        # or handle specifically if needed:
        # elif name_no_ext.isdigit():
        #    base_name = f"doc_{name_no_ext}" # Or keep as is
        #    page_number = int(name_no_ext)

    # If no page pattern matched, the defaults (full name_no_ext as base_name, page 1) are used.

    # Final safety check for empty base name
    if not base_name:
        base_name = "unknown_doc"
        log.warning(f"Could not determine valid base name for '{filename}', using '{base_name}'.")


    # --- DEBUG LOGGING ---
    # Uncomment the line below temporarily to see exactly how filenames are parsed
    log.debug(f"Parsed Filename: '{filename}' -> Base Name: '{base_name}', Page: {page_number}")
    # --- /DEBUG LOGGING ---

    return base_name, page_number