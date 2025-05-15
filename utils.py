# utils.py
import logging
import sys
import re
import os
from pathlib import Path

# Import the centralized settings object
from config import settings

def setup_logger():
    """Configures and returns a logger based on settings."""
    logger = logging.getLogger("DocProcessor")
    # Use LOG_LEVEL from settings, converting string to logging level
    log_level_int = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(log_level_int)

    if logger.hasHandlers():
        logger.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level_int)

    # Use LOG_FILE from settings
    log_file_path = settings.LOG_FILE
    # Ensure log directory exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(log_level_int)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(processName)s - %(threadName)s - %(message)s'
    )
    stdout_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger

# Initialize logger (it will now use settings)
log = setup_logger()

def clean_filename(filename: str) -> str:
    """Removes problematic characters for file paths."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '-', '_')).rstrip()

def is_supported_file_type(filename: str) -> bool:
    """Checks if a file has a supported extension using settings."""
    _, ext = os.path.splitext(filename)
    return ext.lower() in [e.lower() for e in settings.SUPPORTED_FILE_EXTENSIONS]

def parse_filename_for_grouping(filename: str) -> tuple[str, int]:
    """
    Parses filename to extract a base name for grouping and a page number.
    Handles patterns like 'Name 1.pdf', 'Name_1.pdf', 'NamePage1.pdf', 'Name1.pdf', 'Name.pdf'
    Works with any supported file extension (PDF, PNG, JPEG)
    Returns: (base_name, page_number)
    """
    name_with_ext = Path(filename).name
    name_parts = name_with_ext.rsplit('.', 1)

    if len(name_parts) == 2:
        name_no_ext, _ = name_parts # ext is not used here
    else:
        name_no_ext = name_parts[0]

    page_number = 1  # Default page number
    base_name = name_no_ext  # Default base name

    # Regex Explanation:
    # (?:[ _]|Page)? : Optionally matches a separator (space, underscore, or 'Page'). Non-capturing group.
    # (\d+)          : Captures one or more digits (the page number).
    # $              : Anchors the match to the end of the string.
    match = re.search(r'(?:[ _]|Page)?(\d+)$', name_no_ext, re.IGNORECASE)

    if match:
        potential_page_number_str = match.group(1)
        potential_base_name = name_no_ext[:match.start()]

        if potential_base_name:
            page_number = int(potential_page_number_str)
            base_name = potential_base_name.strip(' _')
        # elif name_no_ext.isdigit(): # Edge case: filename is just digits like "1.pdf"
            # base_name = f"doc_{name_no_ext}" # Or keep as is, current logic handles it by default

    if not base_name: # Final safety check
        base_name = "unknown_doc"
        log.warning(f"Could not determine valid base name for '{filename}', using '{base_name}'.")

    log.debug(f"Parsed Filename: '{filename}' -> Base Name: '{base_name}', Page: {page_number}")
    return base_name, page_number
