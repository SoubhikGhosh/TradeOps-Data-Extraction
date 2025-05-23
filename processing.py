# processing.py
import os
import zipfile
import tempfile
import concurrent.futures
import re
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import vertexai
from vertexai.generative_models import GenerativeModel, Part
import google.api_core.exceptions

from config import (
    PROJECT_ID, LOCATION, API_ENDPOINT, MODEL_NAME, SAFETY_SETTINGS,
    DOCUMENT_FIELDS, MAX_WORKERS, TEMP_DIR, OUTPUT_FILENAME,
    EXTRACTION_PROMPT_TEMPLATE, CLASSIFICATION_PROMPT_TEMPLATE # Import new template
)
from utils import log, parse_filename_for_grouping # Import new parsing function

import time
import random
from tenacity import (
    retry,
    stop_after_attempt, 
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)
import logging

RETRYABLE_EXCEPTIONS = (
    google.api_core.exceptions.ResourceExhausted,  # 429 Too Many Requests
    google.api_core.exceptions.ServiceUnavailable,  # 503 Service Unavailable
    google.api_core.exceptions.DeadlineExceeded,    # 504 Gateway Timeout
    google.api_core.exceptions.InternalServerError, # 500 Internal Server Error
    google.api_core.exceptions.Aborted,             # Aborted transaction
    google.api_core.exceptions.TransportError,      # Network connectivity issues
    google.api_core.exceptions.GatewayTimeout,      # Gateway timeout
    google.api_core.exceptions.TooManyRequests,     # Another form of rate limiting
    ConnectionError,                                # General connection issues
    TimeoutError                                    # Timeouts
)
def vertex_ai_retry_decorator(
    max_attempts=50, 
    min_wait_seconds=1, 
    max_wait_seconds=60, 
    jitter_max_seconds=10
):
    """
    Creates a retry decorator specifically for Vertex AI API calls.
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait_seconds: Minimum wait time between retries
        max_wait_seconds: Maximum wait time between retries
        jitter_max_seconds: Maximum jitter to add to wait time
    
    Returns:
        A retry decorator configured for Vertex AI API calls
    """
    return retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(
            multiplier=1, 
            min=min_wait_seconds, 
            max=max_wait_seconds,
            jitter=jitter_max_seconds
        ),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True
    )

@vertex_ai_retry_decorator()
def _make_vertex_call(model, content, generation_config, safety_settings):
    """Makes a Vertex AI API call with built-in retry logic."""
    return model.generate_content(
        content,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False
    )

# --- Initialize Vertex AI ---
try:
    log.info(f"Initializing Vertex AI for project='{PROJECT_ID}', location='{LOCATION}'")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    log.info("Vertex AI initialized successfully.")
    model = GenerativeModel(MODEL_NAME)
    log.info(f"Loaded Vertex AI Model: {MODEL_NAME}")
except Exception as e:
    log.exception(f"FATAL: Failed to initialize Vertex AI or load model: {e}")
    raise

PDF_MIME_TYPE = "application/pdf"

# --- Helper Functions ---

def _prepare_pdf_parts(pdf_files: List[Dict]) -> Tuple[List[Part], List[str]]:
    """Prepares Vertex AI Part objects from a list of PDF file paths."""
    parts = []
    file_paths_for_log = []
    # Sort by page number just in case
    pdf_files.sort(key=lambda x: x["page"])
    for file_info in pdf_files:
        pdf_path = file_info["path"]
        file_paths_for_log.append(pdf_path.name)
        try:
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            parts.append(Part.from_data(data=pdf_content, mime_type=PDF_MIME_TYPE))
        except FileNotFoundError:
            log.error(f"File not found during Vertex AI input prep: {pdf_path}")
            return None, file_paths_for_log # Return None for parts on error
        except Exception as e:
            log.error(f"Error reading file {pdf_path}: {e}")
            return None, file_paths_for_log # Return None for parts on error
    return parts, file_paths_for_log

def _parse_vertex_json_response(response: Any, context: str) -> Dict | None:
    """Parses JSON response from Vertex AI, handling potential errors."""
    try:
        # Handle cases where response might be blocked or have unexpected structure
        if not hasattr(response, 'text') or not response.text:
             if response.candidates and not response.candidates[0].content.parts:
                 block_reason = response.candidates[0].finish_reason
                 safety_ratings = response.candidates[0].safety_ratings
                 log.error(f"Content likely blocked for {context}. Reason: {block_reason}, Ratings: {safety_ratings}")
                 return {"error": f"Content Blocked: {block_reason}", "safety_ratings": str(safety_ratings)} # Make ratings serializable
             else:
                 log.error(f"Received empty or invalid response object for {context}. Response: {response}")
                 return {"error": "Empty or invalid response object"}

        # Strip potential markdown code fences ```json ... ``` if model adds them
        raw_json = response.text.strip()
        if raw_json.startswith("```json"):
             raw_json = raw_json[7:-3].strip() # Remove ```json and ```
        elif raw_json.startswith("```"): # Less common, just ```
            raw_json = raw_json[3:-3].strip()

        # Validate start and end characters for safety
        if not (raw_json.startswith('{') and raw_json.endswith('}')):
             log.error(f"Response for {context} is not valid JSON structure. Raw Text:\n{raw_json}")
             return {"error": "Invalid JSON structure", "raw_response": raw_json}


        parsed_data = json.loads(raw_json)
        log.debug(f"Successfully parsed JSON response for {context}")
        return parsed_data

    except json.JSONDecodeError as json_err:
        log.error(f"Failed to decode JSON response from Vertex AI for {context}. Error: {json_err}")
        log.error(f"Raw Vertex AI Response Text:\n{response.text}")
        return {"error": "JSON Decode Error", "raw_response": response.text}
    except AttributeError as attr_err:
         log.error(f"Attribute error parsing response for {context}. Error: {attr_err}. Response: {response}")
         return {"error": f"AttributeError parsing response: {attr_err}"}
    except Exception as e:
        log.exception(f"Unexpected error parsing Vertex AI response for {context}. Error: {e}")
        return {"error": f"Unexpected Parsing Error: {e}"}

# --- Stage 1: Grouping by Base Filename ---
def _group_files_by_base_name(folder_path: Path) -> Dict[str, List[Dict]]:
    """Groups PDF files in a folder by parsed base name and sorts by page number."""
    doc_groups = defaultdict(list)
    for pdf_file in folder_path.glob('*.pdf'):
        try:
            base_name, page_number = parse_filename_for_grouping(pdf_file.name)
            doc_groups[base_name].append({"path": pdf_file, "page": page_number})
        except Exception as e:
            log.warning(f"Error parsing filename {pdf_file.name} in {folder_path.name}: {e}. Skipping file.")

    # Sort pages within each document group
    for base_name in doc_groups:
        doc_groups[base_name].sort(key=lambda x: x["page"])

    log.debug(f"Grouped files by base_name for {folder_path.name}: { {k: len(v) for k, v in doc_groups.items()} }")
    return dict(doc_groups)

# --- Stage 2: Document Classification ---
def _classify_document_type(case_id: str, base_name: str, pdf_files: list, acceptable_types: list):
    """Uses Vertex AI to classify the document type from a list of PDF pages."""
    log.info(f"Starting classification for Case: {case_id}, Group: '{base_name}', Pages: {len(pdf_files)}")
    context = f"Case: {case_id}, Group: '{base_name}' (Classification)"

    if not pdf_files:
        log.warning(f"No PDF files provided for {context}")
        return {"error": "No PDF files provided"}

    parts, file_paths_for_log = _prepare_pdf_parts(pdf_files)
    if parts is None:
         log.error(f"Failed to prepare PDF parts for {context}")
         return {"error": "Failed to prepare PDF parts"}

    acceptable_types_str = "\n".join([f"- {atype}" for atype in acceptable_types])
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        num_pages=len(parts),
        acceptable_types_str=acceptable_types_str
    )
    log.debug(f"Generated classification prompt for {context}") # Prompt is less sensitive

    try:
        log.info(f"Sending classification request to Vertex AI for {context}")
        full_request_content = [prompt] + parts
        response = _make_vertex_call(
            model,
            full_request_content,
            generation_config={"response_mime_type": "application/json"},
            safety_settings=SAFETY_SETTINGS
        )
        log.info(f"Received classification response from Vertex AI for {context}")

        # Parse the JSON response
        classification_result = _parse_vertex_json_response(response, context)
        return classification_result # Will contain 'classified_type', 'confidence', 'reasoning' or 'error'

    except RetryError as retry_err:
        log.error(f"All retry attempts failed for {context}: {retry_err}")
        return {"error": f"All retry attempts failed: {retry_err.last_attempt.exception()}"}
    except google.api_core.exceptions.GoogleAPIError as api_err:
        log.exception(f"Vertex AI API Error during {context}. Error: {api_err}")
        return {"error": f"Vertex AI API Error: {api_err}"}
    except Exception as e:
        log.exception(f"Unexpected Error during {context}. Error: {e}")
        return {"error": f"Unexpected Error: {e}"}


# --- Stage 3: Data Extraction ---
def _extract_data_from_document(case_id: str, base_name: str, pdf_files: list, classified_doc_type: str, fields_to_extract: list):
    """Uses Vertex AI Gemini model to extract data for a *classified* document type."""
    log.info(f"Starting extraction for Case: {case_id}, Group: '{base_name}', Type: {classified_doc_type}, Pages: {len(pdf_files)}")
    context = f"Case: {case_id}, Group: '{base_name}', Type: {classified_doc_type} (Extraction)"

    if not pdf_files:
        log.warning(f"No PDF files provided for {context}")
        return {"error": "No PDF files provided for extraction"}
    if not fields_to_extract:
        log.warning(f"No fields defined for extraction for type {classified_doc_type} in {context}")
        return {"error": f"No fields defined for type {classified_doc_type}"}


    parts, file_paths_for_log = _prepare_pdf_parts(pdf_files)
    if parts is None:
         log.error(f"Failed to prepare PDF parts for {context}")
         return {"error": "Failed to prepare PDF parts for extraction"}

    # Prepare the field list string with descriptions for the extraction prompt
    field_list_str = "\n".join([f"- **{field_dict['name']}**: {field_dict['description']}" for field_dict in fields_to_extract])

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        # Note: Using classified_doc_type here, not base_name
        doc_type=classified_doc_type,
        case_id=case_id,
        num_pages=len(parts),
        field_list_str=field_list_str
    )
    log.debug(f"Generated extraction prompt for {context}") # Avoid logging full sensitive prompt if necessary

    try:
        log.info(f"Sending extraction request to Vertex AI for {context}")
        full_request_content = [prompt] + parts
        response = _make_vertex_call(
            model,
            full_request_content,
            generation_config={"response_mime_type": "application/json"},
            safety_settings=SAFETY_SETTINGS
        )
        log.info(f"Received extraction response from Vertex AI for {context}")

        # Parse the JSON response
        extracted_data = _parse_vertex_json_response(response, context)
        return extracted_data # Will contain field data or 'error'

    except RetryError as retry_err:
        log.error(f"All retry attempts failed for {context}: {retry_err}")
    except google.api_core.exceptions.GoogleAPIError as api_err:
        log.exception(f"Vertex AI API Error during {context}. Error: {api_err}")
        return {"error": f"Vertex AI API Error: {api_err}"}
    except Exception as e:
        log.exception(f"Unexpected Error during {context}. Error: {e}")
        return {"error": f"Unexpected Error: {e}"}

# --- Main Processing Function ---
def process_zip_file(zip_file_path: str):
    """
    Main function (Revised Workflow):
    1. Extracts zip.
    2. Groups files by base filename within each case.
    3. Classifies document type for each group using Vertex AI.
    4. Extracts data for successfully classified/supported types using Vertex AI.
    5. Aggregates results into a pandas DataFrame and saves to Excel.
    """
    final_results_list = [] # Store final row data here
    output_excel_path = Path(OUTPUT_FILENAME)

    with tempfile.TemporaryDirectory(prefix="doc_proc_", dir=TEMP_DIR) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        log.info(f"Created temporary directory: {temp_dir}")

        # --- 1. Extract Zip File ---
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            log.info(f"Successfully extracted '{zip_file_path}' to '{temp_dir}'")
        except zipfile.BadZipFile:
            log.error(f"Invalid zip file provided: {zip_file_path}")
            raise ValueError(f"Invalid zip file: {zip_file_path}")
        except Exception as e:
            log.exception(f"Error extracting zip file: {e}")
            raise

        # --- 2. Initial Grouping by Base Filename ---
        initial_groups = {} # {case_id: {base_name: [file_info_dict]}}
        case_folders = [d for d in temp_dir.iterdir() if d.is_dir()]
        if not case_folders:
             log.error(f"No case folders found in the extracted zip content at {temp_dir}")
             raise ValueError("No case folders found in the zip file.")

        for case_folder in case_folders:
            case_id = case_folder.name
            log.info(f"Performing initial file grouping for Case ID: {case_id}")
            initial_groups[case_id] = _group_files_by_base_name(case_folder)
            if not initial_groups[case_id]:
                 log.warning(f"No processable PDF groups found in case folder: {case_id}")
                 # Add a row indicating no docs found for this case
                 final_results_list.append({
                     "CASE_ID": case_id,
                     "GROUP_Basename": "N/A",
                     "Processing_Status": "No processable PDF files found"
                 })


        # --- 3. Classify Document Types Concurrently ---
        classification_tasks = []
        acceptable_types = list(DOCUMENT_FIELDS.keys()) # Get types we can potentially handle
        acceptable_types.append("UNKNOWN") # Allow UNKNOWN as a valid classification response

        for case_id, groups in initial_groups.items():
            for base_name, pdf_files in groups.items():
                 if pdf_files: # Only classify if there are files
                     classification_tasks.append((case_id, base_name, pdf_files, acceptable_types))

        classification_results = {} # {(case_id, base_name): classification_dict or error_dict}
        if classification_tasks:
            log.info(f"Submitting {len(classification_tasks)} document classification tasks to {MAX_WORKERS} workers.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="Classifier") as executor:
                future_to_classify = {
                    executor.submit(_classify_document_type, *task_args): task_args
                    for task_args in classification_tasks
                }
                for future in concurrent.futures.as_completed(future_to_classify):
                    case_id, base_name, _, _ = future_to_classify[future]
                    try:
                        result = future.result()
                        classification_results[(case_id, base_name)] = result
                    except Exception as exc:
                        log.exception(f"Error retrieving classification result for Case: {case_id}, Group: '{base_name}'. Error: {exc}")
                        classification_results[(case_id, base_name)] = {"error": f"Task execution failed: {exc}"}
        else:
             log.info("No classification tasks to submit.")


        # --- 4. Extract Data Concurrently (Based on Classification) ---
        extraction_tasks = []
        for (case_id, base_name), class_result in classification_results.items():
            if isinstance(class_result, dict) and "error" not in class_result:
                classified_type = class_result.get("classified_type")
                if classified_type and classified_type != "UNKNOWN" and classified_type in DOCUMENT_FIELDS:
                    fields_to_extract = DOCUMENT_FIELDS[classified_type]
                    if fields_to_extract: # Check if there are fields defined
                        # Retrieve the original pdf_files list for this group
                        pdf_files = initial_groups.get(case_id, {}).get(base_name)
                        if pdf_files:
                             extraction_tasks.append((case_id, base_name, pdf_files, classified_type, fields_to_extract))
                        else:
                             log.error(f"Logic Error: PDF files not found for Case {case_id}, Group '{base_name}' during extraction task prep.")
                    else:
                         log.warning(f"No fields configured for extraction for classified type '{classified_type}' in Case {case_id}, Group '{base_name}'.")
                         # Store classification result, but mark as no extraction fields
                         final_results_list.append({
                            "CASE_ID": case_id,
                            "GROUP_Basename": base_name,
                            "CLASSIFIED_Type": classified_type,
                            "CLASSIFICATION_Confidence": class_result.get('confidence'),
                            "CLASSIFICATION_Reasoning": class_result.get('reasoning'),
                            "Processing_Status": "Extraction skipped - No fields configured"
                         })

                else:
                     # Handle UNKNOWN or unconfigured types
                     status = f"Classification result: {classified_type or 'Not Classified'}"
                     if classified_type == "UNKNOWN": status = "Classified as UNKNOWN"
                     elif classified_type: status = f"Classified as '{classified_type}' (Unsupported/Not Configured)"

                     final_results_list.append({
                         "CASE_ID": case_id,
                         "GROUP_Basename": base_name,
                         "CLASSIFIED_Type": classified_type,
                         "CLASSIFICATION_Confidence": class_result.get('confidence'),
                         "CLASSIFICATION_Reasoning": class_result.get('reasoning'),
                         "Processing_Status": status
                     })
            else:
                 # Handle classification errors
                 error_msg = class_result.get('error', 'Unknown classification error') if isinstance(class_result, dict) else 'Invalid classification result'
                 final_results_list.append({
                    "CASE_ID": case_id,
                    "GROUP_Basename": base_name,
                    "Processing_Status": f"Classification Failed: {error_msg}"
                 })

        # Dictionary to hold extraction results, keyed by (case_id, base_name)
        extraction_results_map = {}
        if extraction_tasks:
            log.info(f"Submitting {len(extraction_tasks)} document extraction tasks to {MAX_WORKERS} workers.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="Extractor") as executor:
                future_to_extract = {
                    executor.submit(_extract_data_from_document, *task_args): task_args[:2] # Key by (case_id, base_name)
                    for task_args in extraction_tasks
                }
                for future in concurrent.futures.as_completed(future_to_extract):
                    key = future_to_extract[future] # (case_id, base_name)
                    try:
                        result = future.result()
                        extraction_results_map[key] = result
                    except Exception as exc:
                        log.exception(f"Error retrieving extraction result for key {key}. Error: {exc}")
                        extraction_results_map[key] = {"error": f"Task execution failed: {exc}"}
        else:
            log.info("No extraction tasks to submit.")

        # --- 5. Aggregate Results ---
        log.info("Aggregating final results...")
        for task_args in extraction_tasks:
            case_id, base_name, _, classified_type, fields_to_extract = task_args
            key = (case_id, base_name)
            extraction_result = extraction_results_map.get(key)
            class_result = classification_results.get(key, {}) # Get classification details too

            row_data = {
                "CASE_ID": case_id,
                "GROUP_Basename": base_name,
                "CLASSIFIED_Type": classified_type,
                "CLASSIFICATION_Confidence": class_result.get('confidence'),
                "CLASSIFICATION_Reasoning": class_result.get('reasoning')
            }

            if isinstance(extraction_result, dict) and "error" not in extraction_result:
                 row_data["Processing_Status"] = "Extraction Successful"
                 # Flatten the extracted data
                 for field_dict in fields_to_extract:
                     field_name = field_dict['name']
                     field_data = extraction_result.get(field_name)
                     # Prefix field names with CLASSIFIED type for clarity
                     prefix = f"{classified_type}_{field_name}"
                     if isinstance(field_data, dict):
                         row_data[f"{prefix}_Value"] = field_data.get('value')
                         row_data[f"{prefix}_Confidence"] = field_data.get('confidence')
                         row_data[f"{prefix}_Reasoning"] = field_data.get('reasoning')
                     else:
                          log.warning(f"Unexpected format for field '{field_name}' in extraction response for {key}. Data: {field_data}")
                          row_data[f"{prefix}_Raw"] = str(field_data) # Store raw if format incorrect
                          row_data["Processing_Status"] = "Extraction Partially Successful (Format Issue)"

            else:
                 # Handle extraction errors
                 error_msg = extraction_result.get('error', 'Unknown extraction error') if isinstance(extraction_result, dict) else 'Invalid extraction result'
                 row_data["Processing_Status"] = f"Extraction Failed: {error_msg}"

            final_results_list.append(row_data)


        # --- 6. Save to Excel ---
        if not final_results_list:
             log.warning("No data rows were generated for the Excel file.")
             df = pd.DataFrame([{"Status": "No data processed or extracted"}])
        else:
            log.info(f"Creating DataFrame from {len(final_results_list)} aggregated results.")
            df = pd.DataFrame(final_results_list)
            # Reorder columns: Case Info, Status, Classification Info, then Extracted Fields
            cols = df.columns.tolist()
            core_cols = ["CASE_ID", "GROUP_Basename", "Processing_Status", "CLASSIFIED_Type", "CLASSIFICATION_Confidence", "CLASSIFICATION_Reasoning"]
            # Ensure core cols exist and move them to the front
            ordered_cols = [c for c in core_cols if c in cols]
            extracted_cols = sorted([c for c in cols if c not in core_cols])
            df = df[ordered_cols + extracted_cols]

        try:
            log.info(f"Saving aggregated data to Excel: {output_excel_path}")
            df.to_excel(output_excel_path, index=False, engine='openpyxl')
            log.info("Excel file saved successfully.")
            return str(output_excel_path)
        except Exception as e:
            log.exception(f"Failed to save DataFrame to Excel file '{output_excel_path}': {e}")
            raise RuntimeError(f"Failed to save results to Excel: {e}")

    # End of `with tempfile.TemporaryDirectory`
    log.info("Temporary directory cleaned up.")