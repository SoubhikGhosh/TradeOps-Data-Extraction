# processing.py
import os
import zipfile
import tempfile
import concurrent.futures
import re
import json
import json5
import random
import time
import traceback
import mimetypes
from pathlib import Path
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Any # 'Any' is already here, which is great

# --- Use the google-generativeai library ---
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions

from config import (
    GEMINI_API_KEY, MODEL_NAME, SAFETY_SETTINGS,
    DOCUMENT_FIELDS, MAX_WORKERS, TEMP_DIR, OUTPUT_FILENAME,
    EXTRACTION_PROMPT_TEMPLATE, CLASSIFICATION_PROMPT_TEMPLATE,
    SUPPORTED_MIME_TYPES, SUPPORTED_FILE_EXTENSIONS, EXCEL_COLUMN_ORDER,
    DEFAULT_CONFIDENCE_THRESHOLD, EXTRACTION_MAX_ATTEMPTS
)
from utils import log, parse_filename_for_grouping

# --- Initialize Gemini API ---
try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set. Please get a key from Google AI Studio.")
    genai.configure(api_key=GEMINI_API_KEY)
    log.info("Gemini API configured successfully.")
    model = genai.GenerativeModel(MODEL_NAME)
    log.info(f"Loaded Gemini API Model: {MODEL_NAME}")
except Exception as e:
    log.exception(f"FATAL: Failed to configure Gemini API or load model: {e}")
    raise

# --- Helper Functions (Updated) ---

def get_mime_type(file_path):
    """Determine the MIME type of a file based on its extension or content."""
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext in ['.pdf']:
        return "application/pdf"
    elif file_ext in ['.png']:
        return "image/png"
    elif file_ext in ['.jpg', '.jpeg']:
        return "image/jpeg"

    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type

    log.warning(f"Could not determine mime type for {file_path}, defaulting to octet-stream")
    return "application/octet-stream"


generation_config = genai.types.GenerationConfig(
    temperature=0.05 # A very low value for deterministic output
)

def _call_gemini_with_retry(
    model_instance: genai.GenerativeModel,
    prompt_parts: List[Any],
    max_retries: int = 5,
    initial_delay: float = 0.5,
    exponential_base: float = 2.0,
    jitter: bool = True
): # <--- FIX #1: REMOVED THE RETURN TYPE HINT
    """
    Calls the Gemini model's generate_content method with exponential backoff.
    """
    num_retries = 0
    delay = initial_delay
    retryable_errors = (
        google_exceptions.ResourceExhausted,
        google_exceptions.TooManyRequests,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
        google_exceptions.InternalServerError,
    )
    while True:
        try:
            log.debug(f"Attempting Gemini API call (Attempt {num_retries + 1}/{max_retries + 1})")
            response = model_instance.generate_content(prompt_parts, generation_config=generation_config, safety_settings=SAFETY_SETTINGS)
            log.debug(f"Gemini API call successful (Attempt {num_retries + 1}/{max_retries + 1})")
            return response
        except retryable_errors as e:
            num_retries += 1
            if num_retries > max_retries:
                log.error(f"Max retries ({max_retries}) exceeded. Last error: {type(e).__name__} - {e}")
                raise
            actual_delay = delay + (random.uniform(0, delay * 0.25) if jitter else 0)
            log.warning(f"API call failed with {type(e).__name__}. Retrying in {actual_delay:.2f}s...")
            time.sleep(actual_delay)
            delay *= exponential_base
        except Exception as e:
            log.error(f"Non-retryable error during Gemini API call: {type(e).__name__} - {e}")
            log.error(traceback.format_exc())
            raise


def _prepare_document_parts(document_files: List[Dict]) -> Tuple[List[Dict], List[str]]:
    """Prepares Gemini API parts (as dictionaries) from a list of document file paths."""
    parts = []
    file_paths_for_log = []
    document_files.sort(key=lambda x: x["page"])
    for file_info in document_files:
        file_path = file_info["path"]
        file_paths_for_log.append(file_path.name)
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
            mime_type = get_mime_type(file_path)
            if mime_type not in SUPPORTED_MIME_TYPES:
                log.warning(f"Unsupported file type: {mime_type} for file {file_path}")
                continue
            parts.append({'mime_type': mime_type, 'data': file_content})
        except FileNotFoundError:
            log.error(f"File not found during input prep: {file_path}")
            return None, file_paths_for_log
        except Exception as e:
            log.error(f"Error reading file {file_path}: {e}")
            return None, file_paths_for_log
    return parts, file_paths_for_log


def _parse_gemini_json_response(response: Any, context: str) -> Dict: # <--- FIX #2: CHANGED TYPE HINT TO Any
    """
    Parses JSON-like response from the Gemini API, handling potential errors and
    allowing for more lenient JSON5 syntax.
    """
    processed_text: str = ""
    try:
        # Check for content blocking first.
        if not response.parts:
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason.name
                safety_ratings_val = response.prompt_feedback.safety_ratings
                log.error(f"Content blocked for {context}. Reason: {block_reason}, Ratings: {safety_ratings_val}")
                return {"error": f"Content Blocked: {block_reason}", "safety_ratings": str(safety_ratings_val)}

        raw_text = response.text.strip()

        if raw_text.startswith("```json"):
            processed_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            processed_text = raw_text[3:-3].strip()
        else:
            processed_text = raw_text

        if not processed_text:
            log.error(f"Response text became empty after stripping markdown for {context}. Original: {raw_text}")
            return {"error": "Empty content after stripping markdown", "raw_response": raw_text}

        if not (processed_text.startswith('{') and processed_text.endswith('}')):
            log.warning(
                f"Response for {context} does not strictly start/end with '{{}}'. "
                f"Attempting to parse with json5 anyway. Processed Text:\n{processed_text}"
            )
            
        parsed_data = json5.loads(processed_text)

        if not isinstance(parsed_data, dict):
            log.error(f"Parsed data for {context} is not a dictionary. Type: {type(parsed_data)}.")
            return {"error": "Parsed JSON is not a dictionary", "type": str(type(parsed_data)), "raw_response": processed_text}

        log.debug(f"Successfully parsed JSON5 response for {context}")
        return parsed_data

    except ValueError as val_err:
        log.error(f"Failed to decode JSON5 response for {context}. Error: {val_err}")
        log.error(f"Processed Text that failed parsing:\n{processed_text}")
        return {"error": "JSON5 Decode Error", "details": str(val_err), "raw_response": processed_text}
    except Exception as e:
        log.exception(f"Unexpected error parsing Gemini API response for {context}. Error: {e}")
        current_text_to_log = processed_text if processed_text else getattr(response, 'text', str(response))
        return {"error": f"Unexpected Parsing Error: {str(e)}", "raw_response": current_text_to_log}

# The rest of the file has no changes.

# --- Stage 1: Grouping by Base Filename ---
def _group_files_by_base_name(folder_path: Path) -> Dict[str, List[Dict]]:
    doc_groups = defaultdict(list)
    pattern = '|'.join([ext.replace('.', '\\.') for ext in SUPPORTED_FILE_EXTENSIONS])
    supported_file_pattern = f'.*({pattern})$'
    for doc_file in folder_path.glob('*'):
        if not doc_file.is_file() or not re.match(supported_file_pattern, doc_file.name, re.IGNORECASE):
            continue
        try:
            base_name, page_number = parse_filename_for_grouping(doc_file.name)
            doc_groups[base_name].append({"path": doc_file, "page": page_number})
        except Exception as e:
            log.warning(f"Error parsing filename {doc_file.name} in {folder_path.name}: {e}. Skipping file.")
    for base_name in doc_groups:
        doc_groups[base_name].sort(key=lambda x: x["page"])
    log.debug(f"Grouped files by base_name for {folder_path.name}: { {k: len(v) for k, v in doc_groups.items()} }")
    return dict(doc_groups)

def _classify_document_type(case_id: str, base_name: str, document_files: list, acceptable_types: list):
    log.info(f"Starting classification for Case: {case_id}, Group: '{base_name}', Pages: {len(document_files)}")
    context = f"Case: {case_id}, Group: '{base_name}' (Classification)"
    if not document_files:
        log.warning(f"No document files provided for {context}")
        return {"error": "No document files provided"}
    parts, file_paths_for_log = _prepare_document_parts(document_files)
    if parts is None:
           log.error(f"Failed to prepare document parts for {context}")
           return {"error": "Failed to prepare document parts"}
    acceptable_types_str = "\n".join([f"- {atype}" for atype in acceptable_types])
    prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(
        num_pages=len(parts),
        acceptable_types_str=acceptable_types_str
    )
    log.debug(f"Generated classification prompt for {context}")
    try:
        log.info(f"Sending classification request to Gemini API for {context}")
        full_request_content = [prompt] + parts
        response = _call_gemini_with_retry(
            model_instance=model,
            prompt_parts=full_request_content,
            max_retries=5,
            initial_delay=1.0
        )
        log.info(f"Received classification response from Gemini API for {context}")
        classification_result = _parse_gemini_json_response(response, context)
        return classification_result
    except google_exceptions.GoogleAPIError as api_err:
        log.exception(f"Gemini API Error during {context}. Error: {api_err}")
        return {"error": f"Gemini API Error: {api_err}"}
    except Exception as e:
        log.exception(f"Unexpected Error during {context}. Error: {e}")
        return {"error": f"Unexpected Error: {e}"}


# --- Stage 3: Data Extraction ---
def _extract_data_from_document(
    case_id: str,
    base_name: str,
    document_files: list,
    classified_doc_type: str,
    fields_to_extract: list,
    max_attempts: int = EXTRACTION_MAX_ATTEMPTS,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    log.info(f"Starting extraction for Case: {case_id}, Group: '{base_name}', Type: {classified_doc_type}, Pages: {len(document_files)}")
    context = f"Case: {case_id}, Group: '{base_name}', Type: {classified_doc_type} (Extraction)"
    if not document_files:
        log.warning(f"No document files provided for {context}")
        return {"error": "No document files provided for extraction"}
    if not fields_to_extract:
        log.warning(f"No fields defined for extraction for type {classified_doc_type} in {context}")
        return {"error": f"No fields defined for type {classified_doc_type}"}
    parts, file_paths_for_log = _prepare_document_parts(document_files)
    if parts is None:
           log.error(f"Failed to prepare document parts for {context}")
           return {"error": "Failed to prepare document parts for extraction"}
    field_list_str = "\n".join([f"- **{field_dict['name']}**: {field_dict['description']}" for field_dict in fields_to_extract])
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        doc_type=classified_doc_type,
        case_id=case_id,
        num_pages=len(parts),
        field_list_str=field_list_str
    )
    log.debug(f"Generated extraction prompt for {context}")
    current_attempt = 0
    best_field_extractions: Dict[str, Dict[str, Any]] = {}
    while current_attempt < max_attempts:
        log.info(f"Extraction attempt {current_attempt + 1}/{max_attempts} for {context}")
        try:
            full_request_content = [prompt] + parts
            response = _call_gemini_with_retry(
                model_instance=model,
                prompt_parts=full_request_content,
                max_retries=5,
                initial_delay=1.0
            )
            extracted_data = _parse_gemini_json_response(response, context)
            if "error" in extracted_data:
                log.warning(f"Extraction attempt {current_attempt + 1} failed with parsing/API error for {context}: {extracted_data.get('error')}")
                pass
            else:
                all_fields_confident_enough = True
                for field_dict in fields_to_extract:
                    field_name = field_dict['name']
                    field_data_from_this_attempt = extracted_data.get(field_name, {})
                    value = field_data_from_this_attempt.get('value', None)
                    confidence = field_data_from_this_attempt.get('confidence', 0.0)
                    if value is not None and (field_name not in best_field_extractions or confidence > best_field_extractions[field_name].get('confidence', -1.0)):
                        best_field_extractions[field_name] = {
                            "value": value,
                            "confidence": confidence,
                            "reasoning": field_data_from_this_attempt.get('reasoning')
                        }
                    if confidence < confidence_threshold and value is not None and value != "null":
                        log.warning(f"Field '{field_name}' extracted with low confidence ({confidence:.2f} < {confidence_threshold:.2f}) for {context} on attempt {current_attempt + 1}. Value: {value}")
                        all_fields_confident_enough = False
                    elif value is None or value == "null":
                        log.warning(f"Field '{field_name}' extracted as null/None for {context} on attempt {current_attempt + 1}. Attempting re-extraction.")
                        all_fields_confident_enough = False
                if all_fields_confident_enough:
                    log.info(f"All required fields extracted with sufficient confidence for {context} on attempt {current_attempt + 1}.")
                    break
                else:
                    log.info(f"Some fields require re-extraction for {context}. Re-attempting...")
        except google_exceptions.GoogleAPIError as api_err:
            log.exception(f"Gemini API Error during {context} on attempt {current_attempt + 1}. Error: {api_err}")
            best_field_extractions["_overall_status"] = {"error": f"Persistent Gemini API Error: {api_err}"}
            break
        except Exception as e:
            log.exception(f"Unexpected Error during {context} on attempt {current_attempt + 1}. Error: {e}")
            best_field_extractions["_overall_status"] = {"error": f"Unexpected Extraction Error: {e}"}
            break
        current_attempt += 1
        if current_attempt < max_attempts:
            time.sleep(random.uniform(0.5, 2.0))
    final_extraction_results = {}
    if "_overall_status" in best_field_extractions:
        return best_field_extractions["_overall_status"]
    for field_dict in fields_to_extract:
        field_name = field_dict['name']
        extracted_info = best_field_extractions.get(field_name, {"value": "null", "confidence": 0.0, "reasoning": "Not found or low confidence after attempts"})
        final_extraction_results[field_name] = {
            "value": extracted_info["value"],
            "confidence": extracted_info["confidence"],
            "reasoning": extracted_info.get("reasoning", "N/A")
        }
    final_extraction_results["_extraction_status"] = "Success"
    for field_name, info in final_extraction_results.items():
        if field_name.startswith("_"): continue
        if info["value"] == "null" or info["confidence"] < confidence_threshold:
            final_extraction_results["_extraction_status"] = "Partial Success (Low Confidence/Missing Fields)"
            break
    log.info(f"Finished extraction attempts for {context}. Status: {final_extraction_results.get('_extraction_status', 'Unknown')}")
    return final_extraction_results


MAX_EXCEL_CELL_LENGTH = 32700
TRUNCATION_ELLIPSIS = "..."

def sanitize_excel_string(text):
    if not isinstance(text, str):
        return text
    try:
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    except TypeError:
        return text
    if len(text) > MAX_EXCEL_CELL_LENGTH:
        text = text[:MAX_EXCEL_CELL_LENGTH - len(TRUNCATION_ELLIPSIS)] + TRUNCATION_ELLIPSIS
    return text

# --- Main Processing Function ---
def process_zip_file(zip_file_path: str):
    final_results_list = []
    output_excel_path = Path(OUTPUT_FILENAME)
    start_time = time.time()
    with tempfile.TemporaryDirectory(prefix="doc_proc_", dir=TEMP_DIR) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        log.info(f"Created temporary directory: {temp_dir}")
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
        initial_groups = {}
        case_folders = [d for d in temp_dir.iterdir() if d.is_dir()]
        if not case_folders:
               log.error(f"No case folders found in the extracted zip content at {temp_dir}")
               raise ValueError("No case folders found in the zip file.")
        for case_folder in case_folders:
            case_id = case_folder.name
            log.info(f"Performing initial file grouping for Case ID: {case_id}")
            initial_groups[case_id] = _group_files_by_base_name(case_folder)
            if not initial_groups[case_id]:
                   log.warning(f"No processable document groups found in case folder: {case_id}")
                   final_results_list.append({
                       "CASE_ID": case_id,
                       "GROUP_Basename": "N/A",
                       "Processing_Status": "No processable document files found"
                   })
        classification_tasks = []
        acceptable_types = list(DOCUMENT_FIELDS.keys())
        acceptable_types.append("UNKNOWN")
        for case_id, groups in initial_groups.items():
            for base_name, document_files in groups.items():
                   if document_files:
                       classification_tasks.append((case_id, base_name, document_files, acceptable_types))
        classification_results = {}
        if classification_tasks:
            log.info(f"Submitting {len(classification_tasks)} classification tasks to {MAX_WORKERS} workers.")
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
        extraction_tasks = []
        for (case_id, base_name), class_result in classification_results.items():
            if isinstance(class_result, dict) and "error" not in class_result:
                classified_type = class_result.get("classified_type")
                if classified_type and classified_type != "UNKNOWN" and classified_type in DOCUMENT_FIELDS:
                    fields_to_extract = DOCUMENT_FIELDS[classified_type]
                    if fields_to_extract:
                        document_files = initial_groups.get(case_id, {}).get(base_name)
                        if document_files:
                               extraction_tasks.append((case_id, base_name, document_files, classified_type, fields_to_extract))
                        else:
                               log.error(f"Logic Error: Document files not found for Case {case_id}, Group '{base_name}' during extraction task prep.")
                    else:
                               log.warning(f"No fields configured for extraction for classified type '{classified_type}' in Case {case_id}, Group '{base_name}'.")
                               final_results_list.append({
                                   "CASE_ID": case_id,
                                   "GROUP_Basename": base_name,
                                   "CLASSIFIED_Type": classified_type,
                                   "CLASSIFICATION_Confidence": class_result.get('confidence'),
                                   "CLASSIFICATION_Reasoning": class_result.get('reasoning'),
                                   "Processing_Status": "Extraction skipped - No fields configured"
                               })
                else:
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
                   error_msg = class_result.get('error', 'Unknown classification error') if isinstance(class_result, dict) else 'Invalid classification result'
                   final_results_list.append({
                       "CASE_ID": case_id,
                       "GROUP_Basename": base_name,
                       "Processing_Status": f"Classification Failed: {error_msg}"
                   })
        extraction_results_map = {}
        if extraction_tasks:
            log.info(f"Submitting {len(extraction_tasks)} extraction tasks to {MAX_WORKERS} workers.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="Extractor") as executor:
                future_to_extract = {
                    executor.submit(_extract_data_from_document, *task_args): task_args[:2]
                    for task_args in extraction_tasks
                }
                for future in concurrent.futures.as_completed(future_to_extract):
                    key = future_to_extract[future]
                    try:
                        result = future.result()
                        extraction_results_map[key] = result
                    except Exception as exc:
                        log.exception(f"Error retrieving extraction result for key {key}. Error: {exc}")
                        extraction_results_map[key] = {"error": f"Task execution failed: {exc}"}
        else:
            log.info("No extraction tasks to submit.")
        log.info("Aggregating final results...")
        for task_args in extraction_tasks:
            case_id, base_name, _, classified_type, fields_to_extract = task_args
            key = (case_id, base_name)
            extraction_result = extraction_results_map.get(key)
            class_result = classification_results.get(key, {})
            row_data = {
                "CASE_ID": case_id,
                "GROUP_Basename": base_name,
                "CLASSIFIED_Type": classified_type,
                "CLASSIFICATION_Confidence": class_result.get('confidence'),
                "CLASSIFICATION_Reasoning": class_result.get('reasoning')
            }
            if isinstance(extraction_result, dict) and "error" not in extraction_result:
                row_data["Processing_Status"] = extraction_result.get("_extraction_status", "Extraction Successful")
                for field_dict in fields_to_extract:
                    field_name = field_dict['name']
                    field_data = extraction_result.get(field_name)
                    prefix = f"{classified_type}_{field_name}"
                    if isinstance(field_data, dict):
                        row_data[f"{prefix}_Value"] = field_data.get('value')
                        row_data[f"{prefix}_Confidence"] = field_data.get('confidence')
                        row_data[f"{prefix}_Reasoning"] = field_data.get('reasoning')
                    else:
                         log.warning(f"Unexpected format for field '{field_name}' in extraction response for {key}. Data: {field_data}")
                         row_data[f"{prefix}_Value"] = str(field_data)
                         row_data[f"{prefix}_Confidence"] = 0.0
                         row_data[f"{prefix}_Reasoning"] = "N/A - Format issue"
                         row_data["Processing_Status"] = "Extraction Partially Successful (Format Issue)"
            else:
                   error_msg = extraction_result.get('error', 'Unknown extraction error') if isinstance(extraction_result, dict) else 'Invalid extraction. Result was not a dictionary.'
                   row_data["Processing_Status"] = f"Extraction Failed: {error_msg}"
            final_results_list.append(row_data)
        if not final_results_list:
               log.warning("No data rows were generated for the Excel file.")
               df = pd.DataFrame([{"Status": "No data processed or extracted"}])
        else:
            log.info(f"Creating DataFrame from {len(final_results_list)} aggregated results.")
            df = pd.DataFrame(final_results_list)
            log.info("Sanitizing DataFrame content for Excel compatibility...")
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).apply(sanitize_excel_string)
            existing_cols = df.columns.tolist()
            ordered_cols = [col for col in EXCEL_COLUMN_ORDER if col in existing_cols]
            remaining_cols = sorted([col for col in existing_cols if col not in ordered_cols])
            final_cols = ordered_cols + remaining_cols
            df = df[final_cols]
        try:
            log.info(f"Saving aggregated data to Excel: {output_excel_path}")
            df.to_excel(output_excel_path, index=False, engine='openpyxl')
            log.info("Excel file saved successfully.")
            elapsed_time = time.time() - start_time
            log.info(f"Total processing time for {zip_file_path}: {elapsed_time:.2f} seconds.")
            return str(output_excel_path)
        except Exception as e:
            log.exception(f"Failed to save DataFrame to Excel file '{output_excel_path}': {e}")
            elapsed_time = time.time() - start_time
            log.error(f"Processing failed after {elapsed_time:.2f} seconds while saving Excel.")
            raise RuntimeError(f"Failed to save results to Excel: {e}")
    log.info("Temporary directory cleaned up.")