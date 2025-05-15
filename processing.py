# processing.py
import os
import zipfile
import tempfile
import concurrent.futures
import re
import json
import random
import time
import traceback
import mimetypes
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Type

import pandas as pd
import pydantic # For pydantic.ValidationError

import vertexai
from vertexai.generative_models import GenerativeModel, Part
import google.api_core.exceptions as google_exceptions

# Import the centralized settings object
from config import settings
# Import Pydantic models for responses and data structures
from models import (
    BaseVertexResponse,
    VertexClassificationResponse,
    VertexExtractionResult,
    ExtractedFieldData,
    DocumentFieldDefinition
)
from utils import log, parse_filename_for_grouping

# --- Initialize Vertex AI ---
try:
    log.info(f"Initializing Vertex AI for project='{settings.GOOGLE_CLOUD_PROJECT}', location='{settings.LOCATION}'")
    vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.LOCATION)
    log.info("Vertex AI initialized successfully.")
    # Use MODEL_NAME from settings
    gemini_model = GenerativeModel(settings.MODEL_NAME, safety_settings=settings.SAFETY_SETTINGS_CONFIG)
    log.info(f"Loaded Vertex AI Model: {settings.MODEL_NAME}")
except Exception as e:
    log.exception(f"FATAL: Failed to initialize Vertex AI or load model: {e}")
    raise

# --- Helper Functions ---

def get_mime_type(file_path: Path) -> str:
    """Determine the MIME type of a file based on its extension or content."""
    file_ext = file_path.suffix.lower()
    if file_ext == '.pdf':
        return "application/pdf"
    elif file_ext == '.png':
        return "image/png"
    elif file_ext in ['.jpg', '.jpeg']:
        return "image/jpeg"

    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type:
        return mime_type

    log.warning(f"Could not determine mime type for {file_path}, defaulting to octet-stream")
    return "application/octet-stream"

def _call_vertex_ai_with_retry(
    model_instance: GenerativeModel,
    prompt_parts: List[Any],
    max_retries: int = 5,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> Any: # Returns the model's response object
    """Calls the Vertex AI model's generate_content method with exponential backoff."""
    num_retries = 0
    delay = initial_delay
    retryable_errors = (
        google_exceptions.ResourceExhausted,
        google_exceptions.TooManyRequests,
        google_exceptions.ServiceUnavailable,
        google_exceptions.DeadlineExceeded,
    )
    while True:
        try:
            log.debug(f"Attempting Vertex AI API call (Attempt {num_retries + 1}/{max_retries + 1})")
            response = model_instance.generate_content(prompt_parts, safety_settings=settings.SAFETY_SETTINGS_CONFIG)
            log.debug(f"Vertex AI API call successful (Attempt {num_retries + 1}/{max_retries + 1})")
            return response
        except retryable_errors as e:
            num_retries += 1
            if num_retries > max_retries:
                log.error(
                    f"Max retries ({max_retries}) exceeded for Vertex AI API call. "
                    f"Last error: {type(e).__name__} - {e}"
                )
                raise
            actual_delay = delay
            if jitter:
                actual_delay += random.uniform(0, delay * 0.25)
            log.warning(
                f"Vertex AI API call failed with {type(e).__name__} (Attempt {num_retries}/{max_retries}). "
                f"Retrying in {actual_delay:.2f} seconds..."
            )
            time.sleep(actual_delay)
            delay *= exponential_base
        except Exception as e:
            log.error(f"Non-retryable error during Vertex AI API call: {type(e).__name__} - {e}")
            log.error(traceback.format_exc())
            raise

def _prepare_document_parts(document_files: List[Dict[str, Any]]) -> Tuple[Optional[List[Part]], List[str]]:
    """Prepares Vertex AI Part objects from a list of document file paths."""
    parts = []
    file_paths_for_log = []
    document_files.sort(key=lambda x: x["page"]) # Sort by page number

    for file_info in document_files:
        file_path = file_info["path"] # Should be a Path object
        file_paths_for_log.append(file_path.name)
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()

            mime_type = get_mime_type(file_path)
            if mime_type not in settings.SUPPORTED_MIME_TYPES:
                log.warning(f"Unsupported file type: {mime_type} for file {file_path.name}")
                continue
            parts.append(Part.from_data(data=file_content, mime_type=mime_type))
        except FileNotFoundError:
            log.error(f"File not found during Vertex AI input prep: {file_path}")
            return None, file_paths_for_log
        except Exception as e:
            log.error(f"Error reading file {file_path}: {e}")
            return None, file_paths_for_log
    return parts, file_paths_for_log


def _parse_vertex_json_response(
    response: Any, context: str, model_type: Type[BaseVertexResponse]
) -> BaseVertexResponse:
    """
    Parses JSON response from Vertex AI into a Pydantic model.
    Handles errors and returns an instance of the specified model_type,
    with error fields populated if parsing fails or API indicates an issue.
    """
    raw_json_text = ""
    try:
        # Check for content blocking or empty parts
        if response.candidates and not response.candidates[0].content.parts:
            candidate = response.candidates[0]
            block_reason = str(candidate.finish_reason)
            # Convert safety ratings to a serializable format (dict of strings)
            safety_ratings_dict = {str(sr.category): str(sr.probability) for sr in candidate.safety_ratings}
            
            log.error(f"Content likely blocked for {context}. Reason: {block_reason}, Ratings: {safety_ratings_dict}")
            if model_type == VertexClassificationResponse:
                return VertexClassificationResponse(error=f"Content Blocked: {block_reason}", safety_ratings=safety_ratings_dict)
            elif model_type == VertexExtractionResult:
                 return VertexExtractionResult(error=f"Content Blocked: {block_reason}")
            return model_type(error=f"Content Blocked: {block_reason}")


        if not hasattr(response, 'text') or not response.text:
            log.error(f"Received empty or invalid response object for {context}. Response: {response}")
            return model_type(error="Empty or invalid response object from Vertex AI")

        raw_json_text = response.text.strip()
        if raw_json_text.startswith("```json"):
            raw_json_text = raw_json_text[7:-3].strip()
        elif raw_json_text.startswith("```"):
            raw_json_text = raw_json_text[3:-3].strip()
        
        # Pydantic will validate the structure based on the model_type
        parsed_model = model_type.model_validate_json(raw_json_text)
        log.debug(f"Successfully parsed and validated JSON response for {context} into {model_type.__name__}")
        return parsed_model

    except pydantic.ValidationError as val_err:
        log.error(f"Pydantic validation failed for {context} with {model_type.__name__}. Error: {val_err}")
        log.error(f"Raw Vertex AI Response Text for {context}:\n{raw_json_text}")
        return model_type(error=f"Pydantic Validation Error: {val_err!s}", raw_response=raw_json_text)
    except json.JSONDecodeError as json_err:
        log.error(f"Failed to decode JSON response from Vertex AI for {context}. Error: {json_err}")
        log.error(f"Raw Vertex AI Response Text for {context}:\n{raw_json_text}")
        return model_type(error=f"JSON Decode Error: {json_err!s}", raw_response=raw_json_text)
    except AttributeError as attr_err:
        log.error(f"Attribute error parsing response for {context}. Error: {attr_err}. Response: {response}")
        return model_type(error=f"AttributeError parsing response: {attr_err!s}")
    except Exception as e:
        log.exception(f"Unexpected error parsing Vertex AI response for {context}. Error: {e}")
        return model_type(error=f"Unexpected Parsing Error: {e!s}", raw_response=raw_json_text)


def _group_files_by_base_name(folder_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Groups document files in a folder by parsed base name and sorts by page number."""
    doc_groups = defaultdict(list)
    pattern_str = '|'.join([ext.replace('.', '\\.') for ext in settings.SUPPORTED_FILE_EXTENSIONS])
    supported_file_pattern = re.compile(f'.*({pattern_str})$', re.IGNORECASE)

    for doc_file_path in folder_path.glob('*'):
        if not doc_file_path.is_file() or not supported_file_pattern.match(doc_file_path.name):
            continue
        try:
            base_name, page_number = parse_filename_for_grouping(doc_file_path.name)
            doc_groups[base_name].append({"path": doc_file_path, "page": page_number})
        except Exception as e:
            log.warning(f"Error parsing filename {doc_file_path.name} in {folder_path.name}: {e}. Skipping file.")

    for base_name in doc_groups:
        doc_groups[base_name].sort(key=lambda x: x["page"])
    log.debug(f"Grouped files by base_name for {folder_path.name}: {{ {k}: len(v) for k, v in doc_groups.items()}}")
    return dict(doc_groups)

def _classify_document_type(case_id: str, base_name: str, document_files: List[Dict[str, Any]], acceptable_types: List[str]) -> VertexClassificationResponse:
    """Uses Vertex AI to classify the document type. Returns VertexClassificationResponse."""
    log.info(f"Starting classification for Case: {case_id}, Group: '{base_name}', Pages: {len(document_files)}")
    context = f"Case: {case_id}, Group: '{base_name}' (Classification)"

    if not document_files:
        log.warning(f"No document files provided for {context}")
        return VertexClassificationResponse(error="No document files provided")

    parts, _ = _prepare_document_parts(document_files)
    if parts is None:
        log.error(f"Failed to prepare document parts for {context}")
        return VertexClassificationResponse(error="Failed to prepare document parts")

    acceptable_types_str = "\n".join([f"- {atype}" for atype in acceptable_types])
    prompt = settings.CLASSIFICATION_PROMPT_TEMPLATE.format(
        num_pages=len(parts),
        acceptable_types_str=acceptable_types_str
    )
    log.debug(f"Generated classification prompt for {context}")

    try:
        log.info(f"Sending classification request to Vertex AI for {context}")
        full_request_content = [prompt] + parts
        response = _call_vertex_ai_with_retry(gemini_model, full_request_content)
        log.info(f"Received classification response from Vertex AI for {context}")
        
        classification_model = _parse_vertex_json_response(response, context, VertexClassificationResponse)
        return classification_model

    except google_exceptions.GoogleAPIError as api_err:
        log.exception(f"Vertex AI API Error during {context}. Error: {api_err}")
        return VertexClassificationResponse(error=f"Vertex AI API Error: {api_err!s}")
    except Exception as e:
        log.exception(f"Unexpected Error during {context}. Error: {e}")
        return VertexClassificationResponse(error=f"Unexpected Error: {e!s}")

def _extract_data_from_document(case_id: str, base_name: str, document_files: List[Dict[str, Any]], classified_doc_type: str, fields_to_extract: List[DocumentFieldDefinition]) -> VertexExtractionResult:
    """Uses Vertex AI Gemini model to extract data. Returns VertexExtractionResult."""
    log.info(f"Starting extraction for Case: {case_id}, Group: '{base_name}', Type: {classified_doc_type}, Pages: {len(document_files)}")
    context = f"Case: {case_id}, Group: '{base_name}', Type: {classified_doc_type} (Extraction)"

    if not document_files:
        log.warning(f"No document files provided for {context}")
        return VertexExtractionResult(error="No document files provided for extraction")
    if not fields_to_extract:
        log.warning(f"No fields defined for extraction for type {classified_doc_type} in {context}")
        return VertexExtractionResult(error=f"No fields defined for type {classified_doc_type}")

    parts, _ = _prepare_document_parts(document_files)
    if parts is None:
        log.error(f"Failed to prepare document parts for {context}")
        return VertexExtractionResult(error="Failed to prepare document parts for extraction")

    field_list_str = "\n".join([f"- **{field_def.name}**: {field_def.description}" for field_def in fields_to_extract])
    prompt = settings.EXTRACTION_PROMPT_TEMPLATE.format(
        doc_type=classified_doc_type,
        case_id=case_id,
        num_pages=len(parts),
        field_list_str=field_list_str
    )
    log.debug(f"Generated extraction prompt for {context}")

    try:
        log.info(f"Sending extraction request to Vertex AI for {context}")
        full_request_content = [prompt] + parts
        response = _call_vertex_ai_with_retry(gemini_model, full_request_content)
        log.info(f"Received extraction response from Vertex AI for {context}")
        
        extraction_model = _parse_vertex_json_response(response, context, VertexExtractionResult)
        return extraction_model

    except google_exceptions.GoogleAPIError as api_err:
        log.exception(f"Vertex AI API Error during {context}. Error: {api_err}")
        return VertexExtractionResult(error=f"Vertex AI API Error: {api_err!s}")
    except Exception as e:
        log.exception(f"Unexpected Error during {context}. Error: {e}")
        return VertexExtractionResult(error=f"Unexpected Error: {e!s}")


def process_zip_file(zip_file_path: str) -> str:
    """Main processing function using Pydantic models and centralized settings."""
    final_results_list = []
    # Use OUTPUT_FILENAME from settings
    output_excel_path = Path(settings.OUTPUT_FILENAME_STR) # Assuming it's relative to current dir or a defined output dir

    # Use TEMP_DIR from settings
    with tempfile.TemporaryDirectory(prefix="doc_proc_", dir=str(settings.TEMP_DIR)) as temp_dir_str:
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
            raise RuntimeError(f"Error extracting zip file: {e!s}")

        initial_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
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
                    "CASE_ID": case_id, "GROUP_Basename": "N/A",
                    "Processing_Status": "No processable document files found"
                })

        classification_tasks = []
        # Use DOCUMENT_FIELDS from settings
        acceptable_types = list(settings.DOCUMENT_FIELDS.keys()) + ["UNKNOWN"]

        for case_id, groups in initial_groups.items():
            for base_name, document_files in groups.items():
                if document_files:
                    classification_tasks.append((case_id, base_name, document_files, acceptable_types))
        
        # Store Pydantic model instances
        classification_results_map: Dict[Tuple[str, str], VertexClassificationResponse] = {}
        if classification_tasks:
            log.info(f"Submitting {len(classification_tasks)} document classification tasks to {settings.MAX_WORKERS} workers.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=settings.MAX_WORKERS, thread_name_prefix="Classifier") as executor:
                future_to_classify = {
                    executor.submit(_classify_document_type, *task_args): task_args[:2] # (case_id, base_name)
                    for task_args in classification_tasks
                }
                for future in concurrent.futures.as_completed(future_to_classify):
                    key_case_base = future_to_classify[future]
                    try:
                        result_model = future.result()
                        classification_results_map[key_case_base] = result_model
                    except Exception as exc:
                        log.exception(f"Error retrieving classification result for {key_case_base}. Error: {exc}")
                        classification_results_map[key_case_base] = VertexClassificationResponse(error=f"Task execution failed: {exc!s}")
        else:
            log.info("No classification tasks to submit.")

        extraction_tasks = []
        for (case_id, base_name), class_model in classification_results_map.items():
            if class_model.error:
                final_results_list.append({
                    "CASE_ID": case_id, "GROUP_Basename": base_name,
                    "Processing_Status": f"Classification Failed: {class_model.error}",
                    "CLASSIFIED_Type": class_model.classified_type, # Might be None
                    "CLASSIFICATION_Confidence": class_model.confidence, # Might be None
                    "CLASSIFICATION_Reasoning": class_model.reasoning, # Might be None
                })
                continue

            classified_type = class_model.classified_type
            if classified_type and classified_type != "UNKNOWN" and classified_type in settings.DOCUMENT_FIELDS:
                # Get List[DocumentFieldDefinition]
                fields_to_extract_defs = settings.DOCUMENT_FIELDS[classified_type]
                if fields_to_extract_defs:
                    document_files = initial_groups.get(case_id, {}).get(base_name)
                    if document_files:
                        extraction_tasks.append((case_id, base_name, document_files, classified_type, fields_to_extract_defs))
                    else:
                        log.error(f"Logic Error: Doc files not found for Case {case_id}, Group '{base_name}' for extraction.")
                else: # No fields configured for this valid type
                    final_results_list.append({
                        "CASE_ID": case_id, "GROUP_Basename": base_name,
                        "CLASSIFIED_Type": classified_type,
                        "CLASSIFICATION_Confidence": class_model.confidence,
                        "CLASSIFICATION_Reasoning": class_model.reasoning,
                        "Processing_Status": "Extraction skipped - No fields configured for this type"
                    })
            else: # UNKNOWN or unconfigured type
                status = f"Classification result: {classified_type or 'Not Classified'}"
                if classified_type == "UNKNOWN": status = "Classified as UNKNOWN"
                elif classified_type: status = f"Classified as '{classified_type}' (Unsupported/Not Configured for extraction)"
                final_results_list.append({
                    "CASE_ID": case_id, "GROUP_Basename": base_name,
                    "CLASSIFIED_Type": classified_type,
                    "CLASSIFICATION_Confidence": class_model.confidence,
                    "CLASSIFICATION_Reasoning": class_model.reasoning,
                    "Processing_Status": status
                })
        
        extraction_results_map: Dict[Tuple[str, str], VertexExtractionResult] = {}
        if extraction_tasks:
            log.info(f"Submitting {len(extraction_tasks)} document extraction tasks to {settings.MAX_WORKERS} workers.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=settings.MAX_WORKERS, thread_name_prefix="Extractor") as executor:
                future_to_extract = {
                    executor.submit(_extract_data_from_document, *task_args): task_args[:2]
                    for task_args in extraction_tasks
                }
                for future in concurrent.futures.as_completed(future_to_extract):
                    key_case_base = future_to_extract[future]
                    try:
                        result_model = future.result()
                        extraction_results_map[key_case_base] = result_model
                    except Exception as exc:
                        log.exception(f"Error retrieving extraction result for {key_case_base}. Error: {exc}")
                        extraction_results_map[key_case_base] = VertexExtractionResult(error=f"Task execution failed: {exc!s}")
        else:
            log.info("No extraction tasks to submit.")

        log.info("Aggregating final results...")
        for task_args_tuple in extraction_tasks: # Iterate through tasks submitted for extraction
            case_id, base_name, _, classified_type, fields_to_extract_defs = task_args_tuple
            key = (case_id, base_name)
            
            class_model = classification_results_map.get(key) # Should always exist if extraction was tasked
            extraction_model = extraction_results_map.get(key) # This is the one we just got

            row_data = {
                "CASE_ID": case_id,
                "GROUP_Basename": base_name,
                "CLASSIFIED_Type": classified_type, # From classification
                "CLASSIFICATION_Confidence": class_model.confidence if class_model else None,
                "CLASSIFICATION_Reasoning": class_model.reasoning if class_model else None
            }

            if extraction_model and not extraction_model.error and extraction_model.extracted_data:
                row_data["Processing_Status"] = "Extraction Successful"
                for field_name_key, field_data_model in extraction_model.extracted_data.items():
                    prefix = f"{classified_type}_{field_name_key}"
                    row_data[f"{prefix}_Value"] = field_data_model.value
                    row_data[f"{prefix}_Confidence"] = field_data_model.confidence
                    row_data[f"{prefix}_Reasoning"] = field_data_model.reasoning
            elif extraction_model and extraction_model.error:
                row_data["Processing_Status"] = f"Extraction Failed: {extraction_model.error}"
            elif not extraction_model: # Should not happen if task was in extraction_tasks
                 row_data["Processing_Status"] = "Extraction result missing (logic error)"
                 log.error(f"Missing extraction result for {key} which was in extraction_tasks.")
            else: # No extracted_data but no error (e.g. model returned empty dict)
                 row_data["Processing_Status"] = "Extraction Successful (No data fields returned by model)"


            final_results_list.append(row_data)

        if not final_results_list:
            log.warning("No data rows were generated for the Excel file.")
            df = pd.DataFrame([{"Status": "No data processed or extracted"}])
        else:
            log.info(f"Creating DataFrame from {len(final_results_list)} aggregated results.")
            df = pd.DataFrame(final_results_list)
            cols = df.columns.tolist()
            core_cols = ["CASE_ID", "GROUP_Basename", "Processing_Status", "CLASSIFIED_Type", "CLASSIFICATION_Confidence", "CLASSIFICATION_Reasoning"]
            ordered_cols = [c for c in core_cols if c in cols] # Ensure core_cols exist
            extracted_cols = sorted([c for c in cols if c not in ordered_cols])
            df = df[ordered_cols + extracted_cols]

        try:
            log.info(f"Saving aggregated data to Excel: {output_excel_path}")
            # Ensure output directory exists if output_excel_path includes directories
            output_excel_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(output_excel_path, index=False, engine='openpyxl')
            log.info("Excel file saved successfully.")
            return str(output_excel_path)
        except Exception as e:
            log.exception(f"Failed to save DataFrame to Excel file '{output_excel_path}': {e}")
            raise RuntimeError(f"Failed to save results to Excel: {e!s}")

    log.info("Temporary directory cleaned up.")
    # This return is only hit if the temp dir context fails, which shouldn't happen.
    # The actual return is inside the 'try' block for saving excel.
    return str(output_excel_path) 
