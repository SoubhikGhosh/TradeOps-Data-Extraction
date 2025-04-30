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

import vertexai
from vertexai.generative_models import GenerativeModel, Part
import google.api_core.exceptions


from config import (
    PROJECT_ID, LOCATION, API_ENDPOINT, MODEL_NAME, SAFETY_SETTINGS,
    DOCUMENT_FIELDS, MAX_WORKERS, TEMP_DIR, OUTPUT_FILENAME,
    EXTRACTION_PROMPT_TEMPLATE
)
from utils import log, get_document_type_from_filename

# --- Initialize Vertex AI ---
try:
    log.info(f"Initializing Vertex AI for project='{PROJECT_ID}', location='{LOCATION}'")
    vertexai.init(project=PROJECT_ID, location=LOCATION) # api_endpoint=API_ENDPOINT - often not needed
    log.info("Vertex AI initialized successfully.")
    # Load the generative model
    model = GenerativeModel(MODEL_NAME)
    log.info(f"Loaded Vertex AI Model: {MODEL_NAME}")
except Exception as e:
    log.exception(f"FATAL: Failed to initialize Vertex AI or load model: {e}")
    raise

# --- PDF Mime Type ---
PDF_MIME_TYPE = "application/pdf"

# --- Helper Functions ---
def _group_files_by_doc_type(folder_path: Path):
    """Groups PDF files in a folder by detected document type and sorts by page number."""
    doc_groups = defaultdict(list)
    for pdf_file in folder_path.glob('*.pdf'):
        try:
            # Basic page number extraction (e.g., "CRL 1.pdf", "Invoice_page_2.pdf")
            match = re.search(r'[ _](\d+)\.pdf$', pdf_file.name, re.IGNORECASE)
            page_number = int(match.group(1)) if match else 1 # Default to 1 if no number

            # Get base name for type detection (e.g., "CRL", "Invoice")
            base_name = re.sub(r'[ _]?\d+\.pdf$', '', pdf_file.name, flags=re.IGNORECASE)
            doc_type = get_document_type_from_filename(base_name) # Use refined function

            doc_groups[doc_type].append({"path": pdf_file, "page": page_number})

        except ValueError as e:
             log.warning(f"Skipping file due to error in processing name '{pdf_file.name}': {e}")
        except Exception as e:
            log.warning(f"Error processing file {pdf_file.name}: {e}")

    # Sort pages within each document group
    for doc_type in doc_groups:
        doc_groups[doc_type].sort(key=lambda x: x["page"])

    log.debug(f"Grouped documents for {folder_path.name}: { {k: len(v) for k, v in doc_groups.items()} }")
    return dict(doc_groups)


def _extract_data_from_document(case_id: str, doc_type: str, pdf_files: list, fields_to_extract: list):
    """
    Uses Vertex AI Gemini model to extract data from a list of PDF files (pages).
    """
    log.info(f"Starting extraction for Case: {case_id}, Document Type: {doc_type}, Pages: {len(pdf_files)}")

    if not pdf_files:
        log.warning(f"No PDF files provided for Case: {case_id}, Document Type: {doc_type}")
        return None

    # --- Prepare input for Vertex AI Model ---
    # Create Part objects for each PDF page. Assumes model can handle PDF MIME type.
    # Sort pages just in case they weren't already
    pdf_files.sort(key=lambda x: x["page"])
    parts = []
    file_paths_for_log = []
    for file_info in pdf_files:
        pdf_path = file_info["path"]
        file_paths_for_log.append(pdf_path.name)
        try:
            with open(pdf_path, "rb") as f:
                pdf_content = f.read()
            parts.append(Part.from_data(data=pdf_content, mime_type=PDF_MIME_TYPE))
        except FileNotFoundError:
            log.error(f"File not found during Vertex AI input prep: {pdf_path}")
            return None # Or handle differently
        except Exception as e:
            log.error(f"Error reading file {pdf_path}: {e}")
            return None

    if not parts:
        log.error(f"Could not create any valid input Parts for Vertex AI. Case: {case_id}, Doc: {doc_type}")
        return None

    log.debug(f"Prepared {len(parts)} parts for Vertex AI from files: {file_paths_for_log}")

    # --- Prepare the Prompt ---
    field_list_str = "\n".join([f"- {field}" for field in fields_to_extract])
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        doc_type=doc_type,
        case_id=case_id,
        num_pages=len(parts),
        field_list_str=field_list_str
    )
    log.debug(f"Generated prompt for Case {case_id}, Doc {doc_type}:\n{prompt[:500]}...") # Log truncated prompt

    # --- Call Vertex AI ---
    try:
        log.info(f"Sending request to Vertex AI for Case: {case_id}, Doc: {doc_type}")
        # Combine the prompt text and the PDF parts
        full_request_content = [prompt] + parts
        response = model.generate_content(
            full_request_content,
            generation_config={"response_mime_type": "application/json"}, # Request JSON output
            safety_settings=SAFETY_SETTINGS,
            stream=False # Get the full response at once
        )
        log.info(f"Received response from Vertex AI for Case: {case_id}, Doc: {doc_type}")

        # --- Parse the Response ---
        # Assuming the model successfully returns valid JSON in response.text
        # Add robust error handling for JSON parsing
        try:
            # Strip potential markdown code fences ```json ... ``` if model adds them
            raw_json = response.text.strip()
            if raw_json.startswith("```json"):
                 raw_json = raw_json[7:]
            if raw_json.endswith("```"):
                raw_json = raw_json[:-3]
            raw_json = raw_json.strip()

            extracted_data = json.loads(raw_json)
            log.debug(f"Successfully parsed JSON response for Case: {case_id}, Doc: {doc_type}")
            return extracted_data

        except json.JSONDecodeError as json_err:
            log.error(f"Failed to decode JSON response from Vertex AI for Case: {case_id}, Doc: {doc_type}. Error: {json_err}")
            log.error(f"Raw Vertex AI Response Text:\n{response.text}")
            # Fallback: Try to return at least the raw text? Or indicate failure.
            # You might try regex or partial extraction here if JSON fails consistently
            return {"error": "JSON Decode Error", "raw_response": response.text}
        except AttributeError:
             log.error(f"Response object lacks 'text' attribute. Type: {type(response)}. Response: {response}")
             # Check for blocked content
             if response.candidates and not response.candidates[0].content.parts:
                 block_reason = response.candidates[0].finish_reason
                 safety_ratings = response.candidates[0].safety_ratings
                 log.error(f"Content likely blocked. Reason: {block_reason}, Ratings: {safety_ratings}")
                 return {"error": f"Content Blocked: {block_reason}", "safety_ratings": safety_ratings}
             return {"error": "AttributeError parsing response"}


    except google.api_core.exceptions.GoogleAPIError as api_err:
        log.exception(f"Vertex AI API Error for Case: {case_id}, Doc: {doc_type}. Error: {api_err}")
        return {"error": f"Vertex AI API Error: {api_err}"}
    except Exception as e:
        log.exception(f"Unexpected Error during Vertex AI call or parsing for Case: {case_id}, Doc: {doc_type}. Error: {e}")
        return {"error": f"Unexpected Error: {e}"}


# --- Main Processing Function ---
def process_zip_file(zip_file_path: str):
    """
    Main function to process the uploaded zip file.
    1. Extracts zip to a temporary location.
    2. Iterates through case folders.
    3. Groups PDFs by document type within each case.
    4. Uses ThreadPoolExecutor to process documents concurrently via Vertex AI.
    5. Aggregates results into a pandas DataFrame.
    6. Saves the DataFrame to an Excel file.
    """
    all_extracted_data = []
    output_excel_path = Path(OUTPUT_FILENAME) # Store locally first

    # Create a temporary directory for extraction
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

        # --- 2. & 3. Iterate Cases and Group Files ---
        case_folders = [d for d in temp_dir.iterdir() if d.is_dir()]
        if not case_folders:
             log.error(f"No case folders found in the extracted zip content at {temp_dir}")
             raise ValueError("No case folders found in the zip file.")

        tasks = []
        # Prepare tasks for concurrent execution
        for case_folder in case_folders:
            case_id = case_folder.name
            log.info(f"Processing Case ID: {case_id}")
            document_groups = _group_files_by_doc_type(case_folder)

            if not document_groups:
                log.warning(f"No processable documents found in case folder: {case_id}")
                # Add a row with just Case ID if needed, or skip
                all_extracted_data.append({"CASE_ID": case_id, "Processing_Status": "No documents found"})
                continue

            for doc_type, pdf_files in document_groups.items():
                if doc_type in DOCUMENT_FIELDS:
                    fields_to_extract = DOCUMENT_FIELDS[doc_type]
                    # Add task: (case_id, doc_type, pdf_files_list, fields)
                    tasks.append((case_id, doc_type, pdf_files, fields_to_extract))
                else:
                    log.warning(f"Skipping unknown document type '{doc_type}' for Case ID: {case_id}. Define it in config.DOCUMENT_FIELDS.")

        # --- 4. Concurrent Processing ---
        case_results = defaultdict(lambda: {"CASE_ID": None}) # Store results per case_id
        if tasks:
             log.info(f"Submitting {len(tasks)} document extraction tasks to {MAX_WORKERS} workers.")
             with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_task = {
                    executor.submit(_extract_data_from_document, *task_args): task_args
                    for task_args in tasks
                }

                for future in concurrent.futures.as_completed(future_to_task):
                    task_args = future_to_task[future]
                    case_id, doc_type, _, _ = task_args
                    case_results[case_id]["CASE_ID"] = case_id # Ensure CASE_ID is set

                    try:
                        result_json = future.result()
                        if result_json:
                            if "error" in result_json:
                                log.error(f"Extraction failed for Case: {case_id}, Doc: {doc_type}. Reason: {result_json['error']}")
                                # Add error status to the row for this doc type
                                case_results[case_id][f"{doc_type}_Processing_Error"] = result_json['error']
                            else:
                                log.info(f"Successfully processed Case: {case_id}, Doc: {doc_type}")
                                # Flatten the result JSON into the case row
                                for field, data in result_json.items():
                                    # Ensure data is a dict with expected keys, handle variations if needed
                                    if isinstance(data, dict) and all(k in data for k in ['value', 'confidence', 'reasoning']):
                                         # Prefix field names with doc_type for clarity in Excel
                                        base_field_name = f"{doc_type}_{field}"
                                        case_results[case_id][f"{base_field_name}_Value"] = data.get('value')
                                        case_results[case_id][f"{base_field_name}_Confidence"] = data.get('confidence')
                                        case_results[case_id][f"{base_field_name}_Reasoning"] = data.get('reasoning')
                                    else:
                                        log.warning(f"Unexpected format for field '{field}' in response for Case: {case_id}, Doc: {doc_type}. Data: {data}")
                                        case_results[case_id][f"{doc_type}_{field}_Raw"] = str(data) # Store raw if format incorrect

                        else:
                             log.warning(f"No result returned for Case: {case_id}, Doc: {doc_type} (may indicate skipped file or internal issue).")
                             case_results[case_id][f"{doc_type}_Processing_Status"] = "No result from extraction"

                    except Exception as exc:
                        log.exception(f"Error retrieving result for Case: {case_id}, Doc: {doc_type}. Error: {exc}")
                        case_results[case_id][f"{doc_type}_Processing_Error"] = f"Task execution failed: {exc}"

             all_extracted_data.extend(case_results.values())

        # --- 5. & 6. Aggregate and Save to Excel ---
        if not all_extracted_data:
             log.warning("No data was successfully extracted from any case.")
             # Create an empty dataframe or handle as needed
             df = pd.DataFrame([{"Status": "No data extracted"}])
        else:
            log.info(f"Aggregating results from {len(all_extracted_data)} cases.")
            df = pd.DataFrame(all_extracted_data)
            # Reorder columns - Put CASE_ID first, then maybe status/errors, then fields
            cols = df.columns.tolist()
            if "CASE_ID" in cols:
                 cols.insert(0, cols.pop(cols.index("CASE_ID")))
            df = df[cols]

        try:
            log.info(f"Saving aggregated data to Excel: {output_excel_path}")
            df.to_excel(output_excel_path, index=False, engine='openpyxl')
            log.info("Excel file saved successfully.")
            return str(output_excel_path)
        except Exception as e:
            log.exception(f"Failed to save DataFrame to Excel file '{output_excel_path}': {e}")
            raise RuntimeError(f"Failed to save results to Excel: {e}")

    # End of `with tempfile.TemporaryDirectory` - temp dir is now deleted
    log.info("Temporary directory cleaned up.")