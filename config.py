# config.py
import os
from dotenv import load_dotenv
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

load_dotenv() # Optional: Load environment variables from a .env file

# --- Vertex AI Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "hbl-uat-ocr-fw-app-prj-spk-4d")
LOCATION = "asia-south1"
# Use a multimodal model capable of handling PDFs or images
MODEL_NAME = "gemini-1.5-flash-001" # Or gemini-1.5-pro-preview-0409, gemini-pro-vision etc.
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"

# --- Safety Settings ---
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- Document Field Definitions ---
# Based on the provided image
DOCUMENT_FIELDS = {
    "CRL": [
        "DATE & TIME OF RECEIPT OF DOCUMENT", "CUSTOMER REQUEST LETTER DATE",
        "BENEFICIARY NAME", "BENEFICIARY ADDRESS", "BENEFICIARY COUNTRY",
        "CURRENCY", "AMOUNT", "BENEFICIARY ACCOUNT NO / IBAN",
        "BENEFICIARY BANK", "BENEFICIARY BANK ADDRESS",
        "BENEFICIARY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE",
        "STANDARD DECLARATIONS AS PER PRODUCTS", "APPLICANT SIGNATURE",
        "APPLICANT NAME", "APPLICANT ADDRESS", "APPLICANT COUNTRY",
        "TRANSACTION Product Code Selection", "TRANSACTION EVENT", "VALUE DATE",
        "HS CODE", "TYPE OF GOODS", "INCOTERM", "DEBIT ACCOUNT NO",
        "FEE ACCOUNT NO", "LATEST SHIPMENT DATE", "DISPATCH PORT",
        "DELIVERY PORT", "FB CHARGES", "INTERMEDIARY BANK NAME",
        "INTERMEDIARY BANK ADDRESS", "INTERMEDIARY BANK COUNTRY",
        "THIRD PARTY EXPORTER NAME", "THIRD PARTY EXPORTER COUNTRY"
    ],
    "INVOICE": [
        "TYPE OF INVOICE - COMMERCIAL/PERFORMA/CUSTOMS/", "INVOICE DATE",
        "INVOICE NO", "BUYER NAME", "BUYER ADDRESS", "BUYER COUNTRY",
        "SELLER NAME", "SELLER ADDRESS", "SELLER COUNTRY", "CURRENCY",
        "AMOUNT", "BENEFICIARY ACCOUNT NO / IBAN", "BENEFICAIRY BANK", # Typo in image preserved? Verify
        "BENEFICAIRY BANK ADDRESS", # Typo in image preserved? Verify
        "BENEFICAIRY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE", # Typo in image preserved? Verify
        "Total Invoice Amount", "Invoice Amount", # Seem redundant, check if different
        "Intermediary Bank ( Field 56)", "Beneficiary Name", "Beneficiary Address",
        "Party Name ( Applicant )", "Party Name ( Beneficiary )",
        "Party Country ( Beneficiary )", "Party Type ( Beneficiary Bank )",
        "Party Name (Beneficiary Bank )", "Party Country ( Beneficiary Bank )",
        "Drawee Address", "PORT OF LOADING", "PORT OF DISCHARGE",
        "VESSEL TYPE", "VESSEL NAME", "THIRD PARTY EXPORTER NAME",
        "THIRD PARTY EXPORTER COUNTRY"
    ],
    # Add other document types (e.g., "PACKING LIST", "BL") here if needed
    "PACKING_LIST": [ # Example - Add fields from the image if present
         "SHIPPER", "CONSIGNEE", "PORT OF LOADING", "PORT OF DISCHARGE",
         "VESSEL NAME", "MARKS & NUMBERS", "NUMBER OF PACKAGES",
         "DESCRIPTION OF GOODS", "GROSS WEIGHT", "NET WEIGHT", "MEASUREMENT"
    ],
     "BL": [ # Example Bill of Lading - Add fields from the image if present
        "SHIPPER", "CONSIGNEE", "NOTIFY PARTY", "VESSEL", "VOYAGE NO",
        "PORT OF LOADING", "PORT OF DISCHARGE", "PLACE OF RECEIPT",
        "PLACE OF DELIVERY", "MARKS & NUMBERS", "NUMBER OF CONTAINERS OR PKGS",
        "DESCRIPTION OF PACKAGES AND GOODS", "GROSS WEIGHT", "MEASUREMENT",
        "FREIGHT & CHARGES", "DATE SHIPPED ON BOARD", "PLACE AND DATE OF ISSUE"
    ]

    # Add other document types like "INSURANCE", "COO" etc. based on full requirements
}

# --- Processing Configuration ---
MAX_WORKERS = 4 # Adjust based on CPU cores and API limits for parallel processing
TEMP_DIR = "temp_processing"
OUTPUT_FILENAME = "extracted_data.xlsx"

# --- Logging Configuration ---
LOG_FILE = "app_log.log"
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Prompt Template ---
# Note: Using f-string deferred evaluation with {}
EXTRACTION_PROMPT_TEMPLATE = """
Analyze the provided document pages, which constitute a single '{doc_type}' document for Case ID '{case_id}'.
The document has {num_pages} pages in total.

Your task is to extract the following fields:
{field_list_str}

For each field, provide:
1.  `value`: The extracted text value. If the field is not found or not applicable, use `null`.
2.  `confidence`: A score between 0.0 (no confidence) and 1.0 (high confidence) indicating your certainty in the extracted value. Base this on clarity, ambiguity, and directness of the text match across the relevant page(s).
3.  `reasoning`: A brief explanation (1-2 sentences) of *how* or *where* you found the value, or *why* it could not be found (e.g., "Found label 'Invoice No:' followed by 'INV-123' on page 1.", "Inferred from the consignee address block on page 2.", "Field 'XYZ' was not mentioned in the document.").

**Return the result ONLY as a single JSON object** where the keys are the field names EXACTLY as provided above, and the values are JSON objects containing `value`, `confidence`, and `reasoning`.

Example of a single field entry in the JSON output:
"INVOICE NO": {{ "value": "INV-123", "confidence": 0.98, "reasoning": "Directly stated next to the 'Invoice No:' label on the top right of page 1." }}

Ensure the entire response is a valid JSON object starting with `{{` and ending with `}}`. Do not include any text before or after the JSON object.
"""