# config.py
import os
from dotenv import load_dotenv
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

load_dotenv() # Optional: Load environment variables from a .env file

# --- Vertex AI Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "hbl-uat-ocr-fw-app-prj-spk-4d")
LOCATION = "asia-south1"
# Use a powerful multimodal model capable of handling PDFs and complex instructions
MODEL_NAME = "gemini-1.5-pro-preview-0409" # Or gemini-1.5-flash / newer appropriate model
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com" # Often not needed if default is correct

# --- Safety Settings ---
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- Document Field Definitions with Descriptions ---
# Structure: { "DOC_TYPE": [ { "name": "Field Name", "description": "Industry standard description" }, ... ] }
DOCUMENT_FIELDS = {
    "CRL": [
        {"name": "DATE & TIME OF RECEIPT OF DOCUMENT", "description": "The exact date and time when the document (e.g., Customer Request Letter) was received by the processing entity (e.g., bank)."},
        {"name": "CUSTOMER REQUEST LETTER DATE", "description": "The date mentioned on the customer's formal request letter."},
        {"name": "BENEFICIARY NAME", "description": "The name of the party (typically the exporter/seller) who is entitled to receive payment under the credit."},
        {"name": "BENEFICIARY ADDRESS", "description": "The full address of the beneficiary (exporter/seller)."},
        {"name": "BENEFICIARY COUNTRY", "description": "The country where the beneficiary is located."},
        {"name": "CURRENCY", "description": "The specific currency code (e.g., USD, EUR, INR) for the transaction amount."},
        {"name": "AMOUNT", "description": "The principal monetary value of the transaction or credit."},
        {"name": "BENEFICIARY ACCOUNT NO / IBAN", "description": "The beneficiary's bank account number or International Bank Account Number (IBAN) for receiving funds."},
        {"name": "BENEFICIARY BANK", "description": "The name of the bank where the beneficiary holds their account."},
        {"name": "BENEFICIARY BANK ADDRESS", "description": "The full address of the beneficiary's bank."},
        {"name": "BENEFICIARY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE", "description": "The unique identification code of the beneficiary's bank (SWIFT/BIC for international, Sort Code for UK, BSB for Australia, IFSC for India)."},
        {"name": "STANDARD DECLARATIONS AS PER PRODUCTS", "description": "Any standard clauses, declarations, or compliance statements required for the specific financial product."},
        {"name": "APPLICANT SIGNATURE", "description": "Indication or confirmation of the applicant's signature (may be text stating 'Signed' or an image area). Focus on confirmation text if present."},
        {"name": "APPLICANT NAME", "description": "The name of the party (typically the importer/buyer) who requested the transaction or credit."},
        {"name": "APPLICANT ADDRESS", "description": "The full address of the applicant (importer/buyer)."},
        {"name": "APPLICANT COUNTRY", "description": "The country where the applicant is located."},
        {"name": "TRANSACTION Product Code Selection", "description": "A specific code identifying the type of financial product or transaction."},
        {"name": "TRANSACTION EVENT", "description": "A code or description identifying the specific event within the transaction lifecycle (e.g., issuance, amendment)."},
        {"name": "VALUE DATE", "description": "The date on which the funds are expected to be credited or debited."},
        {"name": "HS CODE", "description": "The Harmonized System code, an international standard for classifying traded goods."},
        {"name": "TYPE OF GOODS", "description": "A general description of the merchandise being traded."},
        {"name": "INCOTERM", "description": "The standardized trade term (e.g., FOB, CIF, EXW) defining buyer/seller responsibilities for shipping, risk, and costs."},
        {"name": "DEBIT ACCOUNT NO", "description": "The applicant's account number from which funds will be debited."},
        {"name": "FEE ACCOUNT NO", "description": "The account number from which transaction fees will be debited (if different from the main debit account)."},
        {"name": "LATEST SHIPMENT DATE", "description": "The latest date by which the goods must be shipped according to the credit terms."},
        {"name": "DISPATCH PORT", "description": "The port or place from where the goods are dispatched or shipped (Port of Loading)."},
        {"name": "DELIVERY PORT", "description": "The port or place where the goods are to be delivered (Port of Discharge)."},
        {"name": "FB CHARGES", "description": "Details regarding who bears the foreign bank charges (e.g., BEN, OUR, SHA)."},
        {"name": "INTERMEDIARY BANK NAME", "description": "The name of any intermediary bank involved in the payment chain (if applicable)."},
        {"name": "INTERMEDIARY BANK ADDRESS", "description": "The address of the intermediary bank (if applicable)."},
        {"name": "INTERMEDIARY BANK COUNTRY", "description": "The country of the intermediary bank (if applicable)."},
        {"name": "THIRD PARTY EXPORTER NAME", "description": "Name of a third-party exporter involved, if different from the main beneficiary."},
        {"name": "THIRD PARTY EXPORTER COUNTRY", "description": "Country of the third-party exporter, if applicable."}
    ],
    "INVOICE": [
        {"name": "TYPE OF INVOICE - COMMERCIAL/PERFORMA/CUSTOMS/", "description": "The classification of the invoice (e.g., Commercial Invoice for payment, Proforma Invoice for quote, Customs Invoice for declaration)."},
        {"name": "INVOICE DATE", "description": "The date the invoice was issued by the seller."},
        {"name": "INVOICE NO", "description": "The unique identification number assigned to this invoice by the seller."},
        {"name": "BUYER NAME", "description": "The name of the party purchasing the goods (importer/consignee)."},
        {"name": "BUYER ADDRESS", "description": "The full address of the buyer."},
        {"name": "BUYER COUNTRY", "description": "The country where the buyer is located."},
        {"name": "SELLER NAME", "description": "The name of the party selling the goods (exporter/shipper/beneficiary)."},
        {"name": "SELLER ADDRESS", "description": "The full address of the seller."},
        {"name": "SELLER COUNTRY", "description": "The country where the seller is located."},
        {"name": "CURRENCY", "description": "The specific currency code (e.g., USD, EUR, INR) in which the invoice amounts are stated."},
        {"name": "AMOUNT", "description": "The main invoiced amount, often the total or subtotal before specific breakdowns."},
        {"name": "BENEFICIARY ACCOUNT NO / IBAN", "description": "The seller's bank account number or IBAN for receiving payment."},
        {"name": "BENEFICAIRY BANK", "description": "The name of the bank where the seller (beneficiary) holds their account. (Note: Verify spelling, often 'Beneficiary Bank')"},
        {"name": "BENEFICAIRY BANK ADDRESS", "description": "The full address of the seller's (beneficiary's) bank."},
        {"name": "BENEFICAIRY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE", "description": "The unique identification code of the seller's (beneficiary's) bank (SWIFT/BIC, IFSC, etc.)."},
        {"name": "Total Invoice Amount", "description": "The final, total amount due on the invoice, including all charges, taxes, and deductions."},
        {"name": "Invoice Amount", "description": "Often synonymous with 'Total Invoice Amount' or a key subtotal. Clarify based on context."},
        {"name": "Intermediary Bank ( Field 56)", "description": "Details of the intermediary bank used in the payment process, potentially referencing SWIFT field 56."},
        {"name": "Beneficiary Name", "description": "The name of the party (seller/exporter) who should receive the payment stated on the invoice."}, # Repeated for clarity in invoice context
        {"name": "Beneficiary Address", "description": "The address of the party (seller/exporter) receiving payment."}, # Repeated for clarity
        {"name": "Party Name ( Applicant )", "description": "The name of the applicant of the credit, often the buyer/importer, as referenced on the invoice."},
        {"name": "Party Name ( Beneficiary )", "description": "The name of the beneficiary of the credit, the seller/exporter, as referenced on the invoice."},
        {"name": "Party Country ( Beneficiary )", "description": "The country of the beneficiary (seller/exporter)."},
        {"name": "Party Type ( Beneficiary Bank )", "description": "Classification or role of the beneficiary's bank."},
        {"name": "Party Name (Beneficiary Bank )", "description": "The name of the beneficiary's bank."},
        {"name": "Party Country ( Beneficiary Bank )", "description": "The country where the beneficiary's bank is located."},
        {"name": "Drawee Address", "description": "The address of the party (often the buyer or their bank) on whom a draft (Bill of Exchange) associated with the invoice is drawn."},
        {"name": "PORT OF LOADING", "description": "The port or place where the goods were loaded onto the main transport vessel/vehicle."},
        {"name": "PORT OF DISCHARGE", "description": "The port or place where the goods are to be unloaded from the main transport vessel/vehicle."},
        {"name": "VESSEL TYPE", "description": "The type of transport used (e.g., Container Ship, Aircraft, Truck)."},
        {"name": "VESSEL NAME", "description": "The name of the specific vessel or flight number carrying the goods."},
        {"name": "THIRD PARTY EXPORTER NAME", "description": "Name of a third-party exporter involved, if different from the main seller named."},
        {"name": "THIRD PARTY EXPORTER COUNTRY", "description": "Country of the third-party exporter, if applicable."}
    ]
}

# --- Processing Configuration ---
MAX_WORKERS = 4 # Adjust based on CPU cores and API limits for parallel processing
TEMP_DIR = "temp_processing"
OUTPUT_FILENAME = "extracted_data.xlsx"

# --- Logging Configuration ---
LOG_FILE = "app_log.log"
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- NEW: Classification Prompt Template ---
CLASSIFICATION_PROMPT_TEMPLATE = """
**Task:** Analyze the provided document pages ({num_pages} pages) and classify the document type.

**Acceptable Document Types:**
{acceptable_types_str}

**Instructions:**
1.  Examine the content, layout, and typical fields across all pages.
2.  Determine which of the "Acceptable Document Types" listed above best represents the entire document.
3.  Return ONLY a single JSON object containing:
    * `"classified_type"`: The determined document type string (MUST be one of the acceptable types listed above). If the document doesn't clearly match any acceptable type, use "UNKNOWN".
    * `"confidence"`: A score between 0.0 (uncertain) and 1.0 (very certain) indicating your confidence in the classification.
    * `"reasoning"`: A brief explanation for your classification choice (e.g., "Presence of fields like Invoice No, Buyer, Seller indicates INVOICE.", "Layout and terms match standard Bill of Lading format.", "Content does not strongly align with any specific acceptable type.").

**Example Output:**
```json
{{
  "classified_type": "INVOICE",
  "confidence": 0.98,
  "reasoning": "Document contains key fields like 'Invoice Number', 'Seller Address', 'Buyer Address', and line items with amounts, consistent with a Commercial Invoice."
}}
Important: Respond ONLY with the valid JSON object. Do not include any other text, greetings, or explanations outside the JSON structure.
"""

# --- Enhanced Robust Prompt Template (Now uses descriptions) ---
EXTRACTION_PROMPT_TEMPLATE = """
**Your Role:** You are a highly meticulous and accurate AI Document Analysis Specialist. Your primary function is to extract structured data from business documents precisely according to instructions.

**Task:** Analyze the provided {num_pages} pages, which together constitute a single logical '{doc_type}' document associated with Case ID '{case_id}'. Carefully extract the specific data fields listed below. Use the provided descriptions to understand the context and meaning of each field within this document type. Consider all pages to find the most relevant and accurate information for each field.

**Fields to Extract (Name and Description):**
{field_list_str}

**Output Requirements (Strict):**

1.  **JSON Only:** You MUST return ONLY a single, valid JSON object as your response. Do NOT include any introductory text, explanations, summaries, apologies, or any other text outside of the JSON structure. The response must start directly with `{{` and end with `}}`.
2.  **JSON Structure:** The JSON object MUST have keys corresponding EXACTLY to the field **names** provided in the "Fields to Extract" list above.
3.  **Field Value Object:** Each value associated with a field key MUST be another JSON object containing the following three keys EXACTLY:
    * `"value"`: The extracted text value for the field.
        * If the field is clearly present, extract the value accurately.
        * If the field is **not found** or **not applicable** after searching all pages, use the JSON value `null` (not the string "null").
        * If multiple potential values exist, select the most likely one based on context and document conventions, and mention the ambiguity in the reasoning.
    * `"confidence"`: A numerical score between 0.0 (no confidence) and 1.0 (high confidence).
        * Base this score on factors like: direct label matching (high confidence), clarity/legibility, contextual inference using the field description (medium/high confidence), calculation required (medium confidence), ambiguity, or uniqueness.
        * Use lower scores (e.g., < 0.5) if highly uncertain. Use 0.0 if the value is `null`.
    * `"reasoning"`: A concise explanation (1-2 sentences) justifying the extracted `value` and `confidence`.
        * Specify *how* (e.g., "Directly beside label 'Invoice No:'", "Inferred from SHIPPER address block using description", "Calculated sum", "Standard term found") and *where* (e.g., "on page 1, top right", "footer of page 2", "multiple lines pages 1-3").
        * If `null`, explain *why* (e.g., "Field label 'XYZ' or related info based on description not present.", "Section typically containing this info is missing.").

**Example of Expected JSON Output Structure:**
(Note: The actual field names will match those provided in the 'Fields to Extract' list for the specific '{doc_type}')

```json
{{
  "INVOICE NO": {{
    "value": "INV-98765",
    "confidence": 0.99,
    "reasoning": "Extracted directly from the field labeled 'Invoice No:' located at the top right of page 1."
  }},
  "BUYER ADDRESS": {{
    "value": "123 Main St, Anytown, CA 90210",
    "confidence": 0.95,
    "reasoning": "Inferred from the 'Consignee' section on page 1, matching the description of the party purchasing goods."
  }},
  "INTERMEDIARY BANK NAME": {{
    "value": null,
    "confidence": 0.0,
    "reasoning": "No section or field explicitly mentioning an Intermediary Bank or SWIFT Field 56 was found across all pages based on the description."
  }},
  "AMOUNT": {{
    "value": "15000.75",
    "confidence": 0.90,
    "reasoning": "Value '15,000.75' found next to 'Total Amount Due' label on page 2. Formatted as number string."
  }}
  // ... (all other requested fields for the '{doc_type}' document would follow)
}}"""