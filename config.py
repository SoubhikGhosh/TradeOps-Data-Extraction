# config.py
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

# Import models that AppSettings might use or reference
from models import DocumentFieldDefinition

# Load environment variables from a .env file if present
load_dotenv()

# --- HarmCategory and HarmBlockThreshold to String mapping for Pydantic ---
# Pydantic works best with primitive types for environment variable loading and JSON serialization.
# We'll store the string representation and convert back if needed,
# or use the actual enum values when initializing directly in Python.

HARM_CATEGORY_MAP_STR_TO_ENUM = {
    "HARM_CATEGORY_HATE_SPEECH": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    "HARM_CATEGORY_DANGEROUS_CONTENT": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    "HARM_CATEGORY_HARASSMENT": HarmCategory.HARM_CATEGORY_HARASSMENT,
    "HARM_CATEGORY_UNSPECIFIED": HarmCategory.HARM_CATEGORY_UNSPECIFIED, # Added for completeness
}
HARM_BLOCK_THRESHOLD_MAP_STR_TO_ENUM = {
    "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
    "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
    "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    "HARM_BLOCK_THRESHOLD_UNSPECIFIED": HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED, # Added
}


class AppSettings(BaseSettings):
    """
    Centralized application settings managed by Pydantic.
    Loads from environment variables and .env file.
    """
    # --- Vertex AI Configuration ---
    GOOGLE_CLOUD_PROJECT: str = Field(default="hbl-uat-ocr-fw-app-prj-spk-4d", env="GOOGLE_CLOUD_PROJECT")
    LOCATION: str = Field(default="asia-south1", env="LOCATION")
    MODEL_NAME: str = Field(default="gemini-1.5-pro-002", env="GEMINI_MODEL")
    API_ENDPOINT: Optional[str] = Field(default=None, env="API_ENDPOINT") # Often not needed if default is correct

    # --- Supported File Types ---
    SUPPORTED_MIME_TYPES: Dict[str, str] = {
        "application/pdf": "PDF",
        "image/png": "PNG",
        "image/jpeg": "JPEG",
        "image/jpg": "JPEG"
    }
    SUPPORTED_FILE_EXTENSIONS: List[str] = [".pdf", ".png", ".jpg", ".jpeg"]

    # --- Safety Settings ---
    # Storing as Dict[str, str] for easier env var management if needed, then converting.
    # Defaulting to the direct enum values here as it's Python-side initialization.
    # If these were to be loaded from complex env vars, custom parsers would be needed.
    SAFETY_SETTINGS_CONFIG: Dict[HarmCategory, HarmBlockThreshold] = Field(
        default={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    # --- Document Field Definitions with Descriptions ---
    # This is a large structure, directly embedded.
    DOCUMENT_FIELDS: Dict[str, List[DocumentFieldDefinition]] = Field(
        default={
            "CRL": [
                {"name": "DATE & TIME OF RECEIPT OF DOCUMENT",
                "description": """
                    **Objective:** Analyze the provided document image to locate a specific circular seal and accurately extract the date and time indicated by it. Time extraction must be precise, based on the **leading edge** of the arrow pointer and the 24-slot/4-segment time mechanism, including a specific rule for alignment with major hour lines. Extraneous text on the seal not directly part of the date or time scale markings should be ignored for date/time determination.

                    **Seal Description:**
                    1.  **Structure:** Identify a circular seal composed of two concentric circles.
                    2.  **Inner Circle (Date):** Contains a date in a recognizable format (e.g., YY-MM-DD, DD MON YY, MM/DD/YY).
                    3.  **Pointer (Arrow Icon):** A distinct **triangular pointer (arrow icon)** is located near the inner circle, typically centered above the date. The base of the triangle is towards the center of the seal, and its sharp end points radially outwards towards the outer time scale ring.
                    4.  **Outer Ring (24-Hour Time Scale):** The area between the two circles functions as a 24-hour time scale, divided into 24 major 'slots' representing hours 00 through 23.
                        * **Major Hour Lines:** These slots are typically demarcated by prominent lines (major hour lines) marking the beginning of each hour (e.g., the line for '01', '02', ..., '12', ..., '23', '00'). Assume these are arranged sequentially in a clockwise direction, with the '00' hour typically located at the 12 o'clock (top) position.
                        * **Minute Segments within an Hour Slot:** Each major hour slot (e.g., the slot for hour 'H', which is the space between the major hour line 'H' and the major hour line 'H+1') is further divided into **four equal 15-minute segments**. These are defined by the position of the pointer's leading edge *after it has passed the major hour line 'H'*:
                            * The **first segment (for H:00)**: Begins immediately after the major hour line 'H' and ends just before the first minor tick mark.
                            * The **second segment (for H:15)**: Begins at the **first** distinct minor tick mark (short line or point) after major hour line 'H' and ends just before the second minor tick mark.
                            * The **third segment (for H:30)**: Begins at the **second** distinct minor tick mark after major hour line 'H' and ends just before the third minor tick mark.
                            * The **fourth segment (for H:45)**: Begins at the **third** distinct minor tick mark after major hour line 'H' and ends just before the next major hour line ('H+1').
                    5.  **Time Indication (Crucial - "Leading Edge" Rule with Specific Major Hour Line Interpretation):**
                        * The time is determined by the **leading edge of the triangular pointer** (defined as the edge of the pointer that moves foremost in a clockwise direction) as it aligns with the 24-hour time scale.
                        * **Hour and Minute Determination is based on two distinct cases:**

                            * **Case A: Leading Edge Precisely on a Major Hour Line:**
                                If the pointer's **leading edge** aligns *exactly* with a major hour line that marks the beginning of hour 'H' on the dial (e.g., the line for '01', '02', ..., '12', ..., '23', '00'), the time indicated is the completion of the *previous* hour.
                                * The hour is **(H-1)**. (If H is '00' on the dial, the indicated hour is '23'. If H is '01', the hour is '00', etc.)
                                * The minutes are **:00**.
                                * *Example:* If the leading edge is exactly on the '12' major hour line, the time is 11:00. If exactly on the '01' major hour line, the time is 00:00. If exactly on the '00' major hour line, the time is 23:00.

                            * **Case B: Leading Edge *Within* an Hour Slot (i.e., Past a Major Hour Line):**
                                If the pointer's **leading edge** has moved *past* a major hour line 'H' and is now positioned *within* the slot for hour 'H' (i.e., between major hour line 'H' and major hour line 'H+1'), then:
                                * The hour is **'H'** (the hour of the slot it is currently in).
                                * The **minutes** are determined by which of the four 15-minute segments of hour 'H' (as defined in Seal Description point 4) the pointer's **leading edge** currently occupies or has just entered:
                                    * If the leading edge is in the first segment (just past major line 'H', before 1st minor tick): Minutes are **:00** (Time is H:00).
                                    * If the leading edge is in the second segment (past 1st minor tick, before 2nd minor tick): Minutes are **:15** (Time is H:15).
                                    * If the leading edge is in the third segment (past 2nd minor tick, before 3rd minor tick): Minutes are **:30** (Time is H:30).
                                    * If the leading edge is in the fourth segment (past 3rd minor tick, before next major hour line 'H+1'): Minutes are **:45** (Time is H:45).

                    **Task:**
                    1.  **Locate Seal:** Find the described circular seal within the document image.
                    2.  **Extract Date:** Perform OCR on the inner circle to reliably read and extract the full date.
                    3.  **Determine Time (Precise Interpretation using "Leading Edge" and Case A/B Rules):**
                        * Carefully assess the position of the pointer's **leading edge**.
                        * First, determine if **Case A** applies (leading edge exactly on a major hour line). If so, calculate time as (H-1):00.
                        * If Case A does not apply, then **Case B** applies. Identify the current hour slot 'H' the leading edge is within. Then, determine which of the four 15-minute segments (:00, :15, :30, :45) of hour 'H' the leading edge occupies to set the minutes.
                        * **Calculate Time:** Combine the identified hour and minutes into HH:MM format (24-hour clock). Aim for exact alignment and rule application.

                    4.  **Handle Imperfections:** While aiming for exact interpretation based on the rules:
                        * Utilize any visible portions of the seal.
                        * If image blur or quality makes the exact position of the leading edge truly ambiguous between two distinct interpretations (e.g., exactly on a major line vs. just past it, or between two minute segments), this inherent ambiguity should be noted. However, the primary goal is to apply the rules assuming a clear reading if possible. "Estimation" should only refer to resolving slight visual ambiguities to fit one of the defined rules, not inventing new rules.

                    **Output:**
                    * Provide the extracted **Date** and extracted **Time** in DD-MM-YYYY HH:MM format (24-hour clock).
                    * If, despite best efforts for exact rule application, a significant visual ambiguity required a judgment call (e.g., pointer edge extremely close to a line, making Case A vs. Case B decision difficult), briefly note this.
                """},
                {"name": "CUSTOMER REQUEST LETTER DATE",
                "description": """The specific date on which the customer (applicant) formally prepared and dated their request letter or application form. Typically found in the letter's header, near applicant details, often labeled 'Date:' or 'Dated:'. This is the letter's authorship date by the customer.
                    Example: "03-10-2023" or "October 3, 2023".
                """},
                {"name": "BENEFICIARY NAME",
                "description": """The full, official legal name of the party (exporter, seller, service provider) designated to receive funds or benefit from the transaction.
                    Look for labels: 'Beneficiary:', 'Beneficiary Name:', 'Payee:', 'To (Beneficiary):', 'Supplier Name:'. Extract complete name including legal suffixes (Ltd., Inc.).
                    Example: "Global Export Services Ltd." or "Jane Doe International".
                """},
                {"name": "BENEFICIARY ADDRESS",
                "description": """The complete mailing address of the beneficiary, including street, city, state/province, postal code. Found below/next to beneficiary name or in a 'Beneficiary Details' section. Extract as a single string.
                    Example: "123 International Parkway, Suite 500, Export City, EC 12345, Globalia".
                """},
                {"name": "BENEFICIARY COUNTRY",
                "description": """The country where the beneficiary is officially located. Often the last part of the beneficiary's address or labeled 'Country:'.
                    Example: "Germany", "Singapore".
                """},
                {"name": "REMITTANCE CURRENCY",
                "description": """The three-letter ISO 4217 currency code (e.g., USD, EUR, INR) of the funds requested for remittance.
                    Look for labels: 'Currency:', 'CCY:', 'Transaction Currency:', or a code next to the amount.
                    Example: "USD", "EUR".
                """},
                {"name": "REMITTANCE AMOUNT",
                "description": """The principal monetary value of the transaction requested, in the specified 'REMITTANCE CURRENCY'.
                    Look for labels: 'Amount:', 'Transaction Amount:', 'Value:'. Extract as a numerical value (e.g., "21712.18").
                    Example: "50000.00" or "12345.67".
                """},
                {"name": "BENEFICIARY ACCOUNT NO / IBAN",
                "description": """The beneficiary's bank account number or International Bank Account Number (IBAN) for fund credit.
                    Look for labels: 'Account No.:', 'A/C No.:', 'IBAN:', in 'Payment Instructions' or 'Beneficiary Bank Details'. IBANs start with a two-letter country code.
                    Example (IBAN): "DE89370400440532013000" or (Account No.): "001-234567-890".
                """},
                {"name": "BENEFICIARY BANK", 
                "description": """The full official name of the bank where the beneficiary holds their account.
                    Look for labels: 'Beneficiary Bank:', 'Bank Name:', 'Receiving Bank:'. Usually listed near account number and SWIFT code.
                    (Note: Correct common spelling is 'Beneficiary Bank').
                    Example: "Global Standard Commercial Bank", "Exporter's First Union Bank".
                """},
                {"name": "BENEFICIARY BANK ADDRESS",
                "description": """The full mailing address of the beneficiary's bank, including street, city, and country.
                    Found with other beneficiary bank details. (Note: Correct common spelling is 'Beneficiary Bank Address').
                    Example: "789 Finance Avenue, Central Banking District, Capital City, CB 67890, Globalia".
                """},
                {"name": "BENEFICIARY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE", 
                "description": """Unique identification code of the beneficiary's bank/branch (SWIFT/BIC, Sort Code, BSB, IFSC, etc.).
                    Look for 'SWIFT Code:', 'BIC:', 'IFSC:', 'Sort Code:', 'BSB:'. (Note: Correct common spelling is 'Beneficiary Bank SWIFT Code...').
                    Example (SWIFT): "BANKGB2LXXX" or (IFSC): "BKID0001234".
                """},
                {"name": "STANDARD DECLARATIONS AS PER PRODUCTS",
                "description": """Any standard clauses, undertakings, legal statements, or compliance declarations made by the applicant in the request letter, often related to regulations (FEMA, AML), transaction purpose, or applicant responsibility.
                    Look for sections: 'Declarations:', 'Undertakings:', or numbered/bulleted statements, usually before the signature. Extract full relevant text.
                    Example phrase: "I/We declare that this transaction complies with all applicable regulations..."
                """},
                {"name": "APPLICANT SIGNATURE",
                "description": """Confirmation or evidence of the applicant's (or authorized signatory's) signature. Can be a signature image, typed signatory name, or textual confirmation like 'Authorized Signatory'.
                    Capture typed name if present, or note 'Signature Present'.
                    Example text: "For [Applicant Company Name], (Sign) John Smith, Managing Director" or "Signed for and on behalf of Applicant Corp".
                """},
                {"name": "APPLICANT NAME",
                "description": """The full legal name of the individual or company submitting the request letter.
                    Look in letterhead, near applicant's address, or field 'Applicant Name:', 'Customer Name:'.
                    Example: "Local Importers LLC", "Alpha Trading Enterprises".
                """},
                {"name": "APPLICANT ADDRESS",
                "description": """The complete mailing address of the applicant (street, city, state, postal code, country).
                    Usually in letterhead or under 'Applicant Address:'.
                    Example: "456 Business Park, Suite 101, Metro City, MC 67890, Localia".
                """},
                {"name": "APPLICANT COUNTRY",
                "description": """The country where the applicant is officially located/registered. Typically part of applicant's address or explicitly stated.
                    Example: "India", "United Kingdom".
                """},
                {"name": "HS CODE",
                "description": """The Harmonized System (HS) or HSN code, an international standard for classifying traded goods (usually 6-10 digits).
                    Look for 'HS Code:', 'HSN Code:', 'Tariff Code:'.
                    Example: "69072300" or "851762".
                """},
                {"name": "TYPE OF GOODS",
                "description": """A general description of the merchandise or products being imported or paid for.
                    Look for 'Description of Goods:', 'Goods:', 'Nature of Goods:'.
                    Example: "Electronic Components for Manufacturing" or "Industrial Machinery Parts".
                """},
                {"name": "DEBIT ACCOUNT NO",
                "description": """The applicant's bank account number from which the principal transaction funds will be debited.
                    Look for 'Debit Account No.:', 'Account to be Debited:', 'Source Account:'.
                    Example: "123456789012" or "00501000012345".
                """},
                {"name": "FEE ACCOUNT NO",
                "description": """The applicant's account number for transaction fees, if different from the main debit account.
                    Look for 'Fee Account No.:', 'Charges Account:'. May be same as Debit Account No.
                    Example (if different): "987654321000".
                """},
                {"name": "LATEST SHIPMENT DATE",
                "description": """The latest date by which goods must be shipped by the exporter, as per contract or L/C terms.
                    Look for 'Latest Shipment Date:', 'Shipment by:'. Format DD-MM-YYYY.
                    Example: "31-12-2024".
                """},
                {"name": "DISPATCH PORT",
                "description": """The port, airport, or place from where goods are dispatched/shipped (Port of Loading).
                    Look for 'Port of Dispatch:', 'Port of Loading:', 'From Port:'.
                    Example: "Port of Hamburg" or "Shanghai Pudong Airport".
                """},
                {"name": "DELIVERY PORT",
                "description": """The port, airport, or place where goods are to be delivered in the destination country (Port of Discharge).
                    Look for 'Port of Delivery:', 'Port of Discharge:', 'To Port:'.
                    Example: "Port of New York" or "Nhava Sheva Port".
                """},
                {"name": "FB CHARGES",
                "description": """Indicates who bears foreign bank charges (BEN: Beneficiary, OUR: Applicant, SHA: Shared).
                    Look for 'Foreign Bank Charges:', 'Details of Charges:', often with options BEN/OUR/SHA.
                    Example: "BEN", "OUR".
                """},
                {"name": "INTERMEDIARY BANK NAME",
                "description": """Name of any intermediary/correspondent bank used in the payment chain.
                    Look for 'Intermediary Bank:', 'Correspondent Bank:'. If not applicable, null.
                    Example: "Global Correspondent Bank PLC".
                """},
                {"name": "INTERMEDIARY BANK ADDRESS",
                "description": """Address of the intermediary bank, if specified.
                    Example: "1 Financial Square, Global City, GC1 2XX, Interland".
                """},
                {"name": "INTERMEDIARY BANK COUNTRY",
                "description": """Country of the intermediary bank.
                    Example: "Switzerland", "USA".
                """},
                {"name": "CUSTOMER SIGNATURE",
                "description": """Confirmation or evidence of the customer's (applicant's or their authorized signatory's) signature on the request. This is synonymous with 'APPLICANT SIGNATURE'.
                    Can be an actual signature image, a typed name of the signatory, or textual confirmation like 'Authorized Signatory'. Capture typed name or note 'Signature Present'.
                    Example: "For [Customer Company Name], (Signed) Alice Brown, Finance Manager".
                """},
                {"name": "MODE OF REMITTANCE",
                "description": """The method requested by the customer for making the payment to the beneficiary.
                    Look for 'Mode of Payment:', 'Payment Method:', 'Remit by:'.
                    Examples: "Telegraphic Transfer (TT)", "SWIFT Transfer", "Demand Draft (DD)".
                """},
                {"name": "COUNTRY OF ORIGIN",
                "description": """The country where the goods being paid for were originally manufactured, produced, or grown. This might be stated in relation to the goods description. This is synonymous with 'COUNTRY OF ORIGIN OF GOODS'.
                    Look for 'Country of Origin:', 'Origin of Goods:', 'Made in:'.
                    Example: "China", "Germany", "Vietnam".
                """},
                {"name": "IMPORT LICENSE DETAILS",
                "description": """Details of any specific import license or permit required for the goods, including the license number and possibly its validity or the issuing authority.
                    Look for 'Import Licence No.:', 'Permit Number:', 'Authorization Details:'. This may be more than just a number, potentially including date or type of license.
                    Example: "Licence No: IL/COMM/2024/00123, Valid until: 31-12-2024" or "DGFT License XYZ123".
                """},
                {"name": "CURRENCY AND AMOUNT OF REMITTANCE IN WORDS",
                "description": """The total remittance amount written out in words, including the currency. (e.g., 'US Dollars One Hundred Thousand Only', 'EURO Twenty-One Thousand Seven Hundred Twelve and Cents Eighteen Only').
                    Look for labels like 'Amount in Words:', 'Sum of (Currency) in Words:'. Typically found near the numerical amount.
                    Example: "US DOLLARS FIFTY THOUSAND ONLY" or "EURO TEN THOUSAND FIVE HUNDRED AND FIFTY POINT TWENTY FIVE".
                """},
                {"name": "INVOICE NO",
                "description": """The unique identification number of the Proforma or Commercial Invoice related to this remittance request, as issued by the beneficiary/exporter. This field is being extracted from the CRL where it references an invoice.
                    Look for labels in the CRL like 'Invoice No.:', 'Ref. Invoice:', 'Against Invoice No.:'.
                    Example: "PI-2024-001" or "EXPORTINV/789".
                """},
                {"name": "INVOICE DATE",
                "description": """The date on which the referenced Proforma or Commercial Invoice (see 'INVOICE NO' field) was issued by the beneficiary/exporter. This field is being extracted from the CRL where it references an invoice.
                    Look for labels in the CRL like 'Invoice Date:', 'Date of Invoice:'.
                    Example: "10-07-2024" or "July 10, 2024".
                """},
                {"name": "INVOICE VALUE",
                "description": """The total monetary value stated on the referenced Proforma or Commercial Invoice. This should align with the 'REMITTANCE AMOUNT' if the full invoice value is being paid via this CRL. This field is being extracted from the CRL where it references an invoice.
                    Look for labels in the CRL like 'Invoice Amount:', 'Invoice Total Value:', 'Value of Invoice:'.
                    Example: "21712.18" or "150000.00".
                """},
                {"name": "EXCHANGE RATE",
                "description": """The exchange rate applied or requested for converting the remittance amount from one currency to another, if applicable (e.g., from local currency of debit account to the foreign currency of remittance).
                    Look for 'Exchange Rate:', 'Rate Applied:', 'FX Rate:'. May be specified by customer or bank.
                    Example: "1 USD = 83.50 INR" or "0.92 EUR/USD".
                """},
                {"name": "TREASURY REFERENCE NO",
                "description": """A unique reference number for a foreign exchange (forex) deal booked with the bank's treasury to fix the exchange rate, if applicable. Similar to 'DEAL ID'.
                    Look for 'Treasury Ref No:', 'Forex Deal ID:', 'FX Contract No.:'.
                    Example: "TRSY/FX/2024/00567".
                """},
                {"name": "SPECIFIC REFERENCE FOR SWIFT FIELD 70/72",
                "description": """Narrative or specific instructions the applicant wants to be included in the SWIFT payment message's Field 70 (Remittance Information) or Field 72 (Sender to Receiver Information). This often includes invoice numbers, purpose of payment, or other details for the beneficiary or beneficiary's bank.
                    Look for labels 'Payment Reference (for SWIFT F70):', 'Message to Beneficiary Bank (F72):', 'Narrative for Beneficiary:'. Extract the text provided.
                    Example: "/INV/PI-2024-001/ORDER/PO-ABC-123" or "PAYMENT FOR CONSULTANCY SERVICES AGREEMENT DATED 01-06-2024".
                """},
                {"name": "DESCRIPTION OF GOODS", 
                "description": """A detailed account or specific description of the goods or services for which the payment is being made, as stated in the customer's request letter. This might be more detailed than 'TYPE OF GOODS' and directly quoted from the customer's application.
                    Look for sections like 'Description of Goods/Services:', 'Details of Import:', or a narrative part describing the items.
                    Example: "Supply and installation of Model X-500 Industrial Compressor and associated spare parts" or "Annual Subscription Fee for Cloud Software Platform".
                """},
                {"name": "TRANSACTION Product Code Selection",
                "description": """A specific internal code or explicit selection by the applicant identifying the bank's financial product used for this transaction (e.g., 'Import Advance', 'Direct Import Bill').
                    Search for 'Product Code:', 'Transaction Product:', or a highlighted product name.
                    Example: "IMP-ADV-001" or "TF-PAY-SIGHT".
                    """},
                {"name": "TRANSACTION EVENT",
                "description": """Identifies the specific event in the transaction lifecycle (e.g., 'Payment Initiation', 'Remittance Issuance'). For CRL, this is typically the initiation of a payment instruction.
                    Often implicit. Look for explicit statements if any.
                    Example: "Outward Remittance Processing".
                    """},
                {"name": "VALUE DATE",
                "description": """The requested date for funds to be debited from applicant's account and/or credited to the beneficiary (effective settlement date).
                    Look for 'Value Date:', 'Settlement Date:', 'Debit Date:'.
                    Example: "17-07-2024" or "Spot".
                    """},
                {"name": "INCO TERM",
                "description": """The standardized three-letter trade term (e.g., FOB, CIF, EXW) defining buyer/seller responsibilities for delivery, costs, and risks, as mentioned in the CRL (often referencing sales contract/invoice).
                    Look for 'Incoterm:', 'Trade Term:', or terms like 'CIF (Port Name)'.
                    Example: "CIF (Destination Port)" or "EXW (Seller's Factory)".
                    """},
                {"name": "THIRD PARTY EXPORTER NAME",
                "description": """Name of a third-party exporter if goods are exported by an entity different from the main beneficiary receiving payment.
                    Look for 'Third Party Exporter:', 'Actual Exporter (if different):'. If not applicable, null.
                    Example: "Global Sourcing Agents Ltd.".
                """},
                {"name": "THIRD PARTY EXPORTER COUNTRY",
                "description": """Country of the third-party exporter, if applicable.
                    Example: "Hong Kong".
                """}
            ],
            
            "INVOICE": [
                {
                    "name": "TYPE OF INVOICE - COMMERCIAL/PROFORMA/CUSTOMS", 
                    "description": """The explicit classification of the invoice document based on its title and purpose.
                                    Search for prominent titles like 'COMMERCIAL INVOICE', 'PROFORMA INVOICE', 'TAX INVOICE', 'CUSTOMS INVOICE', 'INVOICE', 'PROFORMA', 'PI', 'PO'.
                                    - **COMMERCIAL INVOICE:** A final bill.
                                    - **PROFORMA/PERFORMA INVOICE:** A preliminary bill/quotation. 'Order Confirmation' or 'Sales Order' with full details may function as one.
                                    - **CUSTOMS INVOICE:** For customs authorities.
                                    Infer based on content if title is ambiguous. (Note: Standard spelling is 'Proforma').
                                    Example: "PROFORMA INVOICE" or "COMMERCIAL INVOICE".
                                    """
                },
                {
                    "name": "INVOICE DATE",
                    "description": """The specific date when the invoice was created or issued by the seller/issuer.
                                    Look for labels like 'Invoice Date', 'Date', 'Issue Date', 'Date of Issue'.
                                    It's usually found near the invoice number or seller's details. Ensure it's a clear date format (e.g., DD-MMM-YY, MM/DD/YYYY, YYYY-MM-DD). Example: '27-Sep-23'[cite: 4]."""
                },
                {
                    "name": "INVOICE NO",
                    "description": """The unique alphanumeric identifier assigned to this specific invoice by the seller/issuer.
                                    Search for labels such as 'Invoice No.', 'Invoice #', 'Inv. No.', 'Reference #', 'Document No.'.
                                    This is a critical field and is usually prominently displayed, often in the header or near the seller's information. Example: '2546049' listed under 'Reference #' [cite: 2] for a proforma invoice."""
                },
                {
                    "name": "BUYER NAME",
                    "description": """The full legal name of the individual or company purchasing the goods or services.
                                    Look for labels like 'Buyer', 'Bill To', 'Customer', 'Sold To', 'Consignee' (if also the buyer), 'Importer', 'To:', 'Applicant'.
                                    It's often located in a distinct section detailing the recipient of the invoice. Example: 'Arrow Business Advisory Pvt. Ltd' [cite: 4] under 'BILL TO:'."""
                },
                {
                    "name": "BUYER ADDRESS",
                    "description": """The complete mailing address of the buyer, including street, city, state/province, postal code, and potentially country.
                                    This information is typically found directly below or adjacent to the 'BUYER NAME' under labels like 'Address', or within the 'Bill To' or 'Consignee' block.
                                    Extract the full, multi-line address as a single string. Example: '159 Mittal Industrial Estate Sanjay Building No. 5/B Marol Naka, Andheri (East) Mumbai - 400 059 India'[cite: 4]."""
                },
                {
                    "name": "BUYER COUNTRY",
                    "description": """The country where the buyer is officially located or registered.
                                    This is often the last line of the buyer's address or may be explicitly labeled as 'Country'.
                                    If the address is multi-line, identify the country name. Example: 'India' [cite: 4] as part of the buyer's address."""
                },
                {
                    "name": "SELLER NAME",
                    "description": """The full legal name of the individual or company selling the goods or services and issuing the invoice.
                                    Look for labels like 'Seller', 'From', 'Shipper' (if also the seller), 'Exporter', 'Beneficiary', 'Invoice From', or it might be the company name in the letterhead.
                                    Example: 'TRANSCENDIA, INC' [cite: 1] at the top of the document."""
                },
                {
                    "name": "SELLER ADDRESS",
                    "description": """The complete mailing address of the seller, including street, city, state/province, postal code, and country.
                                    Usually found near the 'SELLER NAME', often in the header or footer of the invoice, or under a 'Remit To' or 'From' section.
                                    Extract the full, multi-line address as a single string. Example: '300 INDUSTRIAL PARKWAY RICHMOND, IN 47374'[cite: 1]. A more complete corporate HQ address might also be '9201 W. Belmont Avenue, Franklin Park, IL 60131'[cite: 27]. Prefer the address most clearly associated with the invoice issuance or seller identity on the primary invoice pages."""
                },
                {
                    "name": "SELLER COUNTRY",
                    "description": """The country where the seller is officially located or registered.
                                    This is typically the last line of the seller's address or may be explicitly labeled.
                                    Based on the address 'RICHMOND, IN 47374'[cite: 1], the country is implicitly USA. For 'Franklin Park, IL 60131'[cite: 27], it's also USA. Explicitly state "USA" if inferred from state codes like IN or IL."""
                },
                {
                    "name": "INVOICE CURRENCY", 
                    "description": """The specific currency in which the invoice amounts are denominated (e.g., USD, EUR, GBP, INR).
                                    Look for currency symbols ($, €, £) or currency codes (USD, EUR) next to monetary values, especially the total amount.
                                    Sometimes explicitly stated like 'All amounts in USD'. Example: 'USD' is appended to the amount '$135,750.00 USD'[cite: 3]."""
                },
                {
                    "name": "INVOICE AMOUNT/VALUE",
                    "description": """The primary financial value of the invoice, typically the total sum of goods/services before certain taxes or after certain discounts, or the grand total if no other total is more prominent.
                                    Search for terms like 'Total', 'Subtotal', 'Net Amount', 'Invoice Total', 'Grand Total'.
                                    This should be a numerical value. Be careful to distinguish it from line item amounts if a clear overall total is present. Example: '$135,750.00'[cite: 3]."""
                },
                {
                    "name": "INVOICE AMOUNT/VALUE IN WORDS",
                    "description": """The total invoice amount written out in words (e.g., 'One Hundred Thirty-Five Thousand Seven Hundred Fifty Dollars Only').
                                    This field is often found near the numerical total amount, sometimes labeled 'Amount in Words', 'Say Total', or just appearing as a textual representation of the sum.
                                    This may not always be present. If not found, state null."""
                },
                {
                    "name": "BENEFICIARY ACCOUNT NO / IBAN",
                    "description": """The bank account number or International Bank Account Number (IBAN) of the seller (beneficiary) where the payment should be sent.
                                    Look for labels like 'Account No.', 'A/C No.', 'IBAN', 'Beneficiary Account'. Often found in a 'Bank Details' or 'Payment Instructions' section.
                                    Example: 'Account #: 830769961' for both ACH and Wire[cite: 29]."""
                },
                {
                    "name": "BENEFICIARY BANK",
                    "description": """The name of the bank where the seller (beneficiary) holds their account.
                                    Search for labels such as 'Bank Name', 'Beneficiary Bank', 'Bank', 'Payable to Bank'.
                                    This is usually listed in the payment instructions or bank details section. Example: 'JPMorgan Chase'[cite: 29]."""
                },
                {
                    "name": "BENEFICIARY BANK ADDRESS",
                    "description": """The full mailing address of the seller's (beneficiary's) bank.
                                    Look for this information near the beneficiary bank's name or within the 'Bank Details' section.
                                    It should include street, city, and country. Example: 'New York, NY 10017'[cite: 29]."""
                },
                {
                    "name": "BENEFICIARY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE / ROUTING NO", # Spelling from user, expanded
                    "description": """The unique identification code for the seller's (beneficiary's) bank. This could be a SWIFT/BIC code (for international payments),
                                    ABA Routing Number (for US payments), Sort Code (UK), BSB (Australia), IFSC (India), etc.
                                    Look for labels like 'SWIFT Code', 'BIC', 'ABA No.', 'Routing No.', 'IFSC', 'Sort Code', 'BSB'. Example: 'Swift Code: CHASUS33' or 'ABA (Routing) #: 071000013' (for ACH) or 'Bank Routing Number: 021000021' (for Wire)[cite: 29]. Prioritize SWIFT if available for international context, or the most relevant routing for the transaction type."""
                },
                {
                    "name": "Total Invoice Amount",
                    "description": """The final, definitive total monetary sum due on the invoice, inclusive of all items, charges, taxes (if applicable and included in the final sum), and less any deductions reflected in the total.
                                    This is often labeled 'Grand Total', 'Total Amount Due', 'Total Invoice Value', 'Please Pay This Amount'.
                                    It should be the ultimate figure the buyer is expected to pay. Example: '$135,750.00 USD' [cite: 3] (appears as the main extension and final sum in this proforma)."""
                },
                {
                    "name": "Invoice Amount", # Repeated field, ensure description helps differentiate or confirms synonymity
                    "description": """This field typically refers to the primary sum of the invoice. It can be synonymous with 'Total Invoice Amount' if only one total is presented.
                                    If there are multiple totals (e.g., Subtotal, Total before Tax, Grand Total), this should ideally capture the most representative invoiced amount, often the grand total.
                                    Verify if it's distinct from other amounts like 'Subtotal'. In many cases, it will be the same as 'Total Invoice Amount'. Example: '$135,750.00 USD'[cite: 3]."""
                },
                {
                    "name": "Beneficiary Name", # Often the same as Seller Name
                    "description": """The name of the ultimate recipient of the funds, usually the seller or exporter.
                                    Look for labels like 'Beneficiary', 'Payable to', 'Beneficiary Name'. This is often the same as the 'SELLER NAME'.
                                    Confirm if explicitly stated in a 'Payment Details' or 'Beneficiary Information' section. Example: 'Transcendia, Inc. - Depository'[cite: 29]. If just 'Transcendia, Inc.' is listed as seller[cite: 1], use that if more direct."""
                },
                {
                    "name": "Beneficiary Address", # Often the same as Seller Address
                    "description": """The full address of the beneficiary (seller/exporter) to whom the payment is directed.
                                    This is commonly the same as the 'SELLER ADDRESS'. Check for specific 'Beneficiary Address' details if provided separately in payment instructions.
                                    Example: '9201 W. Belmont Avenue Franklin Park, IL 60131' [cite: 29] associated with the beneficiary name."""
                },
                {
                    "name": "DESCRIPTION OF GOODS",
                    "description": """A detailed account of the products or services being invoiced.
                                    This is usually found in the main table or line items section of the invoice. It can include product names, codes, specifications, or service descriptions.
                                    Extract all relevant descriptive text for each line item, or a summary if it's a very long list.
                                    Example: 'HA Laminating Film 984mm X 1,829 LM Rolls'[cite: 3]. If multiple items, list them or summarize."""
                },
                {
                    "name": "QUANTITY OF GOODS",
                    "description": """The amount or number of units for each item or service listed on the invoice.
                                    Look for columns labeled 'Quantity', 'Qty', 'Units', 'No. of Items'.
                                    Specify units if mentioned (e.g., pcs, kgs, hrs, SM). Example: '75,000 SM' (Square Meters)[cite: 3]."""
                },
                {
                    "name": "PAYMENT TERMS",
                    "description": """The conditions agreed upon for payment of the invoice, such as the timeframe and method.
                                    Search for labels like 'Payment Terms', 'Terms of Payment', 'Terms'.
                                    Examples include 'Net 30 days', 'Due Upon Receipt', '50% Advance, 50% on Delivery'.
                                    Example: '50% advance and balance 50% after 60 days from the date of Bill of Lading (BL)'[cite: 3]. Also see 'Standard Payment terms - Net 30' [cite: 33] on a general info page, but prefer terms on the invoice itself."""
                },
                {
                    "name": "BENEFICIARY/SELLER'S SIGNATURE",
                    "description": """The handwritten or digital signature of the authorized representative of the seller/beneficiary, or the typed name if a physical signature is replaced by it in a digital document.
                                    Look for a signature line or block often labeled 'Seller's Signature', 'Authorized Signature', 'For [Seller Company Name]'.
                                    This may not always be present, or could be a scanned image. Describe if present (e.g., "Signature present", "Typed name: Diana McGehee"). Example: 'Diana McGehee' typed below 'Sr Customer Service'[cite: 3], which might represent authorization."""
                },
                {
                    "name": "APPLICANT/BUYER'S SIGNATURE",
                    "description": """The handwritten or digital signature of the authorized representative of the applicant/buyer, acknowledging the invoice or associated order.
                                    Look for a signature line or block often labeled 'Buyer's Signature', 'Authorized Signature', 'Accepted By', 'For [Buyer Company Name]'.
                                    This is less common on invoices themselves unless it's a proforma being accepted, but more common on related Purchase Orders. Example: 'Authorised Signatory' with a signature for 'For Arrow Business Advisory Private Limited' [cite: 4] at the bottom, indicating acceptance/issuance from buyer's perspective on a document they might have prepared or signed."""
                },
                {
                    "name": "MODE OF REMITTANCE",
                    "description": """The method by which the payment is to be made (e.g., Wire Transfer, ACH, Cheque, Credit Card).
                                    This information is often found within the 'Payment Instructions', 'Bank Details', or 'Payment Terms' sections.
                                    The document shows 'ACH & Wire Transfer Instructions' [cite: 29] and mentions 'Credit cards are accepted' [cite: 32] and 'Remit to Address for Checks'[cite: 31]. List all applicable or the primary ones mentioned in context of this transaction."""
                },
                {
                    "name": "MODE OF TRANSIT",
                    "description": """The method of transportation used for shipping the goods (e.g., Sea, Air, Road, Rail).
                                    Look for labels like 'Ship Via', 'Mode of Shipment', 'Transport Mode', 'By'.
                                    Example: 'Ship Via Ocean'[cite: 2], 'Mode of Shipment SEA'[cite: 12]."""
                },
                {
                    "name": "INCO TERM",
                    "description": """The Incoterm (International Commercial Term) specifies the responsibilities of buyers and sellers in international trade (e.g., EXW, FOB, CIF, DDP).
                                    Look for labels like 'Incoterms', 'Terms of Sale', 'Freight Terms', or a three-letter code often followed by a location.
                                    Example: 'EX-Works Richmond IN' [cite: 2] (EXW is the Incoterm)."""
                },
                {
                    "name": "HS CODE",
                    "description": """The Harmonized System (HS) code or HTS (Harmonized Tariff Schedule) code, which is an international nomenclature for the classification of products.
                                    It allows customs authorities to identify products and apply duties and taxes. Look for labels like 'HS Code', 'HTS Code', 'Tariff Code'.
                                    This is more common on Commercial or Customs Invoices. May not be present on all Proformas. If not found, state null."""
                },
                {
                    "name": "Intermediary Bank (Field 56)", # Existing field
                    "description": """Details of any intermediary bank (correspondent bank) that is used to route the payment from the buyer's bank to the seller's (beneficiary's) bank.
                                    Often referred to by 'Field 56' in SWIFT messages. Look for labels like 'Intermediary Bank', 'Correspondent Bank', or specific SWIFT field references if available.
                                    This may not always be present or required. If not found, state null."""
                },
                {
                    "name": "INTERMEDIARY BANK NAME",
                    "description": """The name of the intermediary bank, if one is specified in the payment instructions.
                                    This would be under a section labeled 'Intermediary Bank'. If no such section or bank is named, state null."""
                },
                {
                    "name": "INTERMEDIARY BANK ADDRESS",
                    "description": """The full address of the intermediary bank, if specified.
                                    This information would be found along with the intermediary bank's name. If not present, state null."""
                },
                {
                    "name": "INTERMEDIARY BANK COUNTRY",
                    "description": """The country where the intermediary bank is located, if specified.
                                    Typically part of the intermediary bank's address. If not present, state null."""
                },
                {
                    "name": "Party Name ( Applicant )", # Existing field
                    "description": """The name of the party applying for a service related to the invoice, often the buyer/importer, especially in the context of a Letter of Credit or financing.
                                    On an invoice, this is usually synonymous with the 'BUYER NAME'.
                                    Look for labels like 'Applicant', or infer from the 'Buyer' or 'Bill To' details. Example: 'Arrow Business Advisory Pvt. Ltd'[cite: 4]."""
                },
                {
                    "name": "Party Name ( Beneficiary )", # Existing field
                    "description": """The name of the party who is the beneficiary of the payment or transaction related to the invoice, typically the seller/exporter.
                                    On an invoice, this is usually synonymous with the 'SELLER NAME' or 'BENEFICIARY NAME'.
                                    Example: 'TRANSCENDIA, INC'[cite: 1], or more specifically 'Transcendia, Inc. - Depository'[cite: 29]."""
                },
                {
                    "name": "Party Country ( Beneficiary )", 
                    "description": """The country where the beneficiary (seller/exporter) is located.
                                    This is typically the country of the 'SELLER ADDRESS' or 'BENEFICIARY ADDRESS'.
                                    Example: USA (inferred from IN [cite: 1] or IL [cite: 29])."""
                },
                {
                    "name": "Party Type ( Beneficiary Bank )", # Existing field
                    "description": """The role or classification of the beneficiary's bank (e.g., 'Beneficiary Bank', 'Depository Bank').
                                    This might be explicitly stated or inferred from its function of receiving funds for the beneficiary.
                                    Example: 'Beneficiary Bank' is implied for JPMorgan Chase[cite: 29]."""
                },
                {
                    "name": "Party Name ( Beneficiary Bank )", # Existing field, space before closing parenthesis
                    "description": """The name of the bank that holds the account for the beneficiary (seller/exporter).
                                    This is the same as 'BENEFICARY BANK'. Example: 'JPMorgan Chase'[cite: 29]."""
                },
                {
                    "name": "Party Country ( Beneficiary Bank )", # Existing field, space before closing parenthesis
                    "description": """The country where the beneficiary's bank is located.
                                    This is the country of the 'BENEFICAIRY BANK ADDRESS'. Example: USA (inferred from 'New York, NY' [cite: 29])."""
                },
                {
                    "name": "Drawee Address", # Existing field
                    "description": """The name and address of the party on whom a draft or bill of exchange (if applicable to the transaction) is drawn.
                                    This is often the buyer or the buyer's bank. If no draft is mentioned or involved, this may not be applicable.
                                    Look for terms like 'Drawee'. On a standard invoice, this might be the Buyer's address if they are the direct payer. Example: If a draft is drawn on the buyer, it would be the buyer's address '159 Mittal Industrial Estate...India'[cite: 4]."""
                },
                {
                    "name": "PORT OF LOADING", # Existing field
                    "description": """The specific port or airport where the goods are loaded onto the main international transport vessel, aircraft, or vehicle for export.
                                    Look for labels like 'Port of Loading', 'POL', 'From Port', 'Airport of Departure', 'Place of Receipt' (if it's the start of main carriage).
                                    Example: 'Richmond IN' is mentioned under 'Freight EX-Works'[cite: 2], suggesting it's the point of origin/loading for an EXW term."""
                },
                {
                    "name": "PORT OF DISCHARGE", # Existing field
                    "description": """The specific port or airport where the goods are to be unloaded from the main international transport after arrival in the destination country.
                                    Look for labels like 'Port of Discharge', 'POD', 'To Port', 'Airport of Destination', 'Place of Delivery' (if it's the end of main carriage).
                                    Example: 'Nhava Sheva Port' [cite: 2] mentioned under 'Ship Via' and also as delivery location in PO[cite: 17]."""
                },
                {
                    "name": "VESSEL TYPE",
                    "description": """The general type of transport conveyance used for the main leg of the journey (e.g., 'Vessel', 'Aircraft', 'Truck', 'Container Ship').
                                    This may be inferred from 'Mode of Transit' (e.g., if 'Ocean' or 'Sea', then 'Vessel').
                                    Example: 'Ocean' implies a vessel[cite: 2]. If 'SEA' is mentioned[cite: 12], it also implies a vessel."""
                },
                {
                    "name": "VESSEL NAME", # Existing field
                    "description": """The specific name of the ship or vessel carrying the goods, or the flight number if by air, or voyage number.
                                    Look for labels like 'Vessel Name', 'Voyage No.', 'Flight No.', 'Carrier Name'.
                                    This is often found in shipping details sections. May not be present on a proforma invoice issued long before shipment. If not found, state null."""
                },
                {
                    "name": "THIRD PARTY EXPORTER NAME", # Existing field
                    "description": """The name of a third-party exporter, if different from the primary seller listed on the invoice.
                                    This situation arises if another company handles the export formalities or is named as the exporter of record for other reasons.
                                    Look for distinct fields like 'Third Party Exporter', or if the 'Exporter' field names a different entity than the 'Seller' or 'Beneficiary'. If not mentioned or not applicable, state null."""
                },
                {
                    "name": "THIRD PARTY EXPORTER COUNTRY", # Existing field
                    "description": """The country of the third-party exporter, if such a party is named.
                                    This would be part of the third-party exporter's address details. If no third-party exporter is mentioned, state null."""
                }
            ]
        }
    )   
# --- Processing Configuration ---
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    TEMP_DIR_PATH_STR: str = Field(default="temp_processing", env="TEMP_DIR") # Store as string
    OUTPUT_FILENAME_STR: str = Field(default="extracted_data.xlsx", env="OUTPUT_FILENAME") # Store as string

    # --- Logging Configuration ---
    LOG_FILE_PATH_STR: str = Field(default="app_log.log", env="LOG_FILE") # Store as string
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # --- Prompt Templates ---
    # These are large strings, directly embedded.
    CLASSIFICATION_PROMPT_TEMPLATE: str = Field(default="""
**Task:** You are an AI Document Classification Specialist. Your objective is to meticulously analyze the provided document pages ({num_pages} pages) and accurately classify the document's primary type based on its intrinsic purpose, structural characteristics, and specific content elements. The document may consist of multiple pages that collectively form a single logical entity.

**Acceptable Document Types:**
{acceptable_types_str}

**Detailed Instructions for Classification:**

1.  **Holistic Review:** Conduct a comprehensive examination of all pages. Pay close attention to titles, headings, recurring phrases, specific keywords (see below), data tables (e.g., itemized goods, payment details), and the overall layout to discern the document's fundamental function.
2.  **Content and Keyword Analysis (Prioritize explicit titles and document structure):**
    * **INVOICE (Commercial, Proforma, Customs):**
        * **Primary Keywords/Titles:** Search for explicit titles like "Invoice", "Commercial Invoice", "Proforma Invoice", "Tax Invoice", "Customs Invoice", "Proforma", "PI".
        * **Supporting Keywords/Phrases:** "Fattura" (Italian), "Rechnung" (German), "Facture" (French), "Bill", "Statement of Charges". Documents titled "Order Confirmation", "Sales Contract", "Sales Order", "Sales Agreement", "Indent", "PO", "Purchase Order" can function as a **Proforma Invoice** if they provide full itemization, pricing, terms, and are used to initiate payment or L/C.
        * **Core Structural Elements:**
            * A unique "Invoice Number" or "Reference Number".
            * Clear identification of "Seller" (or "Shipper", "Exporter", "Beneficiary", "From") and "Buyer" (or "Consignee", "Importer", "Bill To", "To") with names and addresses.
            * Itemized list of goods or services: descriptions, quantities, unit prices, line totals.
            * A "Total Amount Due", "Grand Total", or similar aggregate financial sum.
            * An "Invoice Date" or "Date of Issue" (a Proforma might use an "Order Date" or "Proforma Date").
            * Payment terms (e.g., "Net 30", "Advance Payment") and often bank details for payment.
        * **Differentiation:**
            * **Commercial Invoice:** Typically a definitive bill for goods already shipped or services rendered; requests payment for a completed transaction part.
            * **Proforma Invoice:** A preliminary quotation or bill issued *before* goods shipment or service completion. Used for buyer to arrange financing (like L/C), make a prepayment, or for customs pre-clearance. Often explicitly titled "Proforma Invoice".
            * **Customs Invoice:** Specifically formatted for customs authorities, detailing goods for import/export, including values, HS codes, country of origin, package details, for duty assessment.
    * **CRL (Customer Request Letter) / Application:**
        * **Primary Keywords/Titles:** "Request Letter", "Application for [Bank Product/Service]", "Letter of Instruction", "Remittance Application", "Import Payment Request".
        * **Supporting Keywords/Phrases:** Addressed "To The Manager, [Bank Name]", phrases like "We request you to...", "Kindly process the remittance for...", "Please debit our account...", "Letter of Undertaking".
        * **Content Focus:** Formal, written instruction from a customer (Applicant) to their bank. Explicitly requests the bank to perform a financial transaction (e.g., "issue a Letter of Credit", "remit funds for import", "process an outward remittance", "make an advance payment"). Contains details of the transaction: amount, currency, purpose, beneficiary name and bank details. Applicant's account to be debited is usually specified. Often includes declarations and signature of the applicant.
        * **Parties:** Clearly identifies an "Applicant" (customer) and a "Beneficiary" (recipient of funds).
3.  **Primary Purpose Determination:** Based on the collective evidence from all pages (explicit titles being a strong indicator), the presence/absence of key fields, and the characteristic markers outlined above, ascertain which single "Acceptable Document Type" most accurately represents the *overall primary purpose* of the document. What is the document's core function or the action it is intended to facilitate?
4.  **Confidence Assessment:** Assign a confidence score based on the clarity and preponderance of evidence.
    * **High Confidence (0.90-1.00):** An explicit, unambiguous title matching an acceptable type (e.g., "COMMERCIAL INVOICE") AND the presence of most core fields/structural elements characteristic of that type. The document's purpose is very clear.
    * **Medium Confidence (0.70-0.89):** The title might be generic (e.g., just "INVOICE" where it could be Commercial or Proforma) or the type is inferred (e.g., a Purchase Order acting as a Proforma Invoice based on its content). Core fields and structure strongly suggest a particular type, but some ambiguity or deviation exists. Or, a clear title but some expected key elements are missing or unclear.
    * **Low Confidence (0.50-0.69):** Title is ambiguous, misleading, or absent. Content could align with multiple types, or is missing several key indicators for any single type, making classification difficult.
    * **Very Low/Unknown (0.0-0.49):** Document does not appear to match any of the acceptable types based on available indicators, or is too fragmented/illegible for reliable classification.
5.  **Output Format (Strict Adherence Required):**
    * Return ONLY a single, valid JSON object.
    * The JSON object must contain exactly three keys: `"classified_type"`, `"confidence"`, and `"reasoning"`.
    * `"classified_type"`: The determined document type string. This MUST be one of the "Acceptable Document Types". If, after thorough analysis, the document does not definitively match any acceptable type based on the provided indicators, use "UNKNOWN".
    * `"confidence"`: A numerical score between 0.0 and 1.0 (e.g., 0.95).
    * `"reasoning"`: A concise but specific explanation for your classification. Reference explicit titles, key terms found (or absent), presence/absence of core fields, or structural elements that led to your decision (e.g., "Document explicitly titled 'PROFORMA INVOICE' on page 1. Contains seller/buyer, itemized goods with prices, total value, and payment terms. Serves as a preliminary bill for payment initiation."). If 'UNKNOWN', explain why (e.g., "Lacks clear title and key invoice fields like invoice number or distinct buyer/seller sections. Appears to be an internal statement not matching defined types.").

**Example Output:**
```json
{{
  "classified_type": "PROFORMA_INVOICE",
  "confidence": 0.98,
  "reasoning": "Document exhibits all core characteristics of a proforma invoice: details seller and buyer, lists specific goods with quantities and unit prices leading to a total amount, specifies payment terms ('50% advance...'), and indicates 'Ship Date TBD'. While not explicitly titled 'Proforma Invoice', its structure and content align perfectly with its function as a preliminary bill for initiating payment, akin to a sales order formatted for external use."
}}
```

Important: Your response must be ONLY the valid JSON object. No greetings, apologies, or any text outside the JSON structure.
""") # End of CLASSIFICATION_PROMPT_TEMPLATE

    EXTRACTION_PROMPT_TEMPLATE: str = Field(default="""
**Your Role:** You are a highly meticulous and accurate AI Document Analysis Specialist. Your primary function is to extract structured data from business documents precisely according to instructions, with an extreme emphasis on the certainty, verifiability, and contextual appropriateness of every character and field extracted.

**Task:** Analyze the provided {num_pages} pages, which together constitute a single logical '{doc_type}' document associated with Case ID '{case_id}'. Carefully extract the specific data fields listed below. Use the provided detailed descriptions to understand the context, meaning, typical location, expected format, and potential variations of each field within this document type. Consider all pages to find the most relevant and accurate information. Pay close attention to nuanced instructions, including differentiation between similar concepts and rules for inference or default values if specified.

**Fields to Extract (Name and Detailed Description):**
{field_list_str}

**Output Requirements (Strict):**

1.  **JSON Only:** You MUST return ONLY a single, valid JSON object as your response. Do NOT include any introductory text, explanations, summaries, apologies, or any other text outside of the JSON structure. The response must start directly with `{{` and end with `}}`.
2.  **JSON Structure:** The JSON object MUST have keys corresponding EXACTLY to the field **names** provided in the "Fields to Extract" list above.
3.  **Field Value Object:** Each value associated with a field key MUST be another JSON object containing the following three keys EXACTLY:
    * `"value"`: The extracted text value for the field.
        * If the field is clearly present, extract the value with absolute precision, ensuring every character is accurately represented and free of extraneous text/formatting (unless the formatting is part of the value, like a specific date format if ISO conversion is not possible).
        * If the field is **not found** or **not applicable** after thoroughly searching all pages and considering contextual clues as per the field description, use the JSON value `null` (not the string "null").
        * If multiple potential values exist (e.g., different addresses for a seller), select the one most pertinent to the field's specific context (e.g., 'Seller Address' for invoice issuance vs. 'Seller Corporate HQ Address' if the field specifically asks for that). Document ambiguity in reasoning.
        * For amounts, extract numerical values (e.g., "15000.75", removing currency symbols or group separators like commas unless they are part of a regional decimal format that must be preserved). Currency is typically a separate field.
        * For dates, if possible and certain, convert to ISO 8601 format (YYYY-MM-DD). If conversion is uncertain due to ambiguous source format (e.g., "01/02/03"), extract as it appears and note the ambiguity and original format in the reasoning.
        * For multi-line addresses, concatenate lines into a single string, typically separated by a comma and space (e.g., "123 Main St, Anytown, ST 12345, Country").

    * `"confidence"`: **Granular Character-Informed, Contextual, and Source-Aware Confidence Score (Strict)**
        * **Core Principle:** The overall confidence score (float, 0.00 to 1.00, recommend 2 decimal places) for each field MUST reflect the system's certainty about **every single character** of the extracted value, AND the **contextual correctness and verifiability** of that extraction. It's a holistic measure.
        * **Key Factors Influencing Confidence:**
            1.  **OCR Character Quality & Ambiguity:** Clarity and sharpness of each character (machine-print vs. handwriting). Low confidence for ambiguous characters (e.g., '0'/'O', '1'/'l'/'I', '5'/'S') unless context makes it near-certain.
            2.  **Handwriting Legibility:** Clarity, consistency, and formation of handwritten characters.
            3.  **Field Format Adherence:** How well the extracted value matches the expected data type and pattern (e.g., all digits for an account number, valid date structure, correct SWIFT code pattern). Deviations drastically lower confidence.
            4.  **Label Presence & Quality:** Was the value found next to a clear, standard, unambiguous label matching the field description? (e.g., "Invoice No.:" vs. inferring from a poorly labeled column). Explicit, standard labels lead to higher confidence.
            5.  **Positional Predictability:** Was the field found in a common, expected location for that document type versus an unusual one?
            6.  **Contextual Plausibility & Consistency:** Does the value make sense for the field and in relation to other extracted fields? (e.g., a 'Latest Shipment Date' should not be before an 'Order Date'). Cross-validation (e.g., amount in words vs. numeric amount) consistency is key.
            7.  **Completeness of Information:** If a field expects multiple components (e.g., full address) and parts are missing/illegible, this reduces confidence for the entire field.
            8.  **Source Document Quality:** Overall document clarity, scan quality, skew, rotation, background noise, stamps/markings obscuring text.
            9.  **Inference Level:** Was the value directly extracted or inferred? Higher degrees of inference lower confidence.

        * **Confidence Benchmarks (Stricter & More Granular):**
            * **0.99 - 1.00 (Very High/Near Certain):** All characters perfectly clear, machine-printed, unambiguous. Value from an explicit, standard label in a predictable location. Perfect format match. Contextually validated and sound. No plausible alternative interpretation. (Example: A clearly printed Invoice Number next to "Invoice No.:" label).
            * **0.95 - 0.98 (High):** Characters very clear and legible (excellent machine print or exceptionally neat handwriting). Minor, non-ambiguity-inducing visual imperfections. Strong label or unmistakable positional/contextual evidence. Correct format. Contextually valid. (Example: A clearly printed total amount next to "Grand Total:").
            * **0.88 - 0.94 (Good):** Generally clear, but minor, identifiable factors prevent higher scores:
                * One or two characters with slight ambiguity resolved with high confidence by context or pattern.
                * Very clean, legible, and consistent handwriting.
                * Information reliably extracted from structured tables with clear headers.
                * Minor print defects (slight fading/smudging) not obscuring character identity.
            * **0.75 - 0.87 (Moderate):** Value is legible and likely correct, but there are noticeable issues affecting certainty for some characters/segments, or some level of inference was required:
                * Moderately clear handwriting with some variability or less common letter forms.
                * Slightly blurry, pixelated, or faded print requiring careful interpretation for several characters.
                * Value inferred from contextual clues or non-standard labels with reasonable, but not absolute, certainty. (e.g., identifying a "Beneficiary Bank" from a block of payment text without an explicit label).
            * **0.60 - 0.74 (Low):** Significant uncertainty. Parts of the value are an educated guess, or the source is challenging:
                * Poor print quality (significant fading, widespread smudging, pixelation) affecting key characters.
                * Difficult or messy handwriting for a substantial portion of the value.
                * High ambiguity for several characters or critical segments where context provides only weak support. Value inferred with significant assumptions or from unclear/damaged source text.
            * **< 0.60 (Very Low / Unreliable):** Extraction is highly speculative or impossible to perform reliably. Value likely incorrect, incomplete, or based on guesswork. Text is largely illegible, critical characters are indecipherable, or contextual validation fails insurmountably.
        * If `"value"` is `null` (field not found/applicable), `"confidence"` MUST be `0.0`.

    * `"reasoning"`: A concise but specific explanation justifying the extracted `value` and the assigned `confidence` score. This is crucial for auditability and improvement.
        * Specify *how* the information was identified (e.g., "Directly beside explicit label 'Invoice No.' on page 1.", "Inferred from the 'BILL TO:' address block on page 2 as buyer's name.", "Calculated sum of all line item totals from table on page 3.").
        * Indicate *where* it was found (e.g., "Page 1, top right section.", "Page 3, under table column 'Description'.", "Page 5, section titled 'Payment Instructions'.").
        * **Mandatory for any confidence score below 0.99:** Briefly explain the *primary factors* leading to the reduced confidence. Reference specific issues:
            * Character ambiguity: "Value is 'INV-O012B'; Confidence 0.78: Second char 'O' could be '0', last char 'B' could be '8'; document slightly blurred in this area."
            * Print/Scan Quality: "Value '123 Main Street'; Confidence 0.85: Slight fading on 'Street', making 'S' and 't' less than perfectly sharp."
            * Handwriting: "Value 'Johnathan Doe'; Confidence 0.70: First name legible but 'Johnathan' has an unclear 'h' and 'n'; 'Doe' is clear."
            * Inference/Labeling: "Value 'Global Exporters Inc.'; Confidence 0.90: Inferred as Seller Name from prominent placement in header, no explicit 'Seller:' label."
            * Formatting Issues: "Value '15/07/2024'; Confidence 0.92: Date format DD/MM/YYYY clearly extracted; slight ink bleed around numbers."
            * Contextual Conflict: "Value for 'Net Weight' is '1500 KG', but 'Gross Weight' is '1400 KG'; Confidence 0.60 for Net Weight due to inconsistency requiring review."
        * If confidence is 0.99-1.00, reasoning can be succinct, e.g., "All characters perfectly clear, machine-printed, explicit standard label, contextually validated."
        * If `"value"` is `null`, briefly explain *why* (e.g., "No field labeled 'HS Code' or any recognizable tariff code found on any page.", "The section for 'Intermediary Bank Details' was present but explicitly marked 'Not Applicable'.").

**Example of Expected JSON Output Structure (Reflecting Stricter Confidence & Generic Reasoning):**
(Note: Actual field names will match those provided in the 'Fields to Extract' list for the specific '{doc_type}')

```json
{{
  "INVOICE_NO": {{
    "value": "INV-XYZ-789",
    "confidence": 0.99,
    "reasoning": "Extracted from explicit label 'Invoice #:' on page 1, header. All characters are machine-printed, clear, and unambiguous. Format matches typical invoice numbering."
  }},
  "BUYER_NAME": {{
    "value": "Generic Trading Co.",
    "confidence": 1.00,
    "reasoning": "Extracted from 'BILL TO:' section, page 1. All characters perfectly clear, machine-printed, standard label, contextually validated."
  }},
  "HS_CODE": {{
    "value": null,
    "confidence": 0.0,
    "reasoning": "No field labeled 'HS Code', 'HTS Code', or 'Tariff Code', nor any recognizable HS code pattern, found on any page of the document."
  }},
  "PAYMENT_TERMS": {{
    "value": "Net 30 days from date of invoice",
    "confidence": 0.98,
    "reasoning": "Extracted from section labeled 'Payment Terms:' on page 2. Text is clearly printed and directly associated with a standard label. All characters legible."
  }},
  "DATE_AND_TIME_OF_RECEIPT_OF_DOCUMENT": {{
    "value": "2024-07-16 11:25",
    "confidence": 0.90,
    "reasoning": "Date '16 JUL 2024' clearly visible in a bank's 'RECEIVED' stamp on page 1. Time '11:25' also part of the stamp, clearly printed. Converted date to ISO format. Confidence slightly below max due to typical minor imperfections in stamp quality."
  }}
  // ... (all other requested fields for the '{doc_type}' document would follow this structure)
}}
```

Important: Your response must be ONLY the valid JSON object. No greetings, apologies, or any text outside the JSON structure.
""") # End of EXTRACTION_PROMPT_TEMPLATE

    # --- Derived Path Properties ---
    @property
    def TEMP_DIR(self) -> Path:
        return Path(self.TEMP_DIR_PATH_STR)

    @property
    def OUTPUT_FILENAME(self) -> Path:
        # This assumes OUTPUT_FILENAME_STR is just the name, not a full path.
        # If it could be a full path, Path(self.OUTPUT_FILENAME_STR) is fine.
        # For consistency with original, it's just a filename.
        return Path(self.OUTPUT_FILENAME_STR)


    @property
    def LOG_FILE(self) -> Path:
        return Path(self.LOG_FILE_PATH_STR)

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",                # Load .env file
        env_file_encoding='utf-8',      # Encoding for .env file
        extra='ignore',                 # Ignore extra fields from environment
        case_sensitive=False            # Environment variable names are case-insensitive
    )

# --- Instantiate settings ---
# This single 'settings' instance will be imported by other modules.
settings = AppSettings()