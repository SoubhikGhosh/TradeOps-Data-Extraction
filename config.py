# config.py
import os
from dotenv import load_dotenv
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

load_dotenv()  # Optional: Load environment variables from a .env file

# --- Vertex AI Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "hbl-uat-ocr-fw-app-prj-spk-4d")
LOCATION = "asia-south1"
# Use a powerful multimodal model capable of handling PDFs and complex instructions
MODEL_NAME = os.getenv(
    "GEMINI_MODEL", "gemini-1.5-pro-002"
)  # Or gemini-1.5-flash / newer appropriate model
API_ENDPOINT = (
    f"{LOCATION}-aiplatform.googleapis.com"  # Often not needed if default is correct
)

# --- Supported File Types ---
SUPPORTED_MIME_TYPES = {
    "application/pdf": "PDF",
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG",
}

SUPPORTED_FILE_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg"]

# --- Safety Settings ---
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- Processing Configuration ---
MAX_WORKERS = 4  # Adjust based on CPU cores and API limits for parallel processing
TEMP_DIR = "temp_processing"
OUTPUT_FILENAME = "extracted_data.xlsx"

# --- Logging Configuration ---
LOG_FILE = "app_log.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Document Field Definitions with Descriptions ---
DOCUMENT_FIELDS = {
    "CRL": [
        {
            "name": "DATE & TIME OF RECEIPT OF DOCUMENT",
            "description": """
    **You are an expert document analysis system. Your task is to extract the Date and Time of Document Receipt from a circular seal present in the document image.**

    **Objective:** Accurately locate a specific circular mechanical stamp (seal) in the document and precisely extract the date and time indicated by its features. The time extraction requires meticulous application of rules based on the **leading edge** of an arrow pointer and a 24-slot/4-segment time mechanism, particularly a specific rule for interpreting alignment with major hour lines. Ignore any extraneous text on the seal that is not part of the date or the time scale markings.

    **I. Seal Identification and Features:**
        1.  **Primary Task:** Locate the circular seal.
        2.  **Seal Structure:**
            * The seal is composed of two concentric circles.
            * **Inner Circle (Date Display):** Contains the date. This date will be in a standard recognizable format (e.g., YY-MM-DD, DD MON YY, MM/DD/YY).
            * **Pointer (Arrow Icon):** A distinct **triangular pointer (arrow icon)** is positioned near the inner circle, typically centered above the date. The base of this triangle faces the center of the seal, and its sharp apex points radially outwards towards the outer time scale ring.
            * **Outer Ring (24-Hour Time Scale):** The annulus between the two circles serves as a 24-hour time scale. This scale is divided into 24 major 'slots,' representing hours 00 through 23.
                * **Major Hour Lines:** These slots are demarcated by prominent lines (major hour lines) indicating the start of each hour (e.g., the line for '01', '02', ..., '12', ..., '23', '00'). Assume these are arranged sequentially in a clockwise direction. The '00' hour is typically located at the 12 o'clock (top) position of the seal.
                * **Minute Segments within each Hour Slot:** Each major hour slot (e.g., the slot for hour 'H', which is the space between the major hour line 'H' and the subsequent major hour line 'H+1') is further subdivided into **four equal 15-minute segments**. The determination of these segments depends on the position of the pointer's leading edge *after it has passed the major hour line 'H'*:
                    * **First Segment (H:00):** Begins immediately after the major hour line 'H' and ends just before the first minor tick mark (or visual equivalent).
                    * **Second Segment (H:15):** Begins at the **first** distinct minor tick mark (a short line or point) after major hour line 'H' and ends just before the second minor tick mark.
                    * **Third Segment (H:30):** Begins at the **second** distinct minor tick mark after major hour line 'H' and ends just before the third minor tick mark.
                    * **Fourth Segment (H:45):** Begins at the **third** distinct minor tick mark after major hour line 'H' and ends just before the next major hour line ('H+1').

    **II. Time Determination Logic (Crucial - Adhere Strictly to "Leading Edge" and Case A/B Rules):**
        * The time is indicated by the **leading edge of the triangular pointer**. The "leading edge" is defined as the edge of the pointer that moves foremost in a clockwise direction as it sweeps across the time scale.
        * **Hour and Minute Determination is based on two mutually exclusive cases:**

            * **Case A: Leading Edge Precisely ON a Major Hour Line:**
                * **Condition:** The pointer's **leading edge** aligns *exactly* with a major hour line that demarcates the beginning of an hour 'H' on the dial (e.g., the line labeled '01', '02', ..., '12', ..., '23', '00').
                * **Interpretation:** The time indicated is the completion of the *previous* hour.
                * **Hour Calculation:** The hour is **(H-1)**. (If the dial shows H='00', the indicated hour is '23'. If H='01', the hour is '00', etc.)
                * **Minute Calculation:** The minutes are always **:00**.
                * **Example:** If the leading edge is exactly on the '12' major hour line (where '12' represents H), the time is 11:00. If exactly on the '01' major hour line, the time is 00:00. If exactly on the '00' major hour line, the time is 23:00.

            * **Case B: Leading Edge *WITHIN* an Hour Slot (i.e., Past a Major Hour Line):**
                * **Condition:** The pointer's **leading edge** has moved *past* a major hour line 'H' and is now positioned *within* the slot corresponding to hour 'H' (i.e., in the space between major hour line 'H' and the next major hour line 'H+1').
                * **Interpretation:** The current hour is 'H', and minutes are determined by the segment occupied by the leading edge.
                * **Hour Calculation:** The hour is **'H'** (the hour of the slot the leading edge is currently within).
                * **Minute Calculation:** Determine which of the four 15-minute segments of hour 'H' (as defined in Section I, Point 2, Sub-point "Minute Segments") the pointer's **leading edge** currently occupies or has just entered:
                    * Leading edge in the first segment (just past major line 'H', before 1st minor tick): Minutes are **:00** (Time is H:00).
                    * Leading edge in the second segment (past 1st minor tick, before 2nd minor tick): Minutes are **:15** (Time is H:15).
                    * Leading edge in the third segment (past 2nd minor tick, before 3rd minor tick): Minutes are **:30** (Time is H:30).
                    * Leading edge in the fourth segment (past 3rd minor tick, before next major hour line 'H+1'): Minutes are **:45** (Time is H:45).

    **III. Extraction Steps:**
        1.  **Locate the Seal:** Identify the circular seal as described in Section I.
        2.  **Extract Date:** Perform OCR on the inner circle of the seal to reliably read and extract the full date.
        3.  **Determine Time (Apply Case A/B Logic):**
            * Carefully examine the precise position of the pointer's **leading edge** relative to the major hour lines and any minor tick marks on the outer time scale.
            * First, rigorously assess if **Case A** applies (i.e., leading edge is *exactly* on a major hour line). If so, calculate the time as (H-1):00.
            * If Case A does not apply, then **Case B** must apply. Identify the current hour slot 'H' that the leading edge is within. Then, determine which of the four 15-minute segments (:00, :15, :30, :45) of hour 'H' the leading edge occupies to set the minutes.
            * Combine the extracted/calculated hour and minutes into HH:MM format (24-hour clock). Strive for exact alignment and strict rule application.

    **IV. Handling Imperfections and Ambiguity:**
        * Utilize all visible portions of the seal for extraction.
        * If image blur, low resolution, or damage makes the exact position of the leading edge genuinely ambiguous between two distinct interpretations (e.g., distinguishing between Case A and Case B when the edge is extremely close to a major line, or between two minute segments), this inherent ambiguity should be noted in your response.
        * However, the primary objective is to meticulously apply the defined rules assuming a clear visual reading is possible. "Estimation" should only be employed to resolve minor visual ambiguities to fit one of the established rules, not to create new rules or deviate from the specified logic.

    **V. Output Requirements:**
        * **Format:** Provide the extracted **Date** and the determined **Time** combined into a single string in the format: **DD-MM-YYYY HH:MM** (24-hour clock). Ensure the date part reflects the day, month, and full four-digit year.
        * **If Not Found:** If the described seal is not found in the document, or if the date or time cannot be reliably determined according to the rules, return **null**.
        * **Ambiguity Note:** If a critical judgment call was necessary due to unavoidable visual ambiguity (as described in Section IV), append a brief note explaining the ambiguity and the choice made. Example: "Ambiguity: Pointer leading edge very close to major line '03'; interpreted as just past line, resulting in 03:00 rather than 02:00."
    """,
        },
        {
            "name": "CUSTOMER REQUEST LETTER DATE",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Customer Request Letter Date from the document.**

    **Objective:** Accurately locate and extract the specific date on which the customer (also referred to as the applicant) formally prepared and dated their request letter or application form. This is considered the authorship date of the letter by the customer.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for explicit labels such as 'Date:', 'Dated:', 'Letter Date:'.
        * **Location:** This date is typically found in the header section of the letter, often near the applicant's details (name, address) or below the sender's address and before the recipient's address or salutation. It can also be near the signature block if not in the header.
        * **Context:** It is the primary date associated with the creation of the letter by the customer/applicant, not necessarily dates mentioned within the body of the letter referring to other events (e.g., invoice dates, shipment dates, unless this letter *is* the invoice).
    2.  **Data Format:**
        * The date can appear in various formats (e.g., "DD-MM-YYYY", "MM-DD-YYYY", "YYYY-MM-DD", "DD MON YYYY", "Month D, YYYY", "DD/MM/YY", "MM.DD.YY").
    3.  **What to Extract:**
        * Extract the complete date.
        * If the date includes a textual month name (e.g., "October"), ensure it's captured correctly.

    **Examples of potential text segments containing the date:**
    * "Date: 03-10-2023"
    * "Dated: October 3, 2023"
    * "2023-10-03"
    * "03 OCT 2023"
    * "22/12/2023"

    **Output Requirements:**
    * **Format:** Standardize and return the extracted date in **DD-MM-YYYY** format. For example, if "October 3, 2023" is found, output "03-10-2023". If "2023/10/03" is found, output "03-10-2023". If "22/12/23" is found, output "22-12-2023".
    * **If Not Found:** If the Customer Request Letter Date cannot be clearly identified or is absent from the document, return **null**.
    * **Clarification:** Distinguish this from other dates like 'Date of Receipt' or 'Invoice Date' unless the document structure clearly indicates this is the primary letter date.
    """,
        },
        {
            "name": "BENEFICIARY NAME",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Beneficiary Name from the document.**

    **Objective:** Accurately locate and extract the full, official legal name of the beneficiary. The beneficiary is the party (e.g., exporter, seller, service provider) designated to receive funds or benefit from the transaction described in the document.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for explicit labels such as 'Beneficiary:', 'Beneficiary Name:', 'Name of Beneficiary:', 'Payee:', 'Pay to:', 'To (Beneficiary):', 'Supplier Name:', 'Exporter Name:', 'Seller Name:'. The label might appear directly preceding or above the name.
        * **Context:** The Beneficiary Name is often found in sections detailing payment instructions, beneficiary details, or recipient information. It is typically a company name or an individual's full name.
    2.  **What to Extract:**
        * Extract the **complete and exact name** as it appears in the document.
        * Ensure to include all parts of the name, including any legal suffixes (e.g., Ltd., Inc., LLC, Corp., GmbH, Pvt. Ltd., S.A., Co., Ltd.).
        * Do not abbreviate or omit any part of the name.
    3.  **Distinguishing from Other Entities:**
        * Be careful to distinguish the Beneficiary Name from the Applicant Name, Bank Names, or any intermediary names. The context and labels are key.

    **Examples of Beneficiary Name text:**
    * "Beneficiary: Global Export Services Ltd."
    * "Payee: Jane Doe International Inc."
    * "Supplier Name: Acmetronics GmbH & Co. KG"
    * "To: John Michael Smith"
    * "Shandong Keerte CNC Technology Co., Ltd."

    **Output Requirements:**
    * **Format:** Return the extracted name as a **single string**.
    * **If Not Found:** If the Beneficiary Name cannot be clearly identified or is absent from the document, return **null**.
    * **Completeness:** Prioritize extracting the most complete version of the name if multiple similar mentions are found; usually, the one associated with a clear label is most reliable.
    """,
        },
        {
            "name": "BENEFICIARY ADDRESS",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Beneficiary Address from the document.**

    **Objective:** Accurately locate and extract the complete mailing address of the beneficiary.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** The address is usually found directly below or next to the 'Beneficiary Name'. Look for labels like 'Beneficiary Address:', 'Address:'. Sometimes it's part of a larger 'Beneficiary Details' section without a specific "Address" label for the address block itself.
        * **Location:** Typically follows the Beneficiary Name.
        * **Content:** The address should include components like street name and number, city, state/province, postal code, and potentially the country (though country might be a separate field).
    2.  **What to Extract:**
        * Extract the **full and complete mailing address** as a single continuous string.
        * Include all lines of the address. Preserve line breaks with a space or newline character if meaningful for readability, but the final output should be one string.
        * Do not include the Beneficiary Name itself in the address string if it's clearly separate.

    **Examples of Beneficiary Address text:**
    * "123 International Parkway, Suite 500, Export City, EC 12345, Globalia"
    * "No. 2707, Nanerhuan Road, Dongcheng Street, Linqu County, Weifang City, Shandong Province, China. P.C. 262600"

    **Output Requirements:**
    * **Format:** Return the extracted address as a **single string**.
    * **If Not Found:** If the Beneficiary Address cannot be clearly identified or is absent, return **null**.
    * **Completeness:** Ensure all components of the address (street, city, postal code, etc.) visible are extracted.
    """,
        },
        {
            "name": "BENEFICIARY COUNTRY",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Beneficiary Country from the document.**

    **Objective:** Accurately locate and extract the country where the beneficiary is officially located or registered.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Country:', 'Beneficiary Country:'.
        * **Location:** Often found as the last part of the beneficiary's full address, or on a separate line near the address. It might also be in a dedicated 'Country' field within a beneficiary details section.
        * **Context:** This should be the country associated with the beneficiary's main address.
    2.  **What to Extract:**
        * Extract the full name of the country.
        * Avoid extracting city or state names unless they are part of a country name (e.g., "United States of America").

    **Examples of Beneficiary Country text:**
    * "Germany"
    * "Singapore"
    * "China"
    * "United Kingdom"

    **Output Requirements:**
    * **Format:** Return the extracted country name as a **string**.
    * **If Not Found:** If the Beneficiary Country cannot be clearly identified or is absent, return **null**.
    * **Standardization (Optional but helpful):** If possible, standardize to common country names (e.g., "USA" to "United States"). If unsure, extract as written.
    """,
        },
        {
            "name": "REMITTANCE CURRENCY",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Remittance Currency code from the document.**

    **Objective:** Accurately locate and extract the three-letter ISO 4217 currency code (e.g., USD, EUR, INR) for the funds requested for remittance.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Currency:', 'CCY:', 'Transaction Currency:', 'Remittance Currency:', 'Amount Currency:'.
        * **Location:** Typically found directly next to, or preceding, the 'REMITTANCE AMOUNT'. It may also be in a dedicated currency field or column.
        * **Format:** The target is a **three-letter code**, usually uppercase (e.g., USD, EUR, GBP, JPY, INR, AUD, CAD, CHF).
    2.  **What to Extract:**
        * Extract only the three-letter ISO 4217 currency code.
        * Avoid extracting currency symbols (e.g., $, €, £) unless the three-letter code is absent and the symbol is the only indicator (in which case, note this or attempt to map to code if instructed). Prefer the code.

    **Examples of Remittance Currency text:**
    * "Currency: USD"
    * "EUR 10,000.00" (Extract "EUR")
    * "Amount (INR): 50000" (Extract "INR")
    * "CCY: JPY"

    **Output Requirements:**
    * **Format:** Return the extracted three-letter ISO 4217 currency code as a **string** (e.g., "USD").
    * **If Not Found:** If the currency code cannot be clearly identified, return **null**.
    * **Consistency:** Ensure the extracted code is indeed a standard currency code.
    """,
        },
        {
            "name": "REMITTANCE AMOUNT",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Remittance Amount from the document.**

    **Objective:** Accurately locate and extract the principal monetary value of the transaction requested for remittance, in the specified 'REMITTANCE CURRENCY'.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Amount:', 'Transaction Amount:', 'Remittance Amount:', 'Value:', 'Net Amount:', 'Amount to Remit:'.
        * **Location:** Typically found near the 'REMITTANCE CURRENCY'. It's often in a prominent position related to the payment details.
        * **Format:** The value will be numerical, potentially including decimal points for cents/sub-units and commas as thousands separators.
    2.  **What to Extract:**
        * Extract the **numerical value** of the amount.
        * Include decimal places if present.
        * Ensure to capture the full amount.
    3.  **Clarification:**
        * This is the principal amount to be remitted, distinct from any fees, charges, or tax amounts unless explicitly stated to be part of the remittance amount.
        * If an amount is written both in figures and words, extract the amount in **figures**.

    **Examples of Remittance Amount text:**
    * "Amount: 50,000.00" (Extract "50000.00")
    * "Transaction Amount: 12,345.67" (Extract "12345.67")
    * "Value: 21712.18" (Extract "21712.18")
    * "USD 638.40" (Extract "638.40" - currency is a separate field)

    **Output Requirements:**
    * **Format:** Return the extracted amount as a **numerical value (float or decimal type if possible, otherwise string representing the number, e.g., "21712.18")**. Remove any currency symbols, currency codes, or thousands separators (like commas) before converting to a number, but retain the decimal separator.
    * **If Not Found:** If the Remittance Amount cannot be clearly identified, return **null**.
    """,
        },
        {
            "name": "BENEFICIARY ACCOUNT NO / IBAN",
            "description": """
            **You are an expert data extraction system. Your task is to extract the Beneficiary's Bank Account Number or IBAN from the document.**

            **Objective:** Accurately locate and extract the specific bank account number or International Bank Account Number (IBAN) designated for the beneficiary, where funds are to be credited.

            **Guidance for Extraction:**
            1.  **Primary Focus - Beneficiary Context:**
                * Concentrate on sections explicitly labeled for 'Beneficiary', 'Beneficiary Bank Details', 'Payment To', 'Remit To', or similar terms indicating the recipient of funds.
                * **Crucially, distinguish** this from any account details provided for the payer, sender, buyer, or ordering customer. Do NOT extract account numbers belonging to these parties.

            2.  **Identification Cues:**
                * **Labels:** Look for labels directly associated with the beneficiary's account, such as 'Account No.:', 'A/C No.:', 'Account Number:', 'Beneficiary Account No.:', 'IBAN:', 'Beneficiary IBAN:', 'Acc No:', 'A/c ID:', 'Payee Account:', 'Recipient Account No:'.
                * **Location:** This information is typically found within the clearly demarcated 'Beneficiary Details' or 'Beneficiary Bank Details' section, often in proximity to the beneficiary's name and bank name.
                * **Format:**
                    * **Account Numbers:** Vary widely (can be numeric, alphanumeric, may contain hyphens or spaces).
                    * **IBANs:** Have a standard structure: a two-letter country code, two check digits, and then a country-specific Basic Bank Account Number (BBAN) which can be up to 30 alphanumeric characters (e.g., DE89370400440532013000, GB29NWBK60161331926819).

            3.  **What to Extract:**
                * Extract the **complete and exact account number or IBAN** as it appears visually.
                * Include all alphanumeric characters and any embedded hyphens or spaces if they are part of the presented number. (Normalization by removing spaces/hyphens can be a separate, subsequent step if required).
                * If both an IBAN and a local account number are clearly present *for the same beneficiary*, prioritize extracting the **IBAN**.

            **Examples of Account No / IBAN text:**
            * "IBAN: DE89370400440532013000"
            * "Account No.: 001-234567-890"
            * "A/C No: 218246110956"
            * "Beneficiary Account Number: FR7630006000011234567890189"
            * "Payee Account: SG9000123456789012"

            **Output Requirements:**
            * **Format:** Return the extracted account number or IBAN as a single **string**.
            * **If Not Found:** If the Beneficiary Account No / IBAN cannot be clearly and confidently identified according to the guidelines above, or if the relevant section is missing from the document, return **null**.
            * **Ambiguity:** If multiple distinct numbers could fit the beneficiary account description in a confusing manner, and a single definitive one cannot be chosen, it is preferable to return **null** rather than a guess to maintain data accuracy.
            """,
        },
        {
            "name": "BENEFICIARY BANK",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Beneficiary Bank Name from the document.**

    **Objective:** Accurately locate and extract the full official name of the bank where the beneficiary holds their account.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Beneficiary Bank:', 'Bank Name:', 'Beneficiary's Bank:', 'Receiving Bank:', 'Payee Bank:'.
        * **Location:** Usually listed in the 'Beneficiary Bank Details' section, typically near the beneficiary's account number and the bank's SWIFT code or address.
        * **Content:** This is the name of a financial institution.
    2.  **What to Extract:**
        * Extract the **full and official name** of the bank.
        * Include any suffixes like 'Ltd.', 'PLC', 'Inc.', 'N.A.', 'AG'.
        * Be mindful of common bank name components (e.g., "Bank of ...", "... Commercial Bank", "First National ...").
    3.  **Handling Variations:**
        * Bank names can be long. Ensure the full name is captured.
        * The document might use slight variations or abbreviations; try to capture the most complete official name presented. The term "Beneficiary Bank" is standard, but documents might have typos like "Beneficary Bank"; your search should be robust to minor variations if context is clear.

    **Examples of Beneficiary Bank text:**
    * "Beneficiary Bank: Global Standard Commercial Bank"
    * "Bank Name: Exporter's First Union Bank PLC"
    * "BANK OF CHINA LINQU SUB BRANCH"

    **Output Requirements:**
    * **Format:** Return the extracted bank name as a **string**.
    * **If Not Found:** If the Beneficiary Bank name cannot be clearly identified, return **null**.
    """,
        },
        {
            "name": "BENEFICIARY BANK ADDRESS",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Beneficiary Bank Address from the document.**

    **Objective:** Accurately locate and extract the complete mailing address of the beneficiary's bank.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels like 'Beneficiary Bank Address:', 'Bank Address:'. It's often found directly under or next to the 'Beneficiary Bank Name'. Sometimes it's part of a larger 'Beneficiary Bank Details' section without a specific "Address" label for the address block itself.
        * **Location:** Typically follows the Beneficiary Bank Name and is near other bank details like SWIFT code.
        * **Content:** The address should include components like street name and number, city, and country. Postal codes and states/provinces may also be present.
    2.  **What to Extract:**
        * Extract the **full and complete mailing address** of the bank as a single continuous string.
        * Include all lines of the address.
        * Do not include the Beneficiary Bank Name itself in the address string if it's clearly separate.

    **Examples of Beneficiary Bank Address text:**
    * "789 Finance Avenue, Central Banking District, Capital City, CB 67890, Globalia"
    * "NO 188 MINZHU ROAD OF LINQU COUNTRY SHANDONG P.R.C"

    **Output Requirements:**
    * **Format:** Return the extracted bank address as a **single string**.
    * **If Not Found:** If the Beneficiary Bank Address cannot be clearly identified or is absent, return **null**.
    * **Completeness:** Ensure all components of the address (street, city, country, etc.) visible are extracted.
    """,
        },
        {
            "name": "BENEFICIARY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Beneficiary Bank's identification code from the document.**

    **Objective:** Accurately locate and extract the unique identification code of the beneficiary's bank or bank branch. This code can be a SWIFT/BIC code, Sort Code, BSB number, IFSC code, or a similar bank identifier (e.g., ABA for US routing).

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'SWIFT Code:', 'BIC Code:', 'SWIFT/BIC:', 'Beneficiary Bank SWIFT:', 'Bank Code:', 'IFSC:', 'IFSC Code:', 'Sort Code:', 'BSB Number:', 'Routing No:', 'ABA:', 'Fedwire:', 'Chips UID:'. These labels are typically found near the beneficiary bank's name and address.
        * **Format:**
            * **SWIFT/BIC codes** are typically 8 or 11 characters long, alphanumeric (e.g., BANKGB2LXXX, DEUTDEFF).
            * **IFSC codes** (Indian Financial System Code) are 11 characters, alphanumeric, with the fifth character usually being '0' (e.g., HDFC0000123).
            * **Sort Codes** (UK & Ireland) are typically 6 digits, often formatted as XX-XX-XX (e.g., 12-34-56).
            * **BSB Numbers** (Australia) are 6 digits, often formatted as XXX-XXX (e.g., 012-345).
            * **Routing Numbers/ABA Numbers** (USA) are 9 digits.
        * **Context:** This information is usually located in the 'Beneficiary Bank Details' section, alongside the beneficiary bank's name and account number.
    2.  **What to Extract:**
        * Extract the **complete and exact code** as it appears.
        * If multiple types of codes are present for the beneficiary bank (e.g., both a SWIFT code and a local clearing code), prioritize the SWIFT/BIC code if available. If not, extract the available code. If the specific type of code is ambiguous but clearly a bank identifier, extract it.
        * Remove any prefixes or descriptive text that are not part of the code itself (e.g., if "SWIFT Code: BANKGB2LXXX" is found, extract "BANKGB2LXXX").

    **Examples of codes:**
    * SWIFT/BIC: "BANKGB2LXXX", "DEUTDEFFXXX", "BKCHCNBJ500"
    * IFSC: "BKID0001234"
    * Sort Code: "20-30-40"
    * BSB: "062-000"
    * Routing No: "123456789"

    **Output Requirements:**
    * **Format:** Return the extracted code as a **single string**.
    * **If Not Found:** If no such bank identification code for the beneficiary bank can be found, return **null**.
    * **Preference:** If both a SWIFT/BIC and another local code (IFSC, Sort Code, BSB, ABA) are clearly listed for the beneficiary bank, and no other instruction specifies which to take, prefer the SWIFT/BIC. If only one type is present, extract that.
    """,
        },
        {
            "name": "STANDARD DECLARATIONS AS PER PRODUCTS",
            "description": """
    **You are an expert data extraction system. Your task is to extract Standard Declarations made by the applicant from the document.**

    **Objective:** Accurately locate and extract the full text of any standard clauses, undertakings, legal statements, or compliance declarations made by the applicant (customer) within the request letter. These declarations often pertain to regulatory compliance (e.g., FEMA, AML, OFAC), the nature/purpose of the transaction, or affirmations of applicant responsibility.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Section Headers:** Look for specific section titles such as 'Declarations:', 'Undertakings:', 'Applicant's Declaration:', 'Terms and Conditions:', 'FEMA Declaration:', 'Compliance Statement:', 'Customer Declarations'.
        * **Structure:** Declarations are often presented as:
            * Numbered or bulleted lists of statements (e.g., using Roman numerals (I), (II), (III) or Arabic numerals 1., 2., 3.).
            * Paragraphs of formal text, typically located towards the end of the document, often preceding the applicant's signature block.
        * **Keywords within the text:** Phrases like "I/We declare...", "We confirm...", "It is undertaken...", "This transaction complies with...", "As per FEMA guidelines...", "AML compliant...", "We certify that...", "We agree..." can indicate declaration content.
    2.  **What to Extract:**
        * Extract the **full and complete text** of all relevant declaration clauses or statements.
        * If there are multiple distinct declarations (e.g., under different subheadings or numbered points), concatenate them ensuring all are captured. It is crucial to capture all such statements.
        * Include all sub-points or complete paragraphs that form part of a declaration.
    3.  **Scope:**
        * Focus on declarations made *by the applicant*. Do not extract general bank terms and conditions unless they are explicitly part of an applicant's signed declaration section.
        * This refers to statements that assert something or commit the applicant to certain conditions or acknowledgements related to the specific transaction or regulatory compliance.

    **Example of text segments to look for (these are indicative, the actual text will vary):**
    * "I/We declare that this transaction complies with all applicable regulations including FEMA, 1999."
    * "We undertake that we shall make physical import within 6 months from the date of remittance..."
    * "The purpose of this remittance is [purpose], and the information provided is true and correct."
    * "FATF Declaration (if applicable): We are aware that the above remittance is being made to a FATF listed country..."
    * Numbered lists under a 'Declarations' or 'Customer Declarations' heading.

    **Output Requirements:**
    * **Format:** Return the extracted declarations as a **single multi-line string**, preserving the general structure (like paragraph breaks and list formatting if simple) as much as possible.
    * **If Not Found:** If no such standard declarations by the applicant are found, return **null**.
    * **Completeness:** Aim to capture all text that clearly forms part of the applicant's declarations. If a declaration section is extensive, extract all of it.
    """,
        },
        {
            "name": "APPLICANT SIGNATURE",
            "description": """
    **You are an expert data extraction system. Your task is to identify evidence of the Applicant's Signature or Authorization on the document.**

    **Objective:** Determine if the document contains evidence of the applicant's (or their authorized signatory's) signature or formal authorization. This is not about extracting an image of the signature but rather textual confirmation or details related to it.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Location:** Typically found at the end of the letter or application form, in a dedicated signature block.
        * **Keywords/Labels:** Look for labels like 'Authorized Signatory', 'For [Applicant Company Name]', 'Signature', 'Signed by:', 'Name of Signatory:'.
        * **Content:**
            * A typed name of the person who signed (e.g., "John Smith, Managing Director").
            * A company stamp accompanied by a handwritten signature (you can only confirm presence).
            * Textual confirmation like "(Signed)" or "Electronically Signed".
            * A job title or capacity of the signatory (e.g., "Director", "Finance Manager").
    2.  **What to Extract:**
        * If a **typed name of the signatory** is present, extract that name and their title if available (e.g., "John Smith, Managing Director").
        * If a physical signature is clearly present but the name is not typed or legible, you can note "Signature Present" or "Signed".
        * If a company name is part of the signature block (e.g., "For ABC Corp"), extract that as part of the signatory details.
        * If the document is electronically signed, capture any textual representation of this (e.g., "Digitally signed by John Doe").

    **Examples of text indicating signature/authorization:**
    * "For TRIDISENO INDIA LTD., (Signature image) Director" (Extract: "For TRIDISENO INDIA LTD., Director" or "For TRIDISENO INDIA LTD., Signature Present, Director")
    * "Authorized Signatory: Jane Doe, CEO" (Extract: "Jane Doe, CEO")
    * "(Signed for and on behalf of Applicant Corp)" (Extract: "Signed for and on behalf of Applicant Corp")
    * "Customers Signature and Stamp" (If a signature is visible nearby, extract: "Signature Present")

    **Output Requirements:**
    * **Format:** Return the extracted information as a **string**. This could be the signatory's typed name and title, a confirmation like "Signature Present", or the text from the authorization line.
    * **If Not Found:** If no clear evidence of an applicant signature or authorization is found in typical locations, return **null**.
    * **Focus:** Prioritize extracting typed names and titles over generic "Signature Present" if both are available.
    """,
        },
        {
            "name": "APPLICANT NAME",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Applicant Name from the document.**

    **Objective:** Accurately locate and extract the full legal name of the individual or company submitting the request letter (the applicant or customer).

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Applicant Name:', 'Customer Name:', 'From:', 'Name of Applicant:', 'Remitter Name:'.
        * **Location:**
            * Often found in the letterhead at the top of the document.
            * Near the applicant's address.
            * In a specific field designated for applicant/customer name.
            * As the sender in the opening or closing of the letter.
        * **Context:** This is the entity initiating the request or transaction described in the document.
    2.  **What to Extract:**
        * Extract the **full and complete legal name** as it appears.
        * Include any legal suffixes (e.g., Ltd., Inc., LLC, Corp., Pvt. Ltd.).
        * If it's an individual, extract their full name.

    **Examples of Applicant Name text:**
    * "Applicant Name: Local Importers LLC"
    * "TRIDISENO INDIA LIMITED" (from letterhead or a field)
    * "Customer: Alpha Trading Enterprises"

    **Output Requirements:**
    * **Format:** Return the extracted name as a **string**.
    * **If Not Found:** If the Applicant Name cannot be clearly identified, return **null**.
    * **Distinction:** Differentiate from the Beneficiary Name. The Applicant is the one making the payment/request.
    """,
        },
        {
            "name": "APPLICANT ADDRESS",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Applicant Address from the document.**

    **Objective:** Accurately locate and extract the complete mailing address of the applicant (customer).

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels like 'Applicant Address:', 'Customer Address:', 'Address:'.
        * **Location:**
            * Often found in the letterhead, usually below or near the Applicant Name.
            * In a specific section for applicant/customer details.
        * **Content:** The address should include components like street name and number, city, state/province, postal code, and country.
    2.  **What to Extract:**
        * Extract the **full and complete mailing address** as a single continuous string.
        * Include all lines of the address. Preserve line breaks with a space or newline character if meaningful for readability, but the final output should be one string.
        * Do include the Applicant Name itself in the address string.

    **Examples of Applicant Address text:**
    * "456 Business Park, Suite 101, Metro City, MC 67890, Localia"
    * "Prashant Gosh, A-112, KUSHAL MANGAL IND'S ESTATE TUNGAR PHATA, NEAR WESTER EXPRESS HIGHWAY, VASAI EAST 401208"

    **Output Requirements:**
    * **Format:** Return the extracted address as a **single string**.
    * **If Not Found:** If the Applicant Address cannot be clearly identified or is absent, return **null**.
    * **Completeness:** Ensure all components of the address visible are extracted.
    """,
        },
        {
            "name": "APPLICANT COUNTRY",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Applicant Country from the document.**

    **Objective:** Accurately locate and extract the country where the applicant (customer) is officially located or registered.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Country:', 'Applicant Country:'.
        * **Location:**
            * Often found as the last part of the applicant's full address.
            * May be on a separate line near the address or in a dedicated 'Country' field within applicant details.
            * May not be explicitly mentioned as well, in which we need to make the obvious guess from the address.
        * **Context:** This should be the country associated with the applicant's main address.
    2.  **What to Extract:**
        * Extract the name of country and return it as the country code for that country.
        * Avoid extracting city or state names as the country.

    **Examples of Applicant Country text:**
    * "IN"
    * "UK"

    **Output Requirements:**
    * **Format:** Return the extracted country name as a **string**.
    * **If Not Found:** If the Applicant Country cannot be clearly identified or is absent, return **null**.
    * **Standardization (Optional):** If possible, standardize to common country codes. If unsure, extract as written.
    """,
        },
        {
            "name": "HS CODE",
            "description": """
    **You are an expert data extraction system. Your task is to extract the HS Code(s) from the document.**

    **Objective:** Accurately locate and extract the Harmonized System (HS) or Harmonized System Nomenclature (HSN) code(s) mentioned in the document. These are international standardized codes for classifying traded goods.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'HS Code:', 'HSN Code:', 'Tariff Code:', 'Customs Code:', 'HS Classification Code:'.
        * **Location:** Usually found in relation to the 'Description of Goods' or 'Type of Goods', or in a table detailing the products.
        * **Format:** HS codes are typically 6 to 10 digits, sometimes more with national subdivisions. They can be presented as a continuous string of digits or with punctuation like periods (e.g., "1234.56.78").
    2.  **What to Extract:**
        * Extract the **HS code digits**.
        * If there are multiple HS codes listed for different goods, extract all of them. They can be returned as a comma-separated string or an array if multiple are clearly distinct.
        * Remove any surrounding descriptive text or labels.

    **Examples of HS Code text:**
    * "HS Code: 69072300" (Extract "69072300")
    * "HSN Code: 8517.62.00" (Extract "85176200" or "8517.62.00" - clarify desired normalization of punctuation)
    * "Tariff Code: 900130" (Extract "900130")

    **Output Requirements:**
    * **Format:** Return the extracted HS code(s) as a **string**. If multiple codes are found, they can be concatenated with a comma and space (e.g., "69072300, 85176200"). Digits only is preferred, so "8517.62.00" should become "85176200".
    * **If Not Found:** If no HS Code is found, return **null**.
    """,
        },
        {
            "name": "TYPE OF GOODS",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Type of Goods description from the document.**

    **Objective:** Accurately locate and extract a general description of the merchandise, products, or services being imported or paid for.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Type of Goods:', 'Goods:', 'Nature of Goods:', 'Description of Goods/Services:', 'Product Description:'. Note that 'DESCRIPTION OF GOODS' is a separate, potentially more detailed field. This field aims for a more general typing.
        * **Location:** Often found in the main body of the request, detailing the purpose of the remittance. It might be near the HS Code.
        * **Content:** This is a brief textual description.
    2.  **What to Extract:**
        * Extract a concise but informative description of the goods or services.
        * Capture the primary nature of the items.

    **Examples of Type of Goods text:**
    * "Electronic Components for Manufacturing"
    * "Industrial Machinery Parts"
    * "Laser Cutting Machine"
    * "Raw Material / Capital Goods" (This is quite generic, try to find more specific if available, but extract this if it's the stated type)

    **Output Requirements:**
    * **Format:** Return the extracted description as a **string**.
    * **If Not Found:** If the Type of Goods cannot be clearly identified, return **null**.
    * **Distinction:** If both a general 'Type of Goods' and a more detailed 'DESCRIPTION OF GOODS' (another field) are present, ensure this field captures the more general categorization if one exists, otherwise it might overlap with 'DESCRIPTION OF GOODS'.
    """,
        },
        {
            "name": "DEBIT ACCOUNT NO",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Applicant's Debit Account Number from the document.**

    **Objective:** Accurately locate and extract the applicant's bank account number from which the principal transaction funds will be debited.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Debit Account No.:', 'Account to be Debited:', 'Source Account:', 'From Account:', 'Applicant Account No.:', 'Our Account No.:', 'A/C No:'.
        * **Location:** Typically found in the payment instruction section, often associated with the applicant's details or where funding for the transaction is specified.
        * **Context:** This account belongs to the applicant/customer making the payment, not the beneficiary. It is the source of funds for the main remittance amount.
        * **Format:** Account numbers vary widely (numeric, alphanumeric, may include hyphens).
    2.  **What to Extract:**
        * Extract the **complete and exact account number**.
        * Sometimes the account number might be prefixed by currency (e.g., "INR A/C No: 50200040555100"); extract only the account number "50200040555100".
        * Ensure it is explicitly linked to debiting for the main remittance, not just for fees (see 'FEE ACCOUNT NO').

    **Examples of Debit Account No text:**
    * "Debit Account No.: 123456789012"
    * "Account to be Debited: 00501000012345"
    * "INR A/C No: 50200040555100" (Extract "50200040555100")

    **Output Requirements:**
    * **Format:** Return the extracted account number as a **string**.
    * **If Not Found:** If the Debit Account Number for the principal amount cannot be clearly identified, return **null**.
    * **Clarification:** If the same account is used for both principal and fees, this field should still be populated. If a separate fee account is mentioned, that goes into 'FEE ACCOUNT NO'.
    """,
        },
        {
            "name": "FEE ACCOUNT NO",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Applicant's Fee Account Number from the document, if specified as different from the main debit account.**

    **Objective:** Accurately locate and extract the applicant's bank account number from which transaction fees or charges will be debited, IF this account is explicitly stated as being separate from the 'DEBIT ACCOUNT NO' used for the principal remittance amount.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels specifically indicating an account for charges, such as 'Fee Account No.:', 'Charges Account:', 'Account for Charges:', 'Debit Charges from A/c:'.
        * **Location:** Often found near the main debit account information or in a section discussing bank charges.
        * **Context:** The key is that this account is *specifically designated for fees* and is potentially different from the account debited for the remittance amount.
    2.  **What to Extract:**
        * Extract the **complete and exact account number** if a separate account for fees is explicitly mentioned.
        * If the document indicates fees are to be debited from the same account as the principal, or if no separate fee account is mentioned, this field should reflect that (e.g., by being null or by instruction).
        * Example context: "Account to be debited for charges of HDFC Bank Ltd. on us a/c no 50200040555100". Here, 50200040555100 is the fee account.

    **Examples of Fee Account No text (when different):**
    * "Fee Account No.: 987654321000"
    * "Charges to be debited from A/C: 112233445566"

    **Output Requirements:**
    * **Format:** Return the extracted fee account number as a **string**.
    * **If Not Found or Same as Debit:** If no *separate* fee account is specified, or if it's explicitly stated that fees are from the main debit account, return **null**. This field is specifically for a *different* fee account. If charges are from "on us a/c no X", then X is the fee account.
    """,
        },
        {
            "name": "LATEST SHIPMENT DATE",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Latest Shipment Date from the document.**

    **Objective:** Accurately locate and extract the latest date by which the goods must be shipped by the exporter/seller, as specified in the document (often related to terms in a Letter of Credit, purchase order, or proforma invoice referenced in the request).

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Latest Shipment Date:', 'Shipment by:', 'Latest Date of Shipment:', 'LSD:', 'Expected Date of Despatch:'.
        * **Location:** This date is typically found in sections detailing shipping terms, Letter of Credit conditions, or purchase order specifics.
        * **Format:** Dates can appear in various formats (e.g., DD-MM-YYYY, MM/DD/YYYY, YYYY-MON-DD).
    2.  **What to Extract:**
        * Extract the complete date.

    **Examples of Latest Shipment Date text:**
    * "Latest Shipment Date: 31-12-2024"
    * "Shipment by: 15/01/2025"
    * "Expected Date of Despatch: 30.05.2005"

    **Output Requirements:**
    * **Format:** Standardize and return the extracted date in **DD-MM-YYYY** format.
    * **If Not Found:** If the Latest Shipment Date cannot be clearly identified or is absent, return **null**.
    """,
        },
        {
            "name": "DISPATCH PORT",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Dispatch Port (Port of Loading) from the document.**

    **Objective:** Accurately locate and extract the name of the port, airport, or place from where the goods are to be dispatched or shipped (also known as Port of Loading).

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Port of Dispatch:', 'Port of Loading:', 'From Port:', 'Shipped From:', 'Origin Port:'.
        * **Location:** Usually found in the shipping details section of the document, often alongside Incoterms or transport details.
        * **Content:** This will be a geographical location name (e.g., a city name, specific port name).
    2.  **What to Extract:**
        * Extract the full name of the dispatch port/place.

    **Examples of Dispatch Port text:**
    * "Port of Dispatch: Port of Hamburg"
    * "Port of Loading: Shanghai Pudong Airport"
    * "From: Qingdao"

    **Output Requirements:**
    * **Format:** Return the extracted port name as a **string**.
    * **If Not Found:** If the Dispatch Port cannot be clearly identified or is absent, return **null**.
    """,
        },
        {
            "name": "DELIVERY PORT",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Delivery Port (Port of Discharge) from the document.**

    **Objective:** Accurately locate and extract the name of the port, airport, or place where the goods are to be delivered in the destination country (also known as Port of Discharge or Destination Port).

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Port of Delivery:', 'Port of Discharge:', 'To Port:', 'Destination Port:', 'Final Destination Port:'.
        * **Location:** Usually found in the shipping details section, often near the Dispatch Port information.
        * **Content:** This will be a geographical location name.
    2.  **What to Extract:**
        * Extract the full name of the delivery port/place.

    **Examples of Delivery Port text:**
    * "Port of Delivery: Port of New York"
    * "Port of Discharge: Nhava Sheva Port"
    * "Destination Port: Mumbai"

    **Output Requirements:**
    * **Format:** Return the extracted port name as a **string**.
    * **If Not Found:** If the Delivery Port cannot be clearly identified or is absent, return **null**.
    """,
        },
        {
            "name": "FB CHARGES",
            "description": """
    **You are an expert data extraction system. Your task is to extract who bears the Foreign Bank Charges from the document.**

    **Objective:** Accurately locate and extract the instruction indicating who is responsible for paying foreign bank charges. This is typically represented by a one-letter code: O (Beneficiary), or U (Applicant). 

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Foreign Bank Charges:', 'Details of Charges:', 'Charges Borne By:', 'Bank Charges:'.
        * **Location:** Often found in a section related to payment details or charges.
        * **Content:** The value is usually one of 'U', or 'B'. Sometimes options are presented with checkboxes or one is circled.
            * **U:** Beneficiary pays all charges (foreign bank charges deducted from remittance).
            * **O:** Applicant/Remitter pays all charges (beneficiary receives full amount).
            * **In case neither is ticked:** The default value remains 'U'.
    2.  **What to Extract:**
        * Extract the specific code (U, O) or word indicating the responsible party.
        * If options are given (e.g., checkboxes for "on us", "on beneficiary"), determine which is selected. "on us" generally maps to U, "on beneficiary" to O.

    **Examples of FB Charges text:**
    * "Foreign Bank Charges: BEN" (Extract "O")
    * "Details of Charges: OUR" (Extract "U")
    * A checkbox next to "OUR" is marked. (Extract "U")
    * "Charges: ()on us (X)on beneficiary" (Extract "O" as 'on beneficiary' is selected)

    **Output Requirements:**
    * **Format:** Return the extracted code/term as a **string** (e.g., "O", "U").
    * **If Not Found:** If the instruction for foreign bank charges cannot be clearly identified, return **"U"**.
    """,
        },
        {
            "name": "INTERMEDIARY BANK NAME",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Intermediary Bank Name from the document, if specified.**

    **Objective:** Accurately locate and extract the full official name of any intermediary or correspondent bank involved in the payment chain, if such information is provided.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Intermediary Bank:', 'Correspondent Bank:', 'Intermediary Bank Name:', 'Via Bank:'.
        * **Location:** Typically found in the payment instructions or bank details section, often separate from the Beneficiary Bank details.
        * **Context:** This is a bank that facilitates the transfer between the applicant's bank and the beneficiary's bank but is not the final recipient bank.
    2.  **What to Extract:**
        * Extract the **full and official name** of the intermediary bank.
        * Include any legal suffixes (Ltd., PLC, N.A., etc.).

    **Examples of Intermediary Bank Name text:**
    * "Intermediary Bank: Global Correspondent Bank PLC"
    * "Correspondent Bank: International Clearing Services Inc."

    **Output Requirements:**
    * **Format:** Return the extracted bank name as a **string**.
    * **If Not Found / Not Applicable:** If no intermediary bank is specified, or if the section is explicitly marked as N/A, return **null**.
    """,
        },
        {
            "name": "INTERMEDIARY BANK ADDRESS",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Intermediary Bank Address from the document, if specified.**

    **Objective:** Accurately locate and extract the complete mailing address of the intermediary or correspondent bank, if such information is provided.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** The address is usually found directly below or next to the 'Intermediary Bank Name'. Look for labels like 'Intermediary Bank Address:', 'Address:'.
        * **Location:** Typically follows the Intermediary Bank Name.
        * **Content:** The address should include components like street, city, and country.
    2.  **What to Extract:**
        * Extract the **full and complete mailing address** as a single continuous string.
        * Do not include the Intermediary Bank Name itself in the address string.

    **Examples of Intermediary Bank Address text:**
    * "1 Financial Square, Global City, GC1 2XX, Interland"

    **Output Requirements:**
    * **Format:** Return the extracted address as a **single string**.
    * **If Not Found / Not Applicable:** If no intermediary bank address is specified, or if the intermediary bank itself is not mentioned, return **null**.
    """,
        },
        {
            "name": "INTERMEDIARY BANK COUNTRY",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Intermediary Bank Country from the document, if specified.**

    **Objective:** Accurately locate and extract the country where the intermediary or correspondent bank is located, if such information is provided.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for 'Country:' within the intermediary bank details section.
        * **Location:** Often the last part of the intermediary bank's address or on a separate line.
    2.  **What to Extract:**
        * Extract the full name of the country.

    **Examples of Intermediary Bank Country text:**
    * "Switzerland"
    * "USA"

    **Output Requirements:**
    * **Format:** Return the extracted country name as a **string**.
    * **If Not Found / Not Applicable:** If no intermediary bank country is specified, or if the intermediary bank itself is not mentioned, return **null**.
    """,
        },
        {
            "name": "CUSTOMER SIGNATURE",
            "description": """
    **You are an expert data extraction system. Your task is to identify evidence of the Customer's (Applicant's) Signature or Authorization on the document. This field is synonymous with 'APPLICANT SIGNATURE'.**

    **Objective:** Determine if the document contains evidence of the customer's (applicant's or their authorized signatory's) signature or formal authorization. This is not about extracting an image of the signature but rather textual confirmation or details related to it.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels/Keywords:** Look for labels such as 'Customer Signature', 'Applicant Signature', 'Authorized Signatory', 'For [Customer Company Name]', 'Signature', 'Signed by:', 'Name of Signatory:', 'Customers Signature and Stamp'.
        * **Location:** Typically found at the end of the letter or application form, in a dedicated signature block.
        * **Content:**
            * A typed name of the person who signed (e.g., "Alice Brown, Finance Manager").
            * A company stamp often accompanies a handwritten signature.
            * Textual confirmation like "(Signed)", "Electronically Signed", or "s/d".
            * A job title or capacity of the signatory (e.g., "Director", "Proprietor").
    2.  **What to Extract:**
        * If a **typed name of the signatory** is present, extract that name and their title if available (e.g., "Alice Brown, Finance Manager").
        * If a physical signature is clearly present but the name is not typed or legible, you can note "Signature Present" or "Signed".
        * If a company name is part of the signature block (e.g., "For ABC Corp"), extract that as part of the signatory details.
        * If the document is electronically signed, capture any textual representation of this.

    **Examples of text indicating signature/authorization:**
    * "For [Customer Company Name], (Signed) Alice Brown, Finance Manager" (Extract: "Alice Brown, Finance Manager" or "For [Customer Company Name], Alice Brown, Finance Manager")
    * "Customers Signature and Stamp" (followed by a signature, extract: "Signature Present")
    * "Director" (under a signature line, if name is unreadable but title clear, extract: "Director, Signature Present")

    **Output Requirements:**
    * **Format:** Return the extracted information as a **string**. This could be the signatory's typed name and title, a confirmation like "Signature Present", or the text from the authorization line.
    * **If Not Found:** If no clear evidence of a customer signature or authorization is found, return **null**.
    * **Synonymy:** Treat this field as identical in purpose and extraction logic to 'APPLICANT SIGNATURE'.
    """,
        },
        {
            "name": "MODE OF REMITTANCE",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Mode of Remittance from the document.**

    **Objective:** Accurately locate and extract the method requested by the customer for making the payment to the beneficiary.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Mode of Payment:', 'Payment Method:', 'Remit by:', 'Method of Remittance:', 'Transaction Type:'.
        * **Location:** Usually found in the introductory part of the payment request or in a specific section for payment instructions.
        * **Content:** Common modes include 'SWIFT Transfer', 'Telegraphic Transfer (TT)', 'Demand Draft (DD)', 'Wire Transfer', 'Online Transfer'. Often presented as options where one is selected (e.g., by a checkmark or circling).
    2.  **What to Extract:**
        * Extract the specific mode of remittance selected or stated.
        * If options are given (e.g., "Request you to process the Import remittance by (X) Swift ( ) Demand Draft"), extract the selected option ("Swift").

    **Examples of Mode of Remittance text:**
    * "Mode of Payment: Telegraphic Transfer (TT)" (Extract "Telegraphic Transfer (TT)")
    * "Remit by: SWIFT Transfer" (Extract "SWIFT Transfer")
    * "Payment Method: Demand Draft" (Extract "Demand Draft")

    **Output Requirements:**
    * **Format:** Return the extracted mode of remittance as a **string**.
    * **If Not Found:** If the Mode of Remittance cannot be clearly identified, return **null**.
    """,
        },
        {
            "name": "COUNTRY OF ORIGIN",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Country of Origin of Goods from the document. This field is synonymous with 'COUNTRY OF ORIGIN OF GOODS'.**

    **Objective:** Accurately locate and extract the country where the goods being paid for were originally manufactured, produced, or grown.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Country of Origin:', 'Origin of Goods:', 'Made in:', 'Goods Origin:', 'Country of Supply:'.
        * **Location:** This information is often found in the section describing the goods, shipping details, or near the invoice details referenced in the letter.
        * **Context:** Refers to the manufacturing/production origin of the goods, not the beneficiary's country or applicant's country unless they happen to be the same as the goods' origin.
    2.  **What to Extract:**
        * Extract the full name of the country.

    **Examples of Country of Origin text:**
    * "Country of Origin: China"
    * "Made in: Germany"
    * "Origin of Goods: Vietnam"

    **Output Requirements:**
    * **Format:** Return the extracted country name as a **string**.
    * **If Not Found:** If the Country of Origin for the goods cannot be clearly identified, return **null**.
    * **Synonymy:** Treat this as equivalent to "COUNTRY OF ORIGIN OF GOODS".
    """,
        },
        {
            "name": "IMPORT LICENSE DETAILS",
            "description": """
    **You are an expert data extraction system. Your task is to extract Import License Details from the document.**

    **Objective:** Accurately locate and extract details of any specific import license or permit required for the goods, as mentioned in the document. This can include the license number, validity, issuing authority, or type.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Import Licence No.:', 'Import License Details:', 'Permit Number:', 'Authorization Details:', 'License No.:', 'IE (Import Exemption) Certificate No.:'.
        * **Location:** Typically found in sections related to regulatory compliance, goods description, or specific instructions for the import.
        * **Content:** This field should capture more than just a number if other details like validity dates or type of license are provided alongside the number.
    2.  **What to Extract:**
        * Extract all relevant details provided for the import license. This might be a license number, dates (issue/expiry), name of the issuing authority, or a description of the license type.
        * If multiple pieces of information are present (e.g., number and validity), combine them into a single descriptive string.

    **Examples of Import License Details text:**
    * "Import Licence No.: IL/COMM/2024/00123, Valid until: 31-12-2024" (Extract "Licence No: IL/COMM/2024/00123, Valid until: 31-12-2024")
    * "Permit Number: DGFT License XYZ123" (Extract "DGFT License XYZ123")
    * "Authorization: Special Import License No. A987 for restricted items" (Extract "Special Import License No. A987 for restricted items")

    **Output Requirements:**
    * **Format:** Return the extracted details as a **single string**.
    * **If Not Found / Not Applicable:** If no import license details are mentioned, or if the section is explicitly marked N/A, return **null**.
    """,
        },
        {
            "name": "CURRENCY AND AMOUNT OF REMITTANCE IN WORDS",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Remittance Amount written in words, including the currency, from the document.**

    **Objective:** Accurately locate and extract the total remittance amount (principal sum) as it is written out in words, along with the name of the currency also written in words.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Amount in Words:', 'Sum of (Currency) in Words:', 'Total Amount (in words):', 'Currency and Amount in Words:'.
        * **Location:** Typically found near the numerical 'REMITTANCE AMOUNT' and 'REMITTANCE CURRENCY', often presented for verification or legal formality.
        * **Content:** This will be a textual representation of the number and currency (e.g., "US Dollars One Hundred Thousand Only", "EURO Twenty-One Thousand Seven Hundred Twelve and Cents Eighteen Only").
    2.  **What to Extract:**
        * Extract the **full phrase** representing the currency and amount in words.
        * Ensure both the currency name (e.g., "US DOLLARS", "EURO", "INDIAN RUPEES") and the amount in words (e.g., "FIFTY THOUSAND ONLY", "TEN THOUSAND FIVE HUNDRED AND FIFTY POINT TWENTY FIVE") are included.
        * Pay attention to how cents/decimals are written (e.g., "and cents eighteen", "point two five").

    **Examples of text:**
    * "US DOLLARS FIFTY THOUSAND ONLY"
    * "EURO TEN THOUSAND FIVE HUNDRED AND FIFTY POINT TWENTY FIVE"
    * "Indian Rupees One Lac Twenty Five Thousand and Paise Fifty Only"
    * "(Six Hundred Thirty Eight & cent Forty Only)" - if currency is implied by context or nearby numerical amount with currency code. Extract full phrase along with currency name if separately stated or clearly inferable. If the currency name is part of the phrase as in "USD Six Hundred...", include it.

    **Output Requirements:**
    * **Format:** Return the extracted phrase as a **single string**.
    * **If Not Found:** If the amount in words cannot be clearly identified, return **null**.
    * **Completeness:** Capture the entire textual representation.
    """,
        },
        {
            "name": "INVOICE NO",
            "description": """
        **You are an expert data extraction system. Your primary task is to extract the Invoice Number specifically referenced within the body of a Customer Request Letter (CRL).**

        **Objective:** Accurately locate and extract the unique identification number of the Proforma Invoice or Commercial Invoice to which the remittance request (detailed in the CRL) pertains. This information must be explicitly stated or directly referenced *within the text of the CRL itself*.

        **Guidance for Extraction:**
        1.  **Source Focus:**
            * The invoice number must be found *within the content of the Customer Request Letter*. Do not infer it from other documents or assume it based on file names or references to attachments not directly quoted in the CRL text.

        2.  **Identification Cues within the CRL:**
            * **Labels:** Look for explicit labels such as 'Invoice No.:', 'Ref. Invoice:', 'Against Invoice No.:', 'Proforma Invoice No.:', 'PI No.:', 'Commercial Invoice No.:', 'Invoice Ref.:', 'Our Invoice:', 'Your Invoice:'.
            * **Contextual Phrases:** The invoice number might also follow phrases like "payment for invoice...", "remittance against...", "related to invoice...", "as per PI...".
            * **Location:** This reference is typically found in sections detailing the purpose of the payment, description of goods/services, or in specific fields/boxes designated for invoice details within the CRL. Pay particular attention to:
                * Boxes or sections with headers like "special ref no to be mentioned in swift" (the invoice number may be this special reference number or listed within this box).
                * Sections titled or related to "Shipment details & Proforma Invoice details mandatory for advance import " or similar phrasing indicating invoice information.
            * **Format (on document):** The invoice number as it appears on the document is commonly alphanumeric and can include characters like hyphens (-), slashes (/), periods (.), and spaces. It is an identifier issued by the beneficiary/exporter on their invoice.

        3.  **What to Extract:**
            * Extract the **complete and exact invoice number** precisely as it is referenced in the CRL. (The removal of hyphens is handled in the output formatting stage).

        **Examples of Invoice No. text found *within a CRL* (showing final extracted output):**
        * "Please process payment for Invoice No.: PI-2024-001" (Extract "PI2024001")
        * "Our remittance is for Ref. Invoice: EXPORTINV/789/ABC" (Extract "EXPORTINV/789/ABC")
        * "Funds transfer against Proforma Invoice KET-20231222 for machinery." (Extract "KET20231222")
        * "...as detailed in PI #INV00567/B..." (Extract "INV00567/B")

        **Output Requirements:**
        * **Format:** Return the extracted invoice number as a single **string**. Any hyphens (-) present in the identified invoice number should be removed from the final returned string. Other special characters (e.g., slashes (/), periods (.)) should be preserved as they appear in the source text.
        * **Multiple References Consideration:** If the CRL could explicitly list multiple distinct invoice numbers for a single remittance item and the requirement is to extract only one, prioritize the first one clearly labeled or referenced. If all distinct invoice numbers pertaining to the request need to be captured, this prompt might need adjustment for list output. (Current prompt targets a single, primary reference).
        * **If Not Found:** If no invoice number is explicitly referenced in the CRL text itself, return **null**.
        * **Avoid Inference:** If an invoice is mentioned merely as an attachment (e.g., "see attached invoice") but its number is not quoted or directly referenced in the CRL's text, this should also result in **null** for this field.
        """
        },
        {
            "name": "INVOICE DATE",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Invoice Date referenced in the Customer Request Letter (CRL).**

    **Objective:** Accurately locate and extract the date on which the referenced Proforma or Commercial Invoice (identified by 'INVOICE NO') was issued. This information is extracted *from the CRL itself* where it makes reference to an invoice's date.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels within CRL:** Look for labels such as 'Invoice Date:', 'Date of Invoice:', 'PI Date:', 'Proforma Invoice Date:'. Often found near the 'INVOICE NO'.
        * **Location:** Typically in the same section of the CRL as the referenced 'INVOICE NO'.
        * **Format:** Dates can appear in various formats.
    2.  **What to Extract:**
        * Extract the complete date from the CRL that refers to the invoice's issuance date.

    **Examples of Invoice Date text found in a CRL:**
    * "Invoice Date: 10-07-2024"
    * "Date of Invoice: July 10, 2024"
    * "Proforma Invoice KET-20231222 Date 22/12/2023" (Extract "22/12/2023")

    **Output Requirements:**
    * **Format:** Standardize and return the extracted date in **DD-MM-YYYY** format.
    * **If Not Found:** If no invoice date is referenced in the CRL, return **null**.
    """,
        },
        {
            "name": "INVOICE VALUE",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Invoice Value referenced in the Customer Request Letter (CRL).**

    **Objective:** Accurately locate and extract the total monetary value stated on the referenced Proforma or Commercial Invoice, as mentioned in the CRL. This value should ideally align with the 'REMITTANCE AMOUNT' if the full invoice value is being paid through this CRL.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels within CRL:** Look for labels such as 'Invoice Amount:', 'Invoice Total Value:', 'Value of Invoice:', 'Invoice Value:'. Often found near the 'INVOICE NO' and 'INVOICE DATE' within the CRL.
        * **Location:** Typically in the same section of the CRL as other referenced invoice details.
        * **Format:** Numerical value, potentially with currency information (though currency might be implicit from remittance currency).
    2.  **What to Extract:**
        * Extract the **numerical value** of the invoice as stated in the CRL.
        * Include decimal places if present.
        * The currency might be mentioned alongside (e.g., "USD 21712.18"); extract only the numerical part "21712.18". The currency of the invoice is often the same as 'REMITTANCE CURRENCY'.

    **Examples of Invoice Value text found in a CRL:**
    * "Invoice Amount: 21712.18" (Extract "21712.18")
    * "Invoice Total Value: USD 150,000.00" (Extract "150000.00")
    * "Amount USD 638.40" (under an invoice details section, extract "638.40")

    **Output Requirements:**
    * **Format:** Return the extracted amount as a **numerical value (float or decimal type if possible, otherwise string representing the number, e.g., "21712.18")**. Remove currency symbols/codes and thousands separators.
    * **If Not Found:** If no invoice value is referenced in the CRL, return **null**.
    """,
        },
        {
            "name": "EXCHANGE RATE",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Exchange Rate from the document, if specified.**

    **Objective:** Accurately locate and extract the exchange rate that has been applied or is requested for converting the remittance amount from one currency to another (e.g., from the applicant's local currency in their debit account to the foreign currency of the remittance).

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Exchange Rate:', 'Rate Applied:', 'FX Rate:', 'Conversion Rate:', 'Rate:'.
        * **Location:** Could be in a section detailing charges, currency conversion, or treasury deal information.
        * **Format:** Typically expressed as a ratio (e.g., "1 USD = 83.50 INR", "0.92 EUR/USD", or just a number like "83.50" if currencies are clear from context).
    2.  **What to Extract:**
        * Extract the full exchange rate information provided. This might be the rate itself or the equation.
        * If only a numerical rate is given, extract that. If a pair is given (e.g., "1 USD = 83.50 INR"), extract the full string.

    **Examples of Exchange Rate text:**
    * "Exchange Rate: 1 USD = 83.50 INR" (Extract "1 USD = 83.50 INR")
    * "Rate Applied: 0.92 EUR/USD" (Extract "0.92 EUR/USD")
    * "FX Rate: 75.1234" (Extract "75.1234")

    **Output Requirements:**
    * **Format:** Return the extracted exchange rate information as a **string**.
    * **If Not Found:** If no exchange rate is specified, return **null**.
    """,
        },
        {
            "name": "TREASURY REFERENCE NO",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Treasury Reference Number (or Deal ID) from the document, if specified.**

    **Objective:** Accurately locate and extract a unique reference number for a foreign exchange (forex) deal that might have been booked with the bank's treasury department to fix the exchange rate for this transaction. This is sometimes referred to as a Deal ID or FX Contract No.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Treasury Ref No:', 'Forex Deal ID:', 'FX Contract No.:', 'Deal ID:', 'Treasury Deal Reference:', 'Rate/contract booked with treasury - if any'.
        * **Location:** Usually found in sections related to exchange rates, treasury, or specific instructions for the bank.
        * **Format:** Typically an alphanumeric string, may include slashes or other characters.
    2.  **What to Extract:**
        * Extract the **complete and exact reference number or ID**.

    **Examples of Treasury Reference No text:**
    * "Treasury Ref No: TRSY/FX/2024/00567"
    * "Forex Deal ID: FXDEAL_98765"
    * "FX Contract No.: C123-456-B789"

    **Output Requirements:**
    * **Format:** Return the extracted reference number as a **string**.
    * **If Not Found / Not Applicable:** If no treasury reference number is specified, return **null**.
    """,
        },
        {
            "name": "SPECIFIC REFERENCE FOR SWIFT FIELD 70/72",
            "description": """
    **You are an expert data extraction system. Your task is to extract specific reference information intended for SWIFT message fields 70 or 72 from the document.**

    **Objective:** Accurately locate and extract any narrative, specific instructions, or reference information that the applicant explicitly requests to be included in the SWIFT payment message's Field 70 (Remittance Information / Purpose of Payment) or Field 72 (Sender to Receiver Information / Bank to Bank Information).

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Payment Reference (for SWIFT F70):', 'Message to Beneficiary Bank (F72):', 'Narrative for Beneficiary:', 'Details for SWIFT Field 70:', 'Instructions for Field 72:', 'Remittance Information:', 'Sender to Receiver Info:', 'Special Ref No to be mentioned in Swift'.
        * **Location:** This information is usually in a dedicated field or section where the applicant provides additional details for the payment message.
        * **Content:** This is typically free-form text and can include invoice numbers, purchase order numbers, contract references, purpose of payment in brief, or specific codes/instructions for the receiving bank or beneficiary.
    2.  **What to Extract:**
        * Extract the **full text** provided by the applicant for these SWIFT fields.
        * If information for both Field 70 and Field 72 is separately provided, you may need to combine them or prioritize based on further instruction (for now, capture all if clearly labeled). If a general "Special Reference for SWIFT" is given, extract that.

    **Examples of text for SWIFT fields:**
    * "Payment Reference (for SWIFT F70): /INV/PI-2024-001/ORDER/PO-ABC-123"
    * "Message to Beneficiary Bank (F72): URGENT PAYMENT PLEASE CREDIT IMMEDIATELY"
    * "Narrative for Beneficiary: PAYMENT FOR CONSULTANCY SERVICES AGREEMENT DATED 01-06-2024"
    * "Special Ref No to be mentioned in Swift: INV KET-20231222"

    **Output Requirements:**
    * **Format:** Return the extracted text as a **single string**. If multiple distinct instructions are found (e.g. for F70 and F72 separately), concatenate them with a clear separator like "F70: [text] | F72: [text]" or return as found.
    * **If Not Found:** If no such specific reference for SWIFT fields is provided, return **null**.
    """,
        },
        {
            "name": "DESCRIPTION OF GOODS",
            "description": """
    **You are an expert data extraction system. Your task is to extract the detailed Description of Goods or Services from the document.**

    **Objective:** Accurately locate and extract a detailed account or specific description of the goods or services for which the payment is being made, as stated in the customer's request letter. This description might be more detailed than the 'TYPE OF GOODS' field and should be a direct quote or close representation of what's in the application.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Description of Goods/Services:', 'Details of Import:', 'Goods Description:', 'Particulars of Goods/Services:', 'Nature and Description of Goods:'.
        * **Location:** Often found in the main body of the request, elaborating on the purpose of the remittance or the items being purchased. It can be a narrative sentence or a more structured list.
        * **Context:** This should be a comprehensive description. It might be near fields like 'HS CODE' or 'INVOICE NO'.
    2.  **What to Extract:**
        * Extract the **full and detailed textual description** of the goods or services.
        * Capture all relevant particulars provided by the applicant.
        * This might include model numbers, specifications, quantities if mentioned as part of the description, or the nature of services.

    **Examples of Description of Goods text:**
    * "Supply and installation of Model X-500 Industrial Compressor and associated spare parts as per PO #789."
    * "Annual Subscription Fee for Cloud Software Platform - Gold Tier for 10 users."
    * "Shandong Technology Co. Ltd - Laser Cutting Machine" (if this is the most detailed description provided under a relevant label).
    * "Goods description and HS classification code 84561100" (Here, extract the textual description, the HS code is a separate field. If description is just "Laser Cutting Machine", extract that).

    **Output Requirements:**
    * **Format:** Return the extracted description as a **string**.
    * **If Not Found:** If a detailed description of goods/services cannot be clearly identified, return **null**.
    * **Distinction from 'TYPE OF GOODS':** This field seeks a more specific and potentially longer description than the general 'TYPE OF GOODS'. If only one description is present, it might populate both fields or this one preferentially.
    """,
        },
        {
            "name": "TRANSACTION Product Code Selection",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Transaction Product Code Selection from the document.**

    **Objective:** Accurately locate and extract any specific internal bank product code or an explicit selection made by the applicant that identifies the type of financial product being used for this transaction (e.g., 'Import Advance Payment', 'Direct Import Bill Lodgement').

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Product Code:', 'Transaction Product:', 'Service Code:', 'Product Selected:'.
        * **Location:** This might be a specific field on a form, a checkbox selection, or a highlighted product name from a list of bank products. Often near the top of the application or in a designated "For Bank Use" section that the customer might also fill.
        * **Content:** Could be an alphanumeric code, or the name of the product itself if selected from options.
        * Often seen as options where one is ticked or selected, for example: "[X] ADVANCE DIRECT" or "Product: Import Advance".

    **Examples of Transaction Product Code Selection text:**
    * "Product Code: IMP-ADV-001"
    * "Transaction Product: TF-PAY-SIGHT"
    * "LADVANCE DIRECT" (where 'LADVANCE' or 'ADVANCE DIRECT' is selected or highlighted)
    * "[X] Import Collection Bill" (Extract "Import Collection Bill")

    **Output Requirements:**
    * **Format:** Return the extracted product code or selected product name as a **string**.
    * **If Not Found:** If no specific transaction product code or selection is identified, return **null**.
    """,
        },
        {
            "name": "TRANSACTION EVENT",
            "description": """
    **You are an expert data extraction system. Your task is to identify the Transaction Event being initiated by this document.**

    **Objective:** Identify and describe the specific event in the transaction lifecycle that this Customer Request Letter (CRL) represents. For a CRL, this is typically the initiation of a payment instruction or a request for a trade finance instrument.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Document Title/Purpose:** The overall nature of the document (e.g., "Request Letter for Import Payment", "Application for Remittance") is a strong indicator.
        * **Explicit Statements:** Look for phrases like "We wish to make an advance payment...", "Request you to process the Import remittance...", "Application for issuance of...".
        * **Context:** For a Customer Request Letter (CRL) concerning payments, the event is generally related to payment initiation or instruction.
    2.  **What to Extract:**
        * A concise description of the transaction event.
        * This field is often implicitly defined by the document type but look for explicit phrasing.

    **Examples of Transaction Event descriptions:**
    * "Outward Remittance Processing"
    * "Import Payment Initiation"
    * "Request for Advance Import Payment"
    * "Application for Telegraphic Transfer"

    **Output Requirements:**
    * **Format:** Return the description of the transaction event as a **string**.
    * **If Not Found (or to be standardized):** If not explicitly stated, a default like "Payment Instruction" or "Remittance Request" might be appropriate based on the document being a CRL. For now, extract if explicitly mentioned or clearly inferable. If truly ambiguous from text, return **null**.
    """,
        },
        {
            "name": "VALUE DATE",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Value Date from the document.**

    **Objective:** Accurately locate and extract the requested date for the funds to be debited from the applicant's account and/or credited to the beneficiary. This is the effective date for the transaction to take place.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Value Date:', 'Requested Settlement Date:', 'Debit Date:', 'Payment Date:', 'Execution Date:'.
        * **Location:** Typically found in the payment instruction section.
        * **Content:** This will be a specific date. It might also be a term like "Spot", "Today", or "Immediate".
    2.  **What to Extract:**
        * Extract the specified date or term.
        * If a specific date is given, capture it.
        * If a term like "Spot" is used, extract "Spot".

    **Examples of Value Date text:**
    * "Value Date: 17-07-2024"
    * "Settlement Date: Spot"
    * "Debit Date: 2024/07/18"

    **Output Requirements:**
    * **Format:** If a specific date is found, standardize and return it in **DD-MM-YYYY** format. If a term like "Spot" is found, return the **string "Spot"**.
    * **If Not Found:** If the Value Date cannot be clearly identified, return **null**.
    """,
        },
        {
            "name": "INCO TERM",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Incoterm from the document.**

    **Objective:** Accurately locate and extract the standardized three-letter trade term (e.g., FOB, CIF, EXW) that defines the responsibilities of the buyer and seller for the delivery of goods, costs, and risks. This term is usually mentioned in the CRL as it references a sales contract or invoice.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Incoterm:', 'Trade Term:', 'Delivery Term:'.
        * **Location:** Often found in the shipping details section, or near invoice details referenced in the CRL.
        * **Format:** Incoterms are typically three-letter acronyms (e.g., FOB, CIF, EXW, FCA, CPT, DAP, DDP). They are often followed by a named place or port (e.g., "CIF Shanghai", "FOB Qingdao Port").
    2.  **What to Extract:**
        * Extract the **three-letter Incoterm code** itself.
        * Also extract the **named place or port** associated with the Incoterm if it's provided directly alongside the term.

    **Examples of Incoterm text:**
    * "Incoterm: CIF (Destination Port)" (Extract "CIF (Destination Port)" or "CIF Destination Port")
    * "Trade Term: EXW (Seller's Factory, Anytown)" (Extract "EXW (Seller's Factory, Anytown)")
    * "FOB Shanghai" (Extract "FOB Shanghai")
    * "DAP Buyer's Premises London" (Extract "DAP Buyer's Premises London")

    **Output Requirements:**
    * **Format:** Return the extracted Incoterm and associated place as a **single string** (e.g., "CIF Shanghai").
    * **If Not Found:** If no Incoterm is specified, return **null**.
    """,
        },
        {
            "name": "THIRD PARTY EXPORTER NAME",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Third Party Exporter Name from the document, if applicable.**

    **Objective:** Accurately locate and extract the name of a third-party exporter if the goods are being exported by an entity that is different from the main beneficiary who is receiving the payment.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Third Party Exporter:', 'Actual Exporter (if different from Beneficiary):', 'Shipper (if not Beneficiary):', 'Goods Supplied By (if not Beneficiary):'.
        * **Location:** This information might be in a special remarks section, or near beneficiary/goods details, specifically indicated when the exporter is not the direct payee.
        * **Context:** This is relevant in triangular trade or when an agent is involved. The key is that this exporter is *not* the same as the 'BENEFICIARY NAME'.
    2.  **What to Extract:**
        * Extract the **full name** of the third-party exporter, including any legal suffixes.

    **Examples of Third Party Exporter Name text:**
    * "Third Party Exporter: Global Sourcing Agents Ltd."
    * "Actual Exporter: International Manufacturing Co. (goods supplied on behalf of Beneficiary XYZ)"

    **Output Requirements:**
    * **Format:** Return the extracted name as a **string**.
    * **If Not Found / Not Applicable:** If no third-party exporter is mentioned, or if the exporter is the same as the beneficiary, return **null**.
    """,
        },
        {
            "name": "THIRD PARTY EXPORTER COUNTRY",
            "description": """
    **You are an expert data extraction system. Your task is to extract the Third Party Exporter Country from the document, if applicable.**

    **Objective:** Accurately locate and extract the country of the third-party exporter, if such an exporter is named and is different from the main beneficiary.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for 'Country' associated with the 'Third Party Exporter Name', or labels like 'Third Party Exporter Country:'.
        * **Location:** Usually found near the 'THIRD PARTY EXPORTER NAME'. It might be part of their address or a separate field.
    2.  **What to Extract:**
        * Extract the full name of the country.

    **Examples of Third Party Exporter Country text:**
    * "Hong Kong" (if listed next to or as part of the third-party exporter's details)
    * "Country: Singapore" (in a section for third-party exporter)

    **Output Requirements:**
    * **Format:** Return the extracted country name as a **string**.
    * **If Not Found / Not Applicable:** If no third-party exporter country is specified (or no third-party exporter is named), return **null**.
    """,
        },
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
                            """,
        },
        {
            "name": "INVOICE DATE",
            "description": """Extract the specific date when the invoice or proforma invoice was created or issued by the seller/issuer.

            **Typical Location & Labels:**
            This date is usually found in the header of the document, often near the Invoice Number (or Proforma No.), or close to the seller's details.
            Look for labels such as 'Invoice Date', 'Date', 'Issue Date', 'Date of Issue', or similar terms. Prioritize the date most clearly associated with the overall document issuance.

            **Important Considerations for Extraction:**
            1.  **OCR Imperfections in Dates:** The input text (from OCR) can have errors. Numbers in dates are prone to misrecognition (e.g., '0' vs 'O', '1' vs '7', '3' vs '8', '5' vs '6'). Interpret carefully to capture the most plausible date.
            2.  **Date Formats:** Dates can appear in various formats (e.g., DD/MM/YYYY, MM/DD/YYYY, DD-MMM-YY, YYYY-MM-DD, DD.MM.YYYY). Extract the date as it appears in the document.
            3.  **Ambiguity with Multiple Dates:** If multiple dates are present (e.g., order date, shipping date, invoice date), ensure you extract the primary issuance date of this specific invoice/proforma invoice. This is typically the date listed alongside or directly under the main invoice/proforma number.
            4.  **Readability Issues:** If parts of the date text appear garbled or unclear from the OCR output, extract the legible portions accurately.

            **Output Format:**
            Extract the date as a single string, preserving its original format as seen in the document.

            **Example (from the provided document context):**
            In the current document (a proforma invoice), the relevant date is found next to the label 'DATE:' in the section containing the 'PROFORMA NO:', and its value is '10/10/2024'.

            **Task:** Locate and accurately extract this issuance date from the provided document text.
            """,
        },
        {
            "name": "INVOICE NO",
            "description": """Extract the unique alphanumeric identifier assigned to this specific invoice or proforma invoice by the seller/issuer.

            **Typical Location & Labels:**
            This identifier is critical and usually prominently displayed, often in the header of the document, near the date, or near the seller's information.
            Search for labels such as 'Invoice No.', 'Invoice #', 'Inv. No.', 'PROFORMA NO:', 'Proforma Invoice No.', 'Reference #', 'Document No.', or similar terms.

            **Important Considerations for Extraction:**
            1.  **OCR Imperfections & Alphanumeric Strings:** The input text (from OCR) can have errors. Invoice numbers are often alphanumeric and may contain characters that OCR can confuse (e.g., '0' vs 'O', '1' vs 'I' or 'l', 'S' vs '5', 'G' vs '6', 'B' vs '8'). Interpret carefully to capture the most plausible identifier.
            2.  **Special Characters & Formatting:** Invoice numbers can include special characters such as hyphens ('-'), slashes ('/'), spaces, or colons. Ensure these are captured as part of the invoice number if they appear to be integral to it.
            3.  **Completeness:** Extract the entire sequence of characters that form the invoice number. It might be a combination of letters, numbers, and symbols.
            4.  **Readability Issues:** If parts of the invoice number text appear garbled or unclear from the OCR output, extract the legible portions accurately. Note if any part seems particularly ambiguous.

            **Output Format:**
            Extract the identifier as a single string.

            **Example (from the provided document context):**
            In the current document (a proforma invoice), the identifier is found next to the label 'PROFORMA NO:' and its value is 'IN010 / 2024-25'.

            **Task:** Locate and accurately extract this unique invoice identifier from the provided document text.
            """,
        },
        {
           "name": "BUYER NAME",
            "description": """Extract the full legal name of the individual or company purchasing the goods or services, often referred to as the buyer or consignee.

            **Typical Location & Labels:**
            Look for this information under or adjacent to labels such as 'CONSIGNEE:', 'Buyer:', 'Bill To:', 'Customer:', 'Sold To:', 'Importer:', 'To:', or 'Applicant:'. It's generally the primary name listed in the recipient details block.

            **Important Considerations for Extraction:**
            1.  **OCR Imperfections:** The input text is generated by OCR and may contain errors, such as misrecognized characters (e.g., 'I' vs 'L', 'O' vs '0', 'S' vs '5') or inconsistencies in spacing. Please interpret the text carefully to identify the most plausible name.
            2.  **Ambiguous Characters/Formatting:** Company names can sometimes include special characters or varied casing. Extract the name as accurately as it appears, but be mindful that OCR might misinterpret some stylistic elements. If characters are ambiguous, choose the interpretation that forms a coherent name.
            3.  **Readability Issues:** If parts of the name text appear garbled, unclear, or potentially unreadable as provided by the OCR, extract the legible portions to the best of your ability. If a significant part is indecipherable, try to capture what is clear.
            4.  **Completeness:** Ensure you capture the full name. This might be on a single line or could be the first prominent line in the consignee/buyer address block before the street details begin.

            **Output Format:**
            Extract the name as a single string.

            **Example (from the provided document context):**
            For instance, in the current invoice, the buyer's name 'SP IMPEX' is the first line under the 'CONSIGNEE' heading.

            **Task:** Identify and extract this name precisely from the provided document text.
            """,
        },
        {
            "name": "BUYER ADDRESS",
            "description": """Extract the complete mailing address of the buyer or consignee from the provided text.
            This information is typically found under headings like 'CONSIGNEE:', 'Bill To:', 'Deliver To:', or 'Buyer:'.

            **Important Considerations for Extraction:**
            1.  **OCR Imperfections:** The input text is generated by OCR and may contain errors, such as misrecognized characters (e.g., '1' vs 'I', '0' vs 'O', 'S' vs '5', '2' vs 'Z') or inconsistencies in spacing. Please interpret the text carefully, prioritizing common address structures and plausible character sequences.
            2.  **Ambiguous Characters/Numbers:** Pay special attention to numbers within the address (street numbers, postal codes). If a number seems ambiguous or could be misread, extract the most likely interpretation based on context. For instance, if a character could be '2' or '4', choose the one that forms a more coherent address component.
            3.  **Readability Issues:** If parts of the address text appear garbled, unclear, or potentially unreadable as provided by the OCR, extract the legible portions to the best of your ability. If a significant part is indecipherable, try to capture what is clear.
            4.  **Structure:** The address often includes the company name, street details (including number and street name), city, state/province, postal code, and country.

            **Output Format:**
            Extract the full, multi-line address as a single string, preserving line breaks (e.g., using '\n' as a separator).

            **Example based on common structure (verify against document text):**
            'COMPANY NAME\nSTREET NUMBER AND NAME,\nDISTRICT/AREA, CITY\nCITY – POSTAL CODE STATE, COUNTRY'

            **Locate the text block clearly designated for the recipient of the goods or invoice and apply the above considerations during extraction.**
            For instance, in the provided document, this is under 'CONSIGNEE'. The street number initially appeared as 'NO-29', but if it were 'No.-44' and the OCR was slightly off, careful interpretation of the characters would be needed.
            """,
        },
        {
            "name": "BUYER COUNTRY",
            "description": """First, identify the full name of the country where the buyer is officially located or registered from their address details.
            This is often the last line or a prominent part of the buyer's address block (e.g., under 'CONSIGNEE:', 'Bill To:').
            Second, based on the identified full country name, provide its standard 2-letter ISO 3166-1 alpha-2 country code.

            **Process:**
            1.  **Identify Full Country Name:**
                * Scan the buyer's address section for the country name.
                * **OCR Imperfections:** Be aware that the OCR'd text for the country name might have minor errors (e.g., 'Indla' instead of 'India', 'Untted States' instead of 'United States', 'Canda' for 'Canada'). Interpret to identify the most plausible standard English country name.
                * **Address Structure:** The country is usually the most encompassing geographical part of the address, often appearing last.
                * **Readability:** If the country name is significantly garbled or unreadable from the OCR'd text, it may be difficult to determine the code accurately. Extract what is most legible.

            2.  **Convert to 2-Letter ISO Code:**
                * Once the most plausible full country name is identified, convert it to its corresponding 2-letter ISO 3166-1 alpha-2 code.
                * The model should use its general knowledge for this conversion.
                * *Examples of Mapping:*
                    * 'India' should result in 'IN'.
                    * 'United States' or 'USA' should result in 'US'.
                    * 'Germany' should result in 'DE'.
                    * 'TANZANIA' (from the seller's address in the example document) would be 'TZ'.

            **Final Output Value for this Field:**
            The value extracted should **ONLY be the 2-letter ISO country code.**

            **Example (based on the provided document context for the BUYER):**
                * In the 'CONSIGNEE' address, the country is identified as 'INDIA'.
                * The 2-letter ISO 3166-1 alpha-2 code for 'INDIA' is 'IN'.
                * Therefore, the expected output for 'BUYER_COUNTRY' is 'IN'.
            """,
        },
        {
            "name": "SELLER NAME",
            "description": """Extract the full legal name of the company selling the goods or services and issuing the invoice.
            This name is typically found at the top of the invoice, in the letterhead section, or near labels like 'Seller', 'From', 'Exporter', or 'Beneficiary'.

            **Guidelines for Identification:**
            1.  **Letterhead Priority:** The company name displayed prominently in the invoice header or letterhead is usually the seller.
            2.  **Legal Entity:** Prioritize the name that includes a legal suffix (e.g., Ltd, Inc., LLC, Pvt. Ltd) if available, as this often denotes the full legal name.
            3.  **"Unit of" or "Division of" Scenarios:**
                * If the letterhead presents a name like 'Trading Name, A Unit of Legal Entity Ltd' or 'Operating Division, Division of Parent Company Inc.', the 'Legal Entity Ltd' or 'Parent Company Inc.' is generally preferred as the full legal name.
                * However, if 'Trading Name' itself is listed as the 'Beneficiary' for payments and appears to operate as the primary invoicing entity, it may be considered. For consistency, prefer the parent legal entity if clearly stated.
            4.  **Labels:** Check for explicit labels like 'Seller:', 'Exporter:', 'Beneficiary Name:'. The name associated with these can confirm or be the seller name. In the provided document, 'GOLDEN CASHEWS' is the beneficiary, and "Golden Ventures Ltd" is its parent company. The seller is "Exporters of Raw Cashew Nuts". The name "GOLDEN CASHEWS" is most prominent.

            **Important Considerations for Extraction:**
            * **OCR Imperfections:** Text from letterheads or logos can sometimes be stylized. Interpret carefully, watching for misrecognized characters.
            * **Completeness:** Ensure the full name, including any legal suffixes (Ltd., Inc., etc.), is captured if it's part of the identified legal name.

            **Output Format:**
            Extract the name as a single string.

            **Example Interpretation (based on the provided document):**
            The header shows 'GOLDEN CASHEWS' prominently, with 'A Unit of Golden Ventures Ltd' underneath. 'GOLDEN CASHEWS' is also the 'BENEFICIARY NAME'.
            * If strictly seeking the parent legal entity: 'Golden Ventures Ltd'.
            * If seeking the primary operating name as displayed and used for payment: 'GOLDEN CASHEWS'.
            For this field, given "full legal name," prioritize the legal entity if distinct: **'Golden Ventures Ltd'**. If the trading name is the only one with clear legal standing (e.g. has "Ltd" itself, or is the only one identified), then use that.

            **Task:** Identify and extract the seller's full legal name based on these guidelines. For the provided document, this would be 'Golden Ventures Ltd'.
            """,
        },
        {
            "name": "SELLER ADDRESS",
            "description": """Extract the complete mailing address of the seller/issuer. This should include all relevant components such as street information, P.O. Box, city, state/province, postal code, and country.

            **Typical Location & Labels:**
            The seller's address is usually found near the 'SELLER NAME', often in the header or footer of the invoice. It might also be under labels like 'From:', 'Remit To:', or simply be part of the main contact block for the issuing company.

            **Important Considerations for Extraction:**
            1.  **OCR Imperfections:** Address text, especially if dense, in small font, or a long single line (as in the example document), can be prone to OCR errors or misinterpretations (e.g., '1' vs 'I', misread characters in place names). Extract the most plausible address string based on the visual information. Be aware that unfamiliar place names or abbreviations might be present.
            2.  **Formatting & Completeness:**
                * Capture all parts of the address.
                * If the address is visually presented over multiple distinct lines in the document, preserve these line breaks in the extracted string (e.g., using '\\n' as a separator).
                * If the address is presented as a single continuous line of text (like in the provided invoice example), extract it as such.
            3.  **Multiple Addresses:** If multiple seller addresses are present (e.g., corporate HQ vs. remit-to address), prefer the address most clearly associated with the invoice issuance or the seller's primary identity on the main invoice pages.

            **Output Format:**
            Extract the full address as a single string. If the original address spans multiple lines, use '\\n' to denote line breaks.

            **Example (from the provided document context):**
            The seller's address is in the header and appears as a single long line: 'PO BOX 1752 PLOT NO 125 BLOCK M NOOKPO RD MCHIGEOLQ IND AREA SONGEA LOWER SONGEA AVENUE KIBEGE STREET DAR ES SALAAM, TANZANIA'. This should be extracted as a single string. If it were formatted on the invoice over several lines, those breaks would be preserved with '\\n'.

            **Task:** Locate and accurately extract the seller's complete mailing address.
            """,
        },
        {
            "name": "SELLER COUNTRY",
            "description": """Extract the country where the seller is officially located or registered.

            **Typical Location & Labels:**
            The country is typically the last component of the seller's full address or may be explicitly labeled (e.g., 'Country: TANZANIA'). It's often found in the invoice header near the seller's name and address.

            **Important Considerations for Extraction:**
            1.  **OCR Imperfections:** The country name, especially if part of a long address line, might be subject to OCR errors. Interpret common country names even if slightly misspelled (e.g., 'Tanzana' should be recognized as 'TANZANIA').
            2.  **Address Structure:** Focus on the last distinct geographical entity in the seller's address block.
            3.  **Inference (USA Specific):** For US addresses, the country "USA" might be inferred from state abbreviations (e.g., 'IN' for Indiana, 'IL' for Illinois implies USA). For other countries, the name is usually explicit.
            4.  **Readability Issues:** If the country name is significantly garbled, use contextual clues from the city or other address parts if possible.

            **Output Format:**
            Extract the country name as a single string (e.g., 'TANZANIA', 'INDIA', 'USA').

            **Example (from the provided document context):**
            The seller's address in the header concludes with 'TANZANIA'. Therefore, the seller country is 'TANZANIA'.

            **Task:** Locate and accurately extract the seller's country from their address.
            """,
        },
        {
            "name": "INVOICE CURRENCY", 
            "description": """Identify and extract the specific currency in which the invoice amounts are denominated.

            **Typical Location & Labels:**
            Currency information is often found next to monetary values (especially totals or line item amounts), in column headers of financial tables (e.g., 'Amount USD', 'Price EUR'), or explicitly stated (e.g., 'All amounts in USD'). Look for currency symbols (e.g., $, €, £) or standard currency codes (e.g., USD, EUR, GBP, INR).

            **Important Considerations for Extraction:**
            1.  **Explicit Codes vs. Symbols:** Prioritize explicit currency codes (like 'USD', 'EUR') if present. If only symbols are found (like '$'), infer the most likely currency based on context (e.g., '$' on an invoice from a US seller often implies USD, but could be CAD, AUD, etc., so explicit codes are better).
            2.  **Consistency:** Check if the currency is consistently used across multiple monetary fields (line items, totals, payment terms).
            3.  **OCR Imperfections:** Currency codes or symbols might be misread. 'USO' might be 'USD', 'EUR0' might be 'EUR'.
            4.  **Absence:** If no currency symbol or code is clearly associated with the main financial amounts, this field might be indeterminable from the visual data alone.

            **Output Format:**
            Extract the 3-letter ISO 4217 currency code as a string (e.g., 'USD', 'EUR', 'INR').

            **Example (from the provided document context):**
            The table headers for monetary values are 'Rate USD' and 'Amount USD'. Payment terms also explicitly mention 'USD'. Thus, the currency is 'USD'.

            **Task:** Determine and extract the primary currency used for the invoice transactions.
            """,
        },
        {
            "name": "INVOICE AMOUNT/VALUE",
            "description": """Extract the primary financial value of the invoice, typically the total sum of goods/services listed. This could be a subtotal, a net amount before final charges/taxes, or the grand total if no other total is more prominent or if it's the main sum being invoiced.

            **Typical Location & Labels:**
            Look for amounts associated with labels like 'Total', 'Subtotal', 'Net Amount', 'Invoice Total', 'Amount Due before Tax'. It's crucial to distinguish this from individual line item amounts if an overall total for the goods/services is present. This value should be numerical.

            **Important Considerations for Extraction:**
            1.  **Clarity of 'Total':** Identify the most significant sum representing the value of the invoiced items/services. On some invoices, multiple totals exist (e.g., Subtotal, Tax, Grand Total). This field aims for the main sum of the goods/services themselves, which might be a subtotal before other charges, or the grand total if the structure is simple.
            2.  **Numerical Value Only:** Extract only the numerical value. Do not include currency symbols or codes in this specific field output. Ensure correct parsing of thousands separators (commas) and decimal points if present (e.g., '212,800.00' should be extracted as '212800.00' or '212800').
            3.  **OCR Imperfections:** Numbers are prone to OCR errors (e.g., '1' vs '7', '0' vs '8'). Validate against calculations if possible (e.g., quantity * rate).
            4.  **Distinction from other totals:** If 'Grand Total' or 'Total Amount Due' is very distinct and appears to be a final calculation after this sum, this field should capture the sum *before* those final adjustments if it's clearly presented. In simpler invoices, this might be the only total.

            **Output Format:**
            Extract the numerical value as a string, ideally cleaned of currency symbols and non-essential formatting (e.g., '212800', '1500.75').

            **Example (from the provided document context):**
            The 'Description of Goods' table shows a 'TOTAL AMOUNT' of '212800'. This represents the sum of the invoiced goods.

            **Task:** Identify and extract the primary total sum for the goods/services listed on the invoice. For this document, it is '212800'.
            """,
        },
        {
            "name": "INVOICE AMOUNT/VALUE IN WORDS",
            "description": """Extract the total invoice amount written out in words (e.g., 'One Hundred Thirty-Five Thousand Seven Hundred Fifty Dollars Only').

            **Typical Location & Labels:**
            This field is often found near the numerical total amount. It might be labeled 'Amount in Words', 'Say Total', 'In Words', or simply appear as a textual representation of the sum without a specific label.

            **Important Considerations for Extraction:**
            1.  **Presence:** This field is not always present on invoices. If no amount in words is found, this should be indicated clearly (e.g., by outputting 'null' or an empty string).
            2.  **OCR Imperfections:** Textual representations of numbers can be long and prone to OCR errors. Extract the most plausible text.
            3.  **Exact Wording:** Capture the full text as written, including any suffixes like 'Only' or currency mentions if they are part of the worded amount.

            **Output Format:**
            Extract the amount in words as a single string. If not found, return 'null'.

            **Example (from the provided document context):**
            This specific proforma invoice does not appear to have the total amount written out in words. In such a case, the output should be 'null'.

            **Task:** Locate and extract the total invoice amount written in words. If it is not present, indicate 'null'.
            """,
        },
        {
           "name": "BENEFICIARY ACCOUNT NO / IBAN",
            "description": """Extract the International Bank Account Number (IBAN) of the seller (beneficiary) where the payment should be sent. This field should ONLY capture IBANs.

            **Typical Location & Labels:**
            This information is usually found in a dedicated 'Bank Details', 'Payment Instructions', or 'Beneficiary Bank Account Details' section.
            Look specifically for the label 'IBAN'.

            **Important Considerations for Extraction:**
            1.  **IBAN Identification:** An IBAN typically starts with a two-letter country code (e.g., 'IT' for Italy[cite: 5], 'DE' for Germany, 'GB' for Great Britain) followed by check digits and the basic bank account number. Ensure the extracted value matches this structure and is labeled as IBAN.
            2.  **Specificity:** Only extract the value if it is explicitly identified as an IBAN. Do not extract other account numbers for this field.
            3.  **Accuracy:** If an IBAN is found, ensure precise extraction, including all alphanumeric characters.
            4.  **OCR Imperfections:** IBANs are alphanumeric and can be misread by OCR (e.g., '0' vs 'O', 'I' vs 'L', '5' vs 'S'). Interpret carefully.
            5.  **Absence:** If no IBAN is explicitly stated or identifiable, this field should be 'null' or empty.

            **Output Format:**
            Extract the IBAN as a single string. If no IBAN is found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The document's 'Banca d'Appoggio / Bank' section explicitly lists 'IBAN:IT2710503438100000000127869'[cite: 5]. This value should be extracted.

            **Task:** Locate and accurately extract the beneficiary's IBAN. If no IBAN is present, output 'null'.
            """,
        },
        {
            "name": "BENEFICIARY BANK",
            "description": """Extract the name of the bank where the seller (beneficiary) holds their account for payment.

            **Typical Location & Labels:**
            This information is usually found in a 'Bank Details', 'Payment Instructions', or 'Beneficiary Bank Account Details' section, often near the account number or IBAN.
            Look for labels like 'Bank Name', 'Beneficiary Bank', 'Bank', 'Payable to Bank', 'Banca d'Appoggio'.

            **Important Considerations for Extraction:**
            1.  **Clarity:** Extract the official name of the bank.
            2.  **OCR Imperfections:** Bank names can be misread, especially if they are part of a dense text block or stylized.
            3.  **Association:** Ensure the bank name extracted is clearly associated with the beneficiary's account details for receiving payment for *this* invoice.
            4.  **Absence:** If the bank name is not explicitly provided in the relevant sections, this field should be 'null' or empty.

            **Output Format:**
            Extract the bank name as a single string. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` shows an IBAN under 'Banca d'Appoggio / Bank'[cite: 5], but the actual name of the bank is not explicitly stated in this section or clearly nearby in the provided text. Therefore, for this document, based on the provided information, the output would be 'null'. If a name like 'XYZ Bank Italia' were present, that would be extracted.

            **Task:** Locate and accurately extract the beneficiary's bank name. If not found, indicate 'null'.
            """,
        },
        {
            "name": "BENEFICIARY BANK ADDRESS",
            "description": """Extract the full mailing address of the seller's (beneficiary's) bank.

            **Typical Location & Labels:**
            This information is usually found within the 'Bank Details', 'Payment Instructions', or 'Beneficiary Bank Account Details' section, typically co-located with the beneficiary bank's name and account number/IBAN.
            Look for a complete address (street, city, country) associated directly with the beneficiary's bank.

            **Important Considerations for Extraction:**
            1.  **Association with Bank:** Ensure the address extracted is clearly for the beneficiary's bank and not the seller's own business address, unless they are explicitly stated to be the same (which is rare for a bank address).
            2.  **Completeness:** If found, the address should ideally include street information, city, postal/zip code, and country.
            3.  **OCR Imperfections:** Address blocks can be dense and prone to OCR errors. Interpret carefully.
            4.  **Absence:** If the beneficiary bank's address is not provided in the relevant sections, this field should be 'null' or empty. The absence of the bank's name often implies the absence of its address as well.

            **Output Format:**
            Extract the full address as a single string. If the original address spans multiple lines, use '\\n' to denote line breaks. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` lists an IBAN under 'Banca d'Appoggio / Bank' but does not explicitly state the beneficiary bank's name or its address in that section or elsewhere in the provided text. Therefore, for this document, the output would be 'null'. If a bank address like 'Via Roma 1, 50100 Florence, Italy' were present for the beneficiary bank, that would be extracted.

            **Task:** Locate and accurately extract the beneficiary bank's full mailing address. If not found, indicate 'null'.
            """,
        },
        {
            "name": "BENEFICIARY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE / ROUTING NO", # Spelling from user, expanded
            "description": """Extract the unique identification code for the seller's (beneficiary's) bank. This could be a SWIFT/BIC code (for international payments), ABA Routing Number (for US payments), Sort Code (UK), BSB (Australia), IFSC (India), or other relevant national bank clearing codes.

            **Typical Location & Labels:**
            This code is usually found in the 'Bank Details', 'Payment Instructions', or 'Beneficiary Bank Account Details' section, often alongside the bank name and account number/IBAN.
            Look for labels like 'SWIFT Code', 'BIC' (Bank Identifier Code), 'ABA No.', 'Routing No.', 'IFSC', 'Sort Code', 'BSB', or similar.

            **Important Considerations for Extraction:**
            1.  **Type of Code:** Identify the type of code if labeled (e.g., SWIFT, ABA). If unlabeled, extract the alphanumeric code found in the expected location.
            2.  **Accuracy:** These codes are critical for payment routing; ensure precise extraction.
            3.  **OCR Imperfections:** Alphanumeric codes are prone to OCR errors (e.g., '0' vs 'O', '1' vs 'I', '5' vs 'S', 'B' vs '8').
            4.  **Prioritization:** For international invoices, a SWIFT/BIC code is most common. For domestic payments, national codes like ABA, IFSC, etc., are used. If multiple codes are present, the prompt might need to specify which to prioritize, or this specific field might target one type (e.g., a separate field for SWIFT and another for ABA). For this general field, extract any that is clearly a bank identifier code.
            5.  **Absence:** If no such bank identification code is found, this field should be 'null' or empty.

            **Output Format:**
            Extract the code as a single string. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` provides an IBAN for the beneficiary bank [cite: 5] but does not explicitly list a SWIFT/BIC code or any other bank routing code in the 'Banca d'Appoggio / Bank' section or elsewhere in the provided text. Therefore, for this document, the output would be 'null'. If a code like 'PASCITM1FLO' were present, that would be extracted.

            **Task:** Locate and accurately extract the beneficiary bank's SWIFT/BIC or other relevant routing code. If not found, indicate 'null'.
            """,
        },
        {
            "name": "Total Invoice Amount",
            "description": """Extract the final, definitive total monetary sum due on the invoice. This amount should be inclusive of all items, charges, and taxes (if applicable and included in the final sum), and less any deductions reflected directly in this final total.

            **Typical Location & Labels:**
            This value is often the most prominent total on the invoice. Look for labels such as 'Grand Total', 'Total Amount Due', 'Total Invoice Amount', 'Total Invoice Value', 'Please Pay This Amount', 'Net Total', 'Totale Fattura', 'Totale da pagare'.

            **Important Considerations for Extraction:**
            1.  **Definitive Total:** Ensure this is the ultimate figure the buyer is expected to pay. If multiple totals are present (e.g., Subtotal, Total with Tax), this should be the final one.
            2.  **Numerical Value Only:** Extract only the numerical value. Do not include currency symbols or codes in this field's output (currency is typically a separate field). Ensure correct parsing of thousands separators (e.g., periods in European formats, commas in US formats) and decimal points/commas.
            3.  **OCR Imperfections:** Numbers are susceptible to OCR errors (e.g., '8' vs '3', '5' vs '6'). Cross-verify with other totals or sums if possible.
            4.  **Clarity and Prominence:** This amount is usually clearly set apart and emphasized.

            **Output Format:**
            Extract the numerical value as a string, representing the exact monetary amount (e.g., '82.590,00', '15075.50').

            **Example (from the `INVOICE.pdf` context):**
            The document explicitly states 'Totale Fattura / Total Invoice Amount' as '82.590,00 EUR'[cite: 11]. The value '82.590,00' should be extracted for this field.

            **Task:** Locate and accurately extract the final total amount due on the invoice.
            """,
        },
        {
            "name": "Invoice Amount", # Repeated field, ensure description helps differentiate or confirms synonymity
            "description": """Extract the primary sum of the invoice. This often refers to the main total amount and can be synonymous with 'TOTAL_INVOICE_AMOUNT' if only one definitive total is presented. If multiple totals exist (e.g., Subtotal, Total before Tax, Grand Total), this should ideally capture the most representative invoiced amount, frequently the grand total.

            **Typical Location & Labels:**
            This can be found near labels like 'Invoice Amount', 'Total', 'Net Amount', or it might be the same figure as 'Grand Total' or 'Total Amount Due'.

            **Important Considerations for Extraction:**
            1.  **Synonymity with Total:** In many invoices, like the example document, this will be identical to the 'TOTAL_INVOICE_AMOUNT'. The purpose is to capture the main financial figure of the invoice.
            2.  **Numerical Value Only:** Extract only the numerical value, excluding currency symbols or codes. Handle decimal and thousands separators appropriately.
            3.  **OCR Imperfections:** Be cautious of OCR errors in numerical figures.
            4.  **Contextual Understanding:** If the invoice structure is complex with multiple totals, identify which figure best represents the 'Invoice Amount' before specific deductions or charges if it's meant to be different from a final 'Grand Total'. For most straightforward invoices, it will be the main or grand total.

            **Output Format:**
            Extract the numerical value as a string (e.g., '82.590,00', '15075.50').

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` shows 'Totale Fattura / Total Invoice Amount' as '82.590,00 EUR'[cite: 11]. There is no other distinct 'Invoice Amount' that differs from this total. Thus, for this document, the 'INVOICE_AMOUNT' is also '82.590,00'.

            **Task:** Identify and extract the primary invoice amount.
            """,
        },
        {
            "name": "Beneficiary Name", # Often the same as Seller Name
            "description": """Extract the name of the ultimate recipient of the funds for this invoice, who is typically the seller or exporter.

            **Typical Location & Labels:**
            Look for labels such as 'Beneficiary', 'Beneficiary Name', 'Payable to', 'Pay To'. This name is often found in the 'Bank Details' or 'Payment Instructions' section.
            If not explicitly labeled as 'Beneficiary', this is almost always the same as the 'SELLER_NAME'.

            **Important Considerations for Extraction:**
            1.  **Primary Identification:** The goal is to identify the party to whom the payment is owed.
            2.  **Seller as Beneficiary:** If no separate beneficiary name is listed in the payment section, assume the seller is the beneficiary and use the extracted 'SELLER_NAME'.
            3.  **Consistency:** Check if the name in the letterhead/seller identification matches any name given in the payment details.
            4.  **OCR Imperfections:** Names can be subject to OCR errors; extract the most plausible and complete name.
            5.  **Completeness:** Include any legal suffixes (e.g., Srl, Ltd., Inc.) if they are part of the name.

            **Output Format:**
            Extract the name as a single string.

            **Example (from the `INVOICE.pdf` context):**
            The seller is identified as 'La Marzocco Srl'[cite: 1, 13]. The bank details section [cite: 5] does not specify a different beneficiary name. Therefore, the 'BENEFICIARY_NAME' is 'La Marzocco Srl'.

            **Task:** Identify and extract the beneficiary's name.
            """,
        },
        {
            "name": "Beneficiary Address", 
            "description": """Extract the full mailing address of the beneficiary (typically the seller/exporter) to whom the payment is directed.

            **Typical Location & Labels:**
            This address is commonly the same as the 'SELLER_ADDRESS'. Look for it in the seller's contact information section or header. If specific 'Beneficiary Address' details are provided separately in payment instructions, those should be prioritized.

            **Important Considerations for Extraction:**
            1.  **Seller's Address as Default:** If no distinct beneficiary address is given in payment instructions, use the seller's primary business address.
            2.  **Multiple Seller Addresses:** If the seller has multiple addresses listed (e.g., registered office, operational address), use the one most relevant for correspondence or invoicing, often the operational or main address, unless payment instructions specify otherwise. This should be consistent with what is extracted for 'SELLER_ADDRESS' if the beneficiary is the seller.
            3.  **Completeness:** Ensure the full address is captured, including street, city, postal code, state/province, and country.
            4.  **OCR Imperfections:** Address text can be dense and prone to errors.
            5.  **Formatting:** Preserve multi-line formatting using '\\n' if the address is presented over multiple lines.

            **Output Format:**
            Extract the full address as a single string, using '\\n' for line breaks if applicable.

            **Example (from the `INVOICE.pdf` context):**
            The beneficiary is 'La Marzocco Srl'. The document lists their 'Sede Operativa ed Amministrativa' as 'Via La Torre 14/Н 50038 Scarperia e San Piero (FI) - Italia' [cite: 1] and 'Sede Legale' as 'Viale G. Matteotti, 25 50121 FIRENZE (FI)'[cite: 13]. Assuming the operational address is the primary one for such purposes unless specified otherwise, this would be 'Via La Torre 14/Н\\n50038 Scarperia e San Piero (FI) - Italia'.

            **Task:** Identify and extract the beneficiary's full mailing address.
            """,
        },
        {
            "name": "DESCRIPTION OF GOODS",
            "description": """Extract a detailed account of all products or services being invoiced. This is usually found in the main table or line items section of the invoice and can include product names, codes, specifications, or service descriptions.

            **Typical Location & Labels:**
            Look for columns labeled 'Description', 'Description of Goods', 'Item Description', 'Details', or similar within the line items table.

            **Important Considerations for Extraction:**
            1.  **Multiple Items:** If there are multiple line items, extract the description for each.
            2.  **Multi-line Descriptions:** Individual item descriptions may span multiple lines. Capture all relevant descriptive text, preserving internal line breaks if they add clarity.
            3.  **Concatenation/Formatting:** Combine descriptions from all line items into a single string. Use a clear separator (e.g., a double newline '\\n\\n' or a specific marker like 'ITEM_SEPARATOR') between the descriptions of distinct items or POs if not itemizing.
            4.  **Completeness vs. Summary:** Extract all relevant descriptive text for each line item. Summarize only if the list is exceptionally long and a summary is explicitly allowed. For most cases, full descriptions are preferred.
            5.  **OCR Imperfections:** Descriptions can contain alphanumeric codes, special characters, and mixed case text, which can be prone to OCR errors. Interpret carefully.

            **Output Format:**
            Extract as a single string. For multiple items, concatenate their descriptions, preserving internal newlines and using a consistent separator (e.g., '\\n\\n') between descriptions of different items.

            **Example (from the `INVOICE.pdf` context):**
            The invoice lists descriptions for three POs/items. The extracted string should combine these, for instance:
            'LINEA PB 2GR AV ABR HW 220V CE [cite: 9]\\nLA MARZOCCO ESPRESSO COFFEE MACHINE [cite: 9]\\nMACHINE FULLY EQUIPPED [cite: 9]\\nColour-ACCIAIO LUCIDO [cite: 9]\\nOptional kit - High Legs [cite: 9]\\n\\nLINEA PB 2GR AV ABR HW 220V CE [cite: 9]\\nLA MARZOCCO ESPRESSO COFFEE MACHINE [cite: 9]\\nMACHINE FULLY EQUIPPED [cite: 9]\\nColour-ACCIAIO LUCIDO [cite: 9]\\nOptional kit - High Legs [cite: 9]\\n\\nCUSTOMS DOCS FEE [cite: 9]'

            **Task:** Locate and accurately extract all descriptions of goods or services listed on the invoice.
            """,
        },
        {
            "name": "QUANTITY OF GOODS",
            "description": """Extract the amount or number of units for each item or service listed on the invoice, including the unit of measure if specified.

            **Typical Location & Labels:**
            Look for columns labeled 'Quantity', 'Qty', 'Units', 'No. of Items', 'U.M.' (Unit of Measure), 'Unit' within the line items table.

            **Important Considerations for Extraction:**
            1.  **Multiple Items:** If there are multiple line items, extract the quantity for each.
            2.  **Units of Measure:** If a unit of measure (e.g., PC, KG, EA, HRS, M, LBS) is specified alongside the quantity, include it.
            3.  **Concatenation/Formatting:** For multiple items, list each quantity and its unit. Combine these into a single string using a clear separator (e.g., a semicolon '; ' or newline '\\n').
            4.  **Numerical Accuracy:** Ensure quantities are extracted accurately as numbers.
            5.  **OCR Imperfections:** Numbers and unit abbreviations can be misread.

            **Output Format:**
            Extract as a single string. For multiple items, list each quantity with its unit, separated by a consistent delimiter (e.g., '; ' or '\\n'). Example: '2 PC; 10 PC; 1 PC'.

            **Example (from the `INVOICE.pdf` context):**
            The invoice lists quantities and units (U.M.) for three items:
            - Item 1: Q.ty '2', U.M. 'PC' [cite: 9]
            - Item 2: Q.ty '10', U.M. 'PC' [cite: 9]
            - Item 3: Q.ty '1', U.M. 'PC' [cite: 9]
            The extracted string could be '2 PC; 10 PC; 1 PC'.

            **Task:** Locate and accurately extract the quantities for all goods or services, including their units of measure if available.
            """,
        },
        {
            "name": "PAYMENT TERMS",
            "description": """Extract the conditions agreed upon for payment of the invoice, such as the timeframe, percentage due, and method specifics if included in the terms.

            **Typical Location & Labels:**
            Search for labels like 'Payment Terms', 'Terms of Payment', 'Terms', 'Condizioni pagamento'. This information is often found in the header, footer, or a dedicated section of the invoice.

            **Important Considerations for Extraction:**
            1.  **Completeness:** Capture the full text of the payment terms as stated. This can include percentages, due dates relative to an event (e.g., 'Net 30 days', 'Due Upon Receipt'), or specific conditions like 'Letter of Credit'.
            2.  **Clarity:** Ensure the extracted text accurately reflects the conditions.
            3.  **OCR Imperfections:** Terms can sometimes be in smaller print or complex phrasing, so careful OCR interpretation is needed.
            4.  **Distinction from Method:** While related, payment terms (e.g., 'Net 30') are distinct from the payment method (e.g., 'Bank Transfer'). This field is for the terms themselves.

            **Output Format:**
            Extract the payment terms as a single string.

            **Example (from the `INVOICE.pdf` context):**
            The document states 'Condizioni pagamento / Payment Terms' as '100% advanced'. This is the value to be extracted.

            **Task:** Locate and accurately extract the payment terms stated on the invoice.
            """,
        },
        {
            "name": "BENEFICIARY/SELLER'S SIGNATURE",
            "description": """Identify the handwritten or digital signature of the authorized representative of the seller/beneficiary. This can also be a typed name if it clearly serves as an authorization in place of a physical signature.

            **Typical Location & Labels:**
            Look for a signature line or block, often at the bottom of the invoice, labeled 'Seller's Signature', 'Authorized Signature', 'For [Seller Company Name]', or similar.

            **Important Considerations for Extraction:**
            1.  **Nature of Signature:** Determine if it's a handwritten signature (often an image, may be described as 'Signature present'), a digital signature mark, or a typed name of an individual authorized to sign. A typed company name in a footer is generally not considered a signature for this purpose unless explicitly stated as 'Authorized by [Company Name]'.
            2.  **Authorization:** The key is whether it represents authorization from the seller.
            3.  **Presence:** This field may not always be present or legible. If a signature image is present but unreadable, "Signature present" might be appropriate. If a typed name (of an individual) acts as authorization, extract that name.
            4.  **Absence:** If no clear signature or authorizing typed individual name is found, this field should be 'null'.

            **Output Format:**
            Describe the signature if present (e.g., "Handwritten signature present", "Typed name: John Doe, Manager") or return 'null' if absent.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` does not show a clear handwritten signature or a typed name of an individual acting as an authorized signatory for La Marzocco Srl. While the company name "La Marzocco Srl" appears at the bottom, it serves more as company identification in the footer rather than a specific authorization signature by an individual. Therefore, the output would be 'null'. If, for example, 'Giovanni Rossi, Sales Director' was typed in a signature area, that would be extracted.

            **Task:** Identify and describe the beneficiary/seller's signature or authorizing typed name. If absent, indicate 'null'.
            """,
        },
        {
            "name": "APPLICANT/BUYER'S SIGNATURE",
            "description": """Identify the handwritten or digital signature of the authorized representative of the applicant/buyer, or a typed name if it clearly serves as an authorization or acceptance.

            **Typical Location & Labels:**
            Look for a signature line or block, often at the bottom of the document, potentially labeled 'Buyer's Signature', 'Authorized Signature', 'Accepted By', 'For [Buyer Company Name]', or similar. This is less common on invoices than on Purchase Orders or contracts.

            **Important Considerations for Extraction:**
            1.  **Nature of Signature:** Determine if it's a handwritten signature, a digital signature mark, or a typed name of an individual clearly indicating acceptance or authorization on behalf of the buyer. A typed company name alone is generally not a signature unless part of an explicit acceptance statement.
            2.  **Context of Acceptance:** The signature should ideally signify the buyer's acknowledgment or acceptance of the invoice terms or order.
            3.  **Presence:** This field is often not present on standard invoices. If not found, it should be 'null'. If a signature image is present but unreadable, "Signature present" might be appropriate. If a typed name (of an individual) acts as authorization, extract that name.
            4.  **Distinction:** Be careful not to confuse this with the seller's signature or other incidental text.

            **Output Format:**
            Describe the signature if present (e.g., "Handwritten signature present", "Typed name: Jane Smith, Procurement Manager") or return 'null' if absent.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` does not show a clear handwritten signature or a typed name of an individual from 'THREE ONE COFFEE ENGINEERING PVT LTD.' acting as an authorized signatory in a designated acceptance block. The text 'THREE ONE COFFEE ENGINEERING PVT. LTD. Director' appears but its context as a signature for the invoice is unclear. Therefore, the output would likely be 'null'.

            **Task:** Identify and describe the applicant/buyer's signature or authorizing typed name. If absent, indicate 'null'.
            """,
        },
        {
            "name": "MODE OF REMITTANCE",
            "description": """Extract the method by which the payment for the invoice is expected to be made.

            **Typical Location & Labels:**
            This information is often found within 'Payment Instructions', 'Bank Details', 'Payment Terms' sections, or under labels like 'Payment Method', 'Mode of Payment', 'Pay Method', 'Mod, pagamento'.

            **Important Considerations for Extraction:**
            1.  **Clarity:** Extract the specific method mentioned (e.g., Wire Transfer, ACH, Bank Transfer, Cheque, Credit Card).
            2.  **Multiple Methods:** If multiple methods are listed as acceptable, list all of them or the primary one if indicated.
            3.  **OCR Imperfections:** Ensure accurate extraction of the payment method text.

            **Output Format:**
            Extract the mode(s) of remittance as a string. If multiple, separate with a semicolon or list them.

            **Example (from the `INVOICE.pdf` context):**
            The document states 'Mod, pagamento/Pay Method' as 'BANK TRANSFER'. This is the value to be extracted.

            **Task:** Locate and accurately extract the mode of remittance.
            """,
        },
        {
            "name": "MODE OF TRANSIT",
            "description": """Extract the method of transportation used or to be used for shipping the goods (e.g., Sea, Air, Road, Rail, Ocean).

            **Typical Location & Labels:**
            Look for labels such as 'Ship Via', 'Mode of Shipment', 'Transport Mode', 'Method of Dispatch', 'Carrier', 'By'. This information might be near shipping details, Incoterms, or port information.

            **Important Considerations for Extraction:**
            1.  **Explicit Statement:** Prioritize explicitly stated modes of transit.
            2.  **Inference:** Sometimes it might be inferred from other details (e.g., 'Port of Loading' might imply 'Sea'), but explicit statements are preferred. Incoterms like CIF or FOB might imply a sea shipment but don't solely define the mode on the invoice itself.
            3.  **Absence:** This information is not always present on all types of invoices, especially proforma invoices issued before shipping arrangements are finalized. If not found, indicate 'null'.

            **Output Format:**
            Extract the mode of transit as a string. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` does not explicitly state the 'Mode of Transit'. While it mentions 'Ex Works' Incoterms, the actual mode (Sea, Air, Road) is not specified. Therefore, the output would be 'null'.

            **Task:** Locate and accurately extract the mode of transit. If not found, indicate 'null'.
            """,
        },
        {
            "name": "INCO TERM",
            "description": """Extract the Incoterm (International Commercial Term) which specifies the responsibilities of buyers and sellers in the transaction.

            **Typical Location & Labels:**
            Look for labels like 'Incoterms', 'Terms of Sale', 'Freight Terms', 'Delivery Terms', 'Resa merce'. The Incoterm is usually a three-letter code (e.g., EXW, FOB, CIF, DDP) often followed by a named place or port.

            **Important Considerations for Extraction:**
            1.  **Full Term:** Extract the full Incoterm as stated, including the code and any associated named place (e.g., 'EXW Scarperia', 'FOB Shanghai Port').
            2.  **Accuracy:** Ensure the three-letter code and the place name are captured correctly.
            3.  **Standard Terms:** Be familiar with common Incoterms.

            **Output Format:**
            Extract the Incoterm and any associated location as a single string.

            **Example (from the `INVOICE.pdf` context):**
            The document states 'Resa merce/Incoterms' as 'Ex Works'. This is the value to be extracted. If a location was specified, like 'Ex Works Scarperia', that full string would be extracted.

            **Task:** Locate and accurately extract the Incoterm.
            """,
        },
        {
            "name": "HS CODE",
            "description": """Extract the Harmonized System (HS) code or Harmonized Tariff Schedule (HTS) code for the products listed on the invoice.

            **Typical Location & Labels:**
            HS codes are usually found within the line item details in the main table, associated with each product description or item number. Look for labels like 'HS Code', 'HTS Code', 'Tariff Code', 'Customs Code'.

            **Important Considerations for Extraction:**
            1.  **Format:** HS codes are typically sequences of numbers (usually 6 digits for the international standard, but can be longer for national subdivisions).
            2.  **Association:** Ensure the code is clearly linked to a product being invoiced.
            3.  **Presence:** HS Codes are more common on commercial invoices used for customs clearance rather than all proforma invoices. If not found, indicate 'null'.
            4.  **Multiple Items:** If different items have different HS codes, list all of them, clearly associating them with their items if possible, or list them separated by semicolons or newlines.

            **Output Format:**
            Extract the HS code(s) as a string. If multiple, separate them clearly. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` does not appear to list any HS Codes for the items. Therefore, the output would be 'null'. If a code like '8419.81' was listed for an espresso machine, that would be extracted.

            **Task:** Locate and accurately extract the HS Code(s). If not found, indicate 'null'.
            """,
        },
        {
            "name": "Intermediary Bank (Field 56)",
            "description": """Extract details of any intermediary bank (also known as a correspondent bank) if specified in the payment instructions. This bank is used to route payment from the buyer's bank to the seller's (beneficiary's) bank. Information might be referred to by 'Field 56' in SWIFT message contexts.

            **Typical Location & Labels:**
            Look for sections specifically labeled 'Intermediary Bank', 'Correspondent Bank', or payment routing instructions that detail an additional bank in the payment chain.

            **Important Considerations for Extraction:**
            1.  **Explicit Mention:** Only extract information if an intermediary bank is explicitly mentioned.
            2.  **Information to Capture:** This could include the intermediary bank's name, address, SWIFT/BIC code, or account number held with them for the beneficiary bank. The exact details to capture would depend on what's provided. This prompt aims for a general capture of any details provided under such a heading.
            3.  **Absence:** This information is not always required or provided. If no intermediary bank details are found, this field should be 'null'.

            **Output Format:**
            Extract all provided details for the intermediary bank as a single string (e.g., "Bank Name, SWIFT Code, City, Country"). If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` does not mention any Intermediary Bank or Correspondent Bank details in its payment instructions. Therefore, the output would be 'null'.

            **Task:** Locate and extract any specified intermediary bank details. If absent, indicate 'null'.
            """,
        },
        {
            "name": "INTERMEDIARY BANK NAME",
            "description": """Extract the name of the intermediary bank, if one is specified in the payment instructions.

            **Typical Location & Labels:**
            This would be found under a section labeled 'Intermediary Bank', 'Correspondent Bank', or similar.

            **Important Considerations for Extraction:**
            1.  **Specificity:** Ensure the extracted name is explicitly for an intermediary or correspondent bank, not the beneficiary bank itself.
            2.  **Absence:** If no intermediary bank is named, this field should be 'null'.

            **Output Format:**
            Extract the intermediary bank's name as a single string. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` does not name an Intermediary Bank. Therefore, the output would be 'null'.

            **Task:** Locate and accurately extract the intermediary bank's name. If not found, indicate 'null'.
            """,
        },
        {
            "name": "INTERMEDIARY BANK ADDRESS",
            "description": """Extract the full mailing address of the intermediary bank, if one is specified.

            **Typical Location & Labels:**
            This information would be found along with the intermediary bank's name, under a section like 'Intermediary Bank' or 'Correspondent Bank'.

            **Important Considerations for Extraction:**
            1.  **Association:** Ensure the address is for the intermediary bank.
            2.  **Completeness:** If found, include street, city, country, etc.
            3.  **Absence:** If no intermediary bank details (name or address) are provided, this field should be 'null'.

            **Output Format:**
            Extract the full address as a single string, using '\\n' for line breaks if applicable. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` does not provide details for an Intermediary Bank. Therefore, the output for its address would be 'null'.

            **Task:** Locate and accurately extract the intermediary bank's full address. If not found, indicate 'null'.
            """,
        },
        {
            "name": "INTERMEDIARY BANK COUNTRY",
            "description": """Extract the country where the intermediary bank is located, if one is specified.

            **Typical Location & Labels:**
            This would typically be part of the intermediary bank's address, found in a section detailing the intermediary or correspondent bank.

            **Important Considerations for Extraction:**
            1.  **Address Component:** Identify the country from the full address of the intermediary bank.
            2.  **Absence:** If no intermediary bank address is provided, or if the country is not specified within that address, this field should be 'null'.

            **Output Format:**
            Extract the country name as a single string. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` does not provide details for an Intermediary Bank. Therefore, the output for its country would be 'null'.

            **Task:** Locate and accurately extract the intermediary bank's country. If not found, indicate 'null'.
            """,
        },
        {
            "name": "Party Name ( Applicant )",
            "description": """Extract the name of the applicant, which on an invoice typically refers to the buyer, customer, or the party being billed.

            **Typical Location & Labels:**
            This information is usually found in sections labeled 'Bill To:', 'Customer:', 'Buyer:', 'To:', 'Ship To:' (if same as Bill To), or 'Applicant:'.

            **Important Considerations for Extraction:**
            1.  **Role Identification:** Ensure the name extracted is that of the party receiving the invoice and responsible for payment (the buyer/customer).
            2.  **Completeness:** Extract the full name of the company or individual, including any legal suffixes (e.g., Pvt Ltd, Inc, Srl).
            3.  **OCR Imperfections:** Company names can be misread; extract the most plausible and complete name.

            **Output Format:**
            Extract the name as a single string.

            **Example (from the `INVOICE.pdf` context):**
            The 'Bill to:' and 'Ship to:' sections both list 'THREE ONE COFFEE ENGINEERING PVT LTD.'[cite: 6, 7]. This is the applicant/buyer.

            **Task:** Identify and extract the applicant's (buyer's) name.
            """,
        },
        {
            "name": "Party Name ( Beneficiary )", 
            "description": """Extract the name of the party who is the beneficiary of the payment or transaction related to the invoice. This is typically the seller/exporter.

            **Typical Location & Labels:**
            This name is usually the same as the 'SELLER_NAME' or the previously defined 'BENEFICIARY_NAME'. It's the entity issuing the invoice and receiving the payment. Look for the main company name in the letterhead or under specific labels like 'Beneficiary' in payment sections if distinct from the seller's main identity.

            **Important Considerations for Extraction:**
            1.  **Primary Identification:** Identify the ultimate recipient of the funds.
            2.  **Synonymous with Seller:** On most invoices, this will be the seller's name.
            3.  **Completeness:** Extract the full name, including any legal suffixes (e.g., Srl, Ltd, Inc.).
            4.  **OCR Imperfections:** Company names can be subject to OCR errors.

            **Output Format:**
            Extract the name as a single string.

            **Example (from the `INVOICE.pdf` context):**
            The seller, and therefore the beneficiary of the payment, is 'La Marzocco Srl'.

            **Task:** Identify and extract the beneficiary's (seller's) name.
            """,
        },
        {
            "name": "Party Country ( Beneficiary )", 
            "description": """Extract the country where the beneficiary (typically the seller/exporter) is officially located or registered.

            **Typical Location & Labels:**
            This is generally the country found in the 'SELLER_ADDRESS' or 'BENEFICIARY_ADDRESS'. It's usually the last component of the address or explicitly stated.

            **Important Considerations for Extraction:**
            1.  **Based on Address:** Determine the country from the beneficiary's (seller's) main address on the invoice.
            2.  **OCR Imperfections:** Country names can be misread.
            3.  **Standard Name:** Use the standard English name of the country if possible (e.g., 'Italy' instead of 'Italia', though 'Italia' is also understandable).

            **Output Format:**
            Extract the country name as a single string.

            **Example (from the `INVOICE.pdf` context):**
            The beneficiary (La Marzocco Srl) is located in 'Italia' [cite: 1] (Italy).

            **Task:** Identify and extract the beneficiary's (seller's) country.
            """,
        },
        {
           "name": "Party Type ( Beneficiary Bank )", 
            "description": """Extract the role or classification of the beneficiary's bank (e.g., 'Beneficiary Bank', 'Depository Bank', 'Receiving Bank').

            **Typical Location & Labels:**
            This might be explicitly stated, or inferred from the section where the beneficiary's bank details are provided (e.g., a bank listed under 'Beneficiary Bank Details' is implicitly the 'Beneficiary Bank'). Labels like 'Banca d'Appoggio' also indicate this role.

            **Important Considerations for Extraction:**
            1.  **Inference from Context:** If not explicitly labeled, the type is usually inferred from its function of holding the beneficiary's account for payment.
            2.  **Common Terms:** 'Beneficiary Bank' is a common and generally applicable term if no other specific classification is given.

            **Output Format:**
            Extract the party type as a string.

            **Example (from the `INVOICE.pdf` context):**
            The bank details are listed under 'Banca d'Appoggio / Bank'[cite: 5], which translates to Supporting Bank or Depository Bank. This clearly indicates its role as the 'Beneficiary Bank'.

            **Task:** Identify and extract the type or role of the beneficiary's bank.
            """,
        },
        {
            "name": "Party Name ( Beneficiary Bank )",
            "description": """Extract the name of the bank that holds the account for the beneficiary (typically the seller/exporter). This is synonymous with the 'BENEFICIARY_BANK' field.

            **Typical Location & Labels:**
            This information is usually found in a 'Bank Details', 'Payment Instructions', or 'Beneficiary Bank Account Details' section. Look for labels like 'Bank Name', 'Beneficiary Bank', 'Bank', 'Payable to Bank', 'Banca d'Appoggio'.

            **Important Considerations for Extraction:**
            1.  **Clarity:** Extract the official name of the bank.
            2.  **Association:** Ensure the bank name extracted is clearly associated with the beneficiary's account details for receiving payment.
            3.  **Absence:** If the bank name is not explicitly provided in the relevant sections, this field should be 'null'. This often occurs if only an IBAN is provided without the bank's name alongside it.

            **Output Format:**
            Extract the bank name as a single string. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` shows an IBAN under 'Banca d'Appoggio / Bank' but does not explicitly state the beneficiary bank's name in that section or clearly nearby in the provided text. Therefore, for this document, the output would be 'null'.

            **Task:** Locate and accurately extract the beneficiary's bank name. If not found, indicate 'null'.
            """,
        },
        {
            "name": "Party Country ( Beneficiary Bank )", 
            "description": """Extract the country where the beneficiary's bank is located.

            **Typical Location & Labels:**
            This information would typically be part of the beneficiary bank's address, found in the 'Bank Details' or 'Payment Instructions' section. It can also often be reliably inferred from the first two letters of the IBAN (the country code).

            **Important Considerations for Extraction:**
            1.  **Explicit vs. Inferred:** If the bank's address (including country) is explicitly stated, use that. If not, the country can be inferred from the IBAN's country code (e.g., 'IT' in an IBAN indicates Italy, 'DE' indicates Germany).
            2.  **Accuracy of Inference:** Ensure the inference from the IBAN is based on the standard two-letter ISO country code.
            3.  **Absence:** If no bank address is provided and no IBAN is present from which to infer the country, this field should be 'null'.

            **Output Format:**
            Extract the country name as a single string.

            **Example (from the `INVOICE.pdf` context):**
            The beneficiary's IBAN provided is 'IT2710503438100000000127869'[cite: 5]. The 'IT' prefix indicates that the bank is located in Italy. Therefore, the output is 'Italy'.

            **Task:** Locate and accurately extract the beneficiary bank's country, inferring from the IBAN if necessary and explicitly stated otherwise.
            """,
        },
        {
            "name": "Drawee Address", 
            "description": """Extract the full name and address of the drawee, the party on whom a bill of exchange or draft (if applicable to the transaction) is drawn. On a standard invoice without mention of such instruments, the drawee is generally considered the buyer/importer who is responsible for payment.

            **Typical Location & Labels:**
            Look for terms like 'Drawee'. If no specific drawee is mentioned, this typically defaults to the buyer's name and address as listed in 'Bill To:' or 'Customer Details'.

            **Important Considerations for Extraction:**
            1.  **Context of Drawee:** Understand if a bill of exchange is involved. If not, the buyer's details are usually appropriate.
            2.  **Full Address:** Capture the complete name and address, including street, city, postal code, state/province, and country.
            3.  **Absence:** If the invoice is not related to a bill of exchange and buyer information is also missing, this field might be 'null'.

            **Output Format:**
            Extract the full name and address as a single string, using '\\n' for line breaks within the address.

            **Example (from the `INVOICE.pdf` context):**
            The invoice does not mention a specific bill of exchange or a separately identified drawee. Therefore, the drawee is considered to be the buyer. The buyer is 'THREE ONE COFFEE ENGINEERING PVT LTD.' [cite: 6] and their address is 'DMG-180, TULSIWADI\\nBHANJIBHAI RATHOD ROAD\\nTARDEO, MUMBAI\\n400034 MAHARASTRA\\nINDIA'[cite: 7]. This full name and address should be extracted.

            **Task:** Identify and extract the drawee's full name and address.
            """,
        },
        {
            "name": "PORT OF LOADING",
            "description": """Extract the specific port, airport, or place where the goods are loaded onto the main international transport vessel, aircraft, or vehicle for export. This can also be the 'Place of Receipt' if it signifies the start of the main carriage.

            **Typical Location & Labels:**
            Look for labels such as 'Port of Loading', 'POL', 'From Port', 'Airport of Departure', 'Place of Loading', 'Shipped From'. This information is often found in the shipping details section of an invoice or related transport documents. For Incoterms like EXW (Ex Works), the place of loading is the seller's premises or another named place where the buyer takes possession.

            **Important Considerations for Extraction:**
            1.  **Named Place:** This should be a specific geographical location (city, port name, airport code).
            2.  **Incoterm Context:** Pay attention to the Incoterms used (e.g., for EXW, the 'Port of Loading' is the seller's named place of delivery; for FOB, it's a specified port).
            3.  **Absence:** If no port or place of loading is explicitly mentioned or clearly inferable, this field should be 'null'.

            **Output Format:**
            Extract the location name as a single string. Include any relevant context if helpful (e.g., 'Scarperia e San Piero (FI) - Italia (Ex Works)').

            **Example (from the `INVOICE.pdf` context):**
            The invoice specifies the Incoterm as 'Ex Works'. The seller (La Marzocco Srl) is located in 'Scarperia e San Piero (FI) - Italia'. Therefore, the place of loading is effectively the seller's premises in Scarperia. An appropriate extraction would be 'Scarperia e San Piero (FI) - Italia (Ex Works)'.

            **Task:** Identify and extract the port or place of loading.
            """,
        },
        {
            "name": "PORT OF DISCHARGE", 
            "description": """Extract the specific port, airport, or place where the goods are to be unloaded from the main international transport after arrival in the destination country. This can also be the 'Place of Final Delivery' if it signifies the end of the main carriage.

            **Typical Location & Labels:**
            Look for labels such as 'Port of Discharge', 'POD', 'To Port', 'Airport of Destination', 'Place of Unloading', 'Final Destination' (if a port/airport). This information is usually found in the shipping details section.

            **Important Considerations for Extraction:**
            1.  **Named Place:** This should be a specific geographical location.
            2.  **Destination Context:** This is the entry point into the buyer's country or the final agreed delivery point for the main transport.
            3.  **Absence:** This is often specified on Bills of Lading or Commercial Invoices. It may not always be present on a Pro Forma Invoice, especially if shipping arrangements are not finalized or are the buyer's responsibility (e.g., under Ex Works terms). If not found, indicate 'null'.

            **Output Format:**
            Extract the location name as a single string.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` is a proforma invoice with Incoterms 'Ex Works'[cite: 12], meaning the buyer is responsible for shipment from the seller's premises. The buyer is in Mumbai, India[cite: 6, 7]. While a port in/near Mumbai (e.g., Nhava Sheva, Mumbai Port) would be the likely destination port, it is not explicitly stated on this document. Therefore, the output would be 'null'.

            **Task:** Identify and extract the port or place of discharge. If not found, indicate 'null'.
            """,
        },
        {
            "name": "VESSEL TYPE",
            "description": """Extract the general type of transport conveyance used or anticipated for the main leg of the journey for the goods (e.g., 'Vessel', 'Container Ship', 'Aircraft', 'Cargo Plane', 'Truck', 'Rail Car').

            **Typical Location & Labels:**
            This information might be found near shipping details, 'Mode of Transit', or specific conveyance identifiers. Labels could include 'Type of Vessel', 'Conveyance Type', or it might be inferred if the 'Mode of Transit' (e.g., 'Ocean Freight', 'Air Freight') is specified.

            **Important Considerations for Extraction:**
            1.  **Inference from Mode of Transit:** If 'Mode of Transit' is 'Sea' or 'Ocean', 'Vessel' or 'Container Ship' (if more specific details are available) could be inferred. If 'Air', then 'Aircraft' or 'Cargo Plane'.
            2.  **Specificity:** Extract the most specific type mentioned.
            3.  **Absence:** This information may not be present, especially on proforma invoices where transport details are not yet finalized or are the buyer's responsibility (e.g., under EXW Incoterms). If not found or clearly inferable, indicate 'null'.

            **Output Format:**
            Extract the vessel/conveyance type as a string. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` does not specify the 'Mode of Transit' and uses 'Ex Works' Incoterms[cite: 12]. As such, the specific type of vessel or conveyance for the main transport is not detailed by the seller. Therefore, the output would be 'null'.

            **Task:** Identify and extract the type of vessel or main transport conveyance. If not found or inferable, indicate 'null'.
            """,
        },
        {
            "name": "VESSEL NAME", 
            "description": """Extract the specific name of the ship or vessel carrying the goods, or the flight number if transport is by air, or the voyage number if applicable.

            **Typical Location & Labels:**
            Look for labels such as 'Vessel Name', 'Name of Ship', 'Voyage No.', 'Flight No.', 'Carrier Name' (if it refers to a specific vessel/flight rather than the shipping line). This is often found in shipping details sections.

            **Important Considerations for Extraction:**
            1.  **Specificity:** This refers to the unique identifier of a specific transport conveyance, not just the type.
            2.  **Context:** Ensure it's the identifier for the main international transport.
            3.  **Absence:** This information is often not available on proforma invoices, especially when shipping arrangements are not yet made or are handled by the buyer (e.g., EXW terms). If not found, indicate 'null'.

            **Output Format:**
            Extract the name or number as a single string. If not found, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` is a proforma invoice with 'Ex Works' Incoterms[cite: 12]. Specific shipping details like vessel name or flight number are not provided. Therefore, the output would be 'null'.

            **Task:** Identify and extract the vessel name, voyage number, or flight number. If not found, indicate 'null'.
            """,
        },
        {
            "name": "THIRD PARTY EXPORTER NAME",
            "description": """Extract the name of a third-party exporter, if this entity is different from the primary seller listed on the invoice and is explicitly mentioned as handling export formalities or being the exporter of record.

            **Typical Location & Labels:**
            Look for specific fields such as 'Third Party Exporter', 'Exporter (if different from Seller)', or if the 'Exporter' field clearly names an entity different from the one identified as the 'Seller' or 'Beneficiary'.

            **Important Considerations for Extraction:**
            1.  **Distinction from Seller:** This field is specifically for an exporter entity that is *not* the main seller. If the seller is also the exporter (which is common), this field should be 'null'.
            2.  **Explicit Mention:** Only extract a name if a third-party exporter is clearly and explicitly identified.
            3.  **Absence:** If not mentioned or not applicable (i.e., the seller is the sole exporter), this field should be 'null'.

            **Output Format:**
            Extract the name as a single string. If not found or not applicable, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            The `INVOICE.pdf` is issued by 'La Marzocco Srl', who is also the manufacturer and seller. There is no indication or mention of a separate third-party exporter. Therefore, the output for this field would be 'null'.

            **Task:** Identify and extract the name of a third-party exporter, if specified as distinct from the seller. If not specified, indicate 'null'.
            """,
        },
        {
            "name": "THIRD PARTY EXPORTER COUNTRY", 
            "description": """Extract the country of the third-party exporter, if such a party is named on the invoice and is distinct from the primary seller.

            **Typical Location & Labels:**
            This information would be part of the third-party exporter's address details, typically found if a third-party exporter's name and address are provided.

            **Important Considerations for Extraction:**
            1.  **Prerequisite:** A third-party exporter must first be identified. If no third-party exporter is named, this field will be 'null'.
            2.  **Address Component:** If a third-party exporter is listed, extract the country from their provided address.
            3.  **Absence:** If no third-party exporter is mentioned, or if their country is not specified, this field should be 'null'.

            **Output Format:**
            Extract the country name as a single string. If not found or not applicable, return 'null'.

            **Example (from the `INVOICE.pdf` context):**
            As no third-party exporter is mentioned in the `INVOICE.pdf`, their country cannot be extracted. Therefore, the output for this field would be 'null'.

            **Task:** Identify and extract the country of the third-party exporter, if specified. If not applicable, indicate 'null'.
            """
        }
    ]
}

# --- NEW: Classification Prompt Template ---
CLASSIFICATION_PROMPT_TEMPLATE = """
**Task:** You are an AI Document Classification Specialist. Your objective is to meticulously analyze the provided document pages ({num_pages} pages) and accurately classify the document's primary type based on its intrinsic purpose, structural characteristics, and specific content elements. The document may consist of multiple pages that collectively form a single logical entity.

**Acceptable Document Types:**
{acceptable_types_str}
Strictly classify the document ONLY amongst the acceptable document types.

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

Important: Your response must be ONLY the valid JSON object. No greetings, apologies, or any text outside the JSON structure.
"""

EXTRACTION_PROMPT_TEMPLATE = """
**Your Role:** You are a highly meticulous and accurate AI Document Analysis Specialist. Your primary function is to extract structured data from business documents precisely according to instructions, with an extreme emphasis on the certainty, verifiability, and contextual appropriateness of every character and field extracted.

**Task:** Analyze the provided {num_pages} pages, which together constitute a single logical '{doc_type}' document associated with Case ID '{case_id}'. Carefully extract the specific data fields listed below. Use the provided detailed descriptions to understand the context, meaning, typical location, expected format, and potential variations of each field within this document type. Consider all pages to find the most relevant and accurate information. Pay close attention to nuanced instructions, including differentiation between similar concepts and rules for inference or default values if specified.

**Fields to Extract (Name and Detailed Description):**
{field_list_str}

**Output Requirements (Strict):**

1.  **JSON Only:** You MUST return ONLY a single, valid JSON object as your response. Do NOT include any introductory text, explanations, summaries, apologies, or any other text outside of the JSON structure. The response must start directly with `{{` and end with `}}`.
2.  **JSON Structure:** The JSON object MUST have keys corresponding EXACTLY to the field **names** provided in the "Fields to Extract" list above. STRICTLY MAKE SURE IT IS A VALID JSON WITH NO EXTRA QUOTES, COMMAS, SPECIAL CHARACTERS ETC. AND CAN BE PARSED PROGRAMATICALLY BY A PARSER.
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

Important: Your response must be ONLY the valid JSON object. No greetings, apologies, or any text outside the JSON structure.
"""