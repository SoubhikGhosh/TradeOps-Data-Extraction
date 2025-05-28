# config.py
import os
from dotenv import load_dotenv
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

load_dotenv() # Optional: Load environment variables from a .env file

# --- Vertex AI Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "hbl-uat-ocr-fw-app-prj-spk-4d")
LOCATION = "asia-south1"
# Use a powerful multimodal model capable of handling PDFs and complex instructions
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-002") # Or gemini-1.5-flash / newer appropriate model
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com" # Often not needed if default is correct

# --- Supported File Types ---
SUPPORTED_MIME_TYPES = {
    "application/pdf": "PDF",
    "image/png": "PNG", 
    "image/jpeg": "JPEG",
    "image/jpg": "JPEG"
}

SUPPORTED_FILE_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg"]

# --- Safety Settings ---
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
        },
        {
        "name": "BENEFICIARY ACCOUNT NO / IBAN",
        "description": """
    **You are an expert data extraction system. Your task is to extract the Beneficiary's Bank Account Number or IBAN from the document.**

    **Objective:** Accurately locate and extract the beneficiary's bank account number or International Bank Account Number (IBAN) where the funds are to be credited.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Account No.:', 'A/C No.:', 'Account Number:', 'Beneficiary Account No.:', 'IBAN:', 'Beneficiary IBAN:', 'Acc No:', 'A/c ID:'.
        * **Location:** This information is typically found within the 'Beneficiary Details' or 'Beneficiary Bank Details' section, often close to the beneficiary's name and bank name.
        * **Format:**
            * **Account Numbers** vary widely in format (can be numeric, alphanumeric, may contain hyphens or spaces).
            * **IBANs** have a specific structure: they start with a two-letter country code, followed by two check digits, and then a country-specific Basic Bank Account Number (BBAN) which can be up to 30 alphanumeric characters (e.g., DE89370400440532013000, GB29NWBK60161331926819).
    2.  **What to Extract:**
        * Extract the **complete and exact account number or IBAN** as it appears.
        * Include all alphanumeric characters and any embedded hyphens or spaces if they are part of the presented number (though it's common to normalize by removing spaces/hyphens later). For extraction, capture as presented.
        * If both an IBAN and a local account number are present for the beneficiary, prioritize the IBAN.

    **Examples of Account No / IBAN text:**
    * "IBAN: DE89370400440532013000"
    * "Account No.: 001-234567-890"
    * "A/C No: 218246110956"
    * "Beneficiary Account Number: FR7630006000011234567890189"

    **Output Requirements:**
    * **Format:** Return the extracted account number or IBAN as a **string**.
    * **If Not Found:** If the Beneficiary Account No / IBAN cannot be clearly identified, return **null**.
    * **Normalization Note:** While extracting as presented, downstream processes might normalize by removing spaces and hyphens.
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
        * Do not include the Applicant Name itself in the address string if it's clearly separate.

    **Examples of Applicant Address text:**
    * "456 Business Park, Suite 101, Metro City, MC 67890, Localia"
    * "A-112, KUSHAL MANGAL IND'S ESTATE TUNGAR PHATA, NEAR WESTER EXPRESS HIGHWAY, VASAI EAST 401208"

    **Output Requirements:**
    * **Format:** Return the extracted address as a **single string**.
    * **If Not Found:** If the Applicant Address cannot be clearly identified or is absent, return **null**.
    * **Completeness:** Ensure all components of the address visible are extracted.
    """
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
        * **Context:** This should be the country associated with the applicant's main address.
    2.  **What to Extract:**
        * Extract the full name of the country.
        * Avoid extracting city or state names as the country.

    **Examples of Applicant Country text:**
    * "India"
    * "United Kingdom"
    * "Localia" (if 'Localia' is used as a country name in the example context)

    **Output Requirements:**
    * **Format:** Return the extracted country name as a **string**.
    * **If Not Found:** If the Applicant Country cannot be clearly identified or is absent, return **null**.
    * **Standardization (Optional):** If possible, standardize to common country names. If unsure, extract as written.
    """
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
    """
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
    """
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
    """
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
    """
        },
        {
        "name": "LATEST SHIPMENT DATE",
        "description": """
    **You are an expert data extraction system. Your task is to extract the Latest Shipment Date from the document.**

    **Objective:** Accurately locate and extract the latest date by which the goods must be shipped by the exporter/seller, as specified in the document (often related to terms in a Letter of Credit, purchase order, or proforma invoice referenced in the request).

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Latest Shipment Date:', 'Shipment by:', 'Latest Date of Shipment:', 'LSD:'.
        * **Location:** This date is typically found in sections detailing shipping terms, Letter of Credit conditions, or purchase order specifics.
        * **Format:** Dates can appear in various formats (e.g., DD-MM-YYYY, MM/DD/YYYY, YYYY-MON-DD).
    2.  **What to Extract:**
        * Extract the complete date.

    **Examples of Latest Shipment Date text:**
    * "Latest Shipment Date: 31-12-2024"
    * "Shipment by: 15/01/2025"

    **Output Requirements:**
    * **Format:** Standardize and return the extracted date in **DD-MM-YYYY** format.
    * **If Not Found:** If the Latest Shipment Date cannot be clearly identified or is absent, return **null**.
    """
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
    """
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
    """
        },
        {
        "name": "FB CHARGES",
        "description": """
    **You are an expert data extraction system. Your task is to extract who bears the Foreign Bank Charges from the document.**

    **Objective:** Accurately locate and extract the instruction indicating who is responsible for paying foreign bank charges. This is typically represented by a three-letter code: BEN (Beneficiary), OUR (Applicant), or SHA (Shared).

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels:** Look for labels such as 'Foreign Bank Charges:', 'Details of Charges:', 'Charges Borne By:', 'Bank Charges:'.
        * **Location:** Often found in a section related to payment details or charges.
        * **Content:** The value is usually one of 'BEN', 'OUR', or 'SHA'. Sometimes options are presented with checkboxes or one is circled.
            * **BEN:** Beneficiary pays all charges (foreign bank charges deducted from remittance).
            * **OUR:** Applicant/Remitter pays all charges (beneficiary receives full amount).
            * **SHA:** Applicant pays their bank's charges, Beneficiary pays their bank's charges (shared).
    2.  **What to Extract:**
        * Extract the specific code (BEN, OUR, SHA) or word indicating the responsible party.
        * If options are given (e.g., checkboxes for "on us", "on beneficiary"), determine which is selected. "on us" generally maps to OUR, "on beneficiary" to BEN.

    **Examples of FB Charges text:**
    * "Foreign Bank Charges: BEN" (Extract "BEN")
    * "Details of Charges: OUR" (Extract "OUR")
    * A checkbox next to "OUR" is marked. (Extract "OUR")
    * "Charges: ()on us (X)on beneficiary" (Extract "BEN" as 'on beneficiary' is selected)

    **Output Requirements:**
    * **Format:** Return the extracted code/term as a **string** (e.g., "BEN", "OUR", "SHA").
    * **If Not Found:** If the instruction for foreign bank charges cannot be clearly identified, return **null**.
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
        },
        {
        "name": "INVOICE NO",
        "description": """
    **You are an expert data extraction system. Your task is to extract the Invoice Number referenced in the Customer Request Letter (CRL).**

    **Objective:** Accurately locate and extract the unique identification number of the Proforma Invoice or Commercial Invoice that this remittance request pertains to. This information is extracted *from the CRL itself* where it makes reference to an invoice.

    **Guidance for Extraction:**
    1.  **Identification Cues:**
        * **Labels within CRL:** Look for labels such as 'Invoice No.:', 'Ref. Invoice:', 'Against Invoice No.:', 'Proforma Invoice No.:', 'PI No.:', 'Commercial Invoice No.:'.
        * **Location:** This reference is typically found in sections detailing the purpose of payment, goods description, or in a specific area for invoice details within the CRL.
        * **Context:** The invoice number is usually alphanumeric and can contain hyphens, slashes, or other characters. It's an identifier issued by the beneficiary/exporter on their invoice.
    2.  **What to Extract:**
        * Extract the **complete and exact invoice number** as referenced in the CRL.

    **Examples of Invoice No. text found in a CRL:**
    * "Invoice No.: PI-2024-001" (Extract "PI-2024-001")
    * "Ref. Invoice: EXPORTINV/789/ABC" (Extract "EXPORTINV/789/ABC")
    * "Against Proforma Invoice KET-20231222" (Extract "KET-20231222")

    **Output Requirements:**
    * **Format:** Return the extracted invoice number as a **string**.
    * **If Not Found:** If no invoice number is referenced in the CRL, return **null**.
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
        }
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

# --- Processing Configuration ---
MAX_WORKERS = 4 # Adjust based on CPU cores and API limits for parallel processing
TEMP_DIR = "temp_processing"
OUTPUT_FILENAME = "extracted_data.xlsx"

# --- Logging Configuration ---
LOG_FILE = "app_log.log"
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- NEW: Classification Prompt Template ---
CLASSIFICATION_PROMPT_TEMPLATE = """
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

Important: Your response must be ONLY the valid JSON object. No greetings, apologies, or any text outside the JSON structure.
"""

EXTRACTION_PROMPT_TEMPLATE = """
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

Important: Your response must be ONLY the valid JSON object. No greetings, apologies, or any text outside the JSON structure.
"""

