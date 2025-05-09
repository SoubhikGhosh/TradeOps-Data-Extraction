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
        {
            "name": "DATE & TIME OF RECEIPT FROM CIRCULAR SEAL",
            "description": """
    **Objective:** Analyze the provided document image to locate a specific circular seal and accurately extract the date and time indicated by it. The time extraction is of particular importance and should be handled with precision.

    **Seal Description:**
    1.  **Structure:** Identify a circular seal composed of two concentric circles.
    2.  **Inner Circle (Date):** Contains a date in some recognizable format (e.g.,<ctrl3348>-MM-DD, DD MON<ctrl3348>, MM/DD/YY). Directly **above** the date, centered within the inner circle, is a distinct **triangular pointer (arrow)**. The base of the triangle is towards the center of the seal, and the **sharp tip points radially outwards** towards the outer ring.
    3.  **Outer Ring (24-Hour Time Scale):** The area between the two circles functions as a 24-hour time scale.
        * It is divided into **24 equally spaced major segments**, representing the hours 00 through 23. Assume these are arranged sequentially in a clockwise direction, with the 00 hour typically located at the 12 o'clock (top) position.
        * Each major hour segment is further subdivided into **4 equally spaced minor segments**, representing 15-minute intervals within that hour (i.e., :00, :15, :30, :45).
    4.  **Time Indication (Crucial):** The **precise tip** of the triangular pointer (located above the date in the inner circle) points **directly to a specific marking** on the 24-hour time scale in the outer ring. This marking indicates the exact hour and the 15-minute interval of the time.

    **Task:**
    1.  **Locate Seal:** Find the described circular seal within the document image.
    2.  **Extract Date:** Perform OCR on the inner circle to reliably read and extract the full date. Note the extracted date format.
    3.  **Determine Time (Accurate Interpretation Required):**
        * **Identify the Hour:** Determine which of the 24 major hour segments the tip of the triangular pointer is aligned with.
        * **Identify the Minute Interval:** Determine which of the four 15-minute minor segments within that hour segment the tip of the triangular pointer is aligned with.
        * **Calculate Time:** Convert the identified hour and minute interval into HH:MM format (24-hour clock). Ensure accuracy in this conversion. For example, if the pointer is slightly past the '03' hour mark and points to the second minor segment, the time should be interpreted as 03:30.
    4.  **Handle Imperfections and Estimation:** The seal or document image might have issues like fading, partial obstruction, incompleteness, or rotation.
        * Utilize any visible portions of the seal (circles, pointer, date, time markings) to infer the complete information.
        * If the pointer falls between clear markings, use geometric reasoning. Assume a full circle is 360 degrees, each hour segment spans 15 degrees ($360/24$), and each 15-minute segment spans 3.75 degrees ($15/4$). Calculate the angle of the pointer relative to a known reference point (like the inferred center and the 00-hour mark) to estimate the time as accurately as possible. **Explicitly state if estimation was necessary.**

    **Output:**
    * Provide the extracted **Date** and extracted **Time** in DD-MM-YYYY HH:MM format (24-hour clock).
    * If the time was estimated due to imprecise pointer alignment or other visual ambiguities, please include a note indicating that the time was **estimated**.
  """,
        },
        {
            "name": "CUSTOMER REQUEST LETTER DATE",
            "description": "The date mentioned on the customer's formal request letter.",
        },
        {
            "name": "BENEFICIARY NAME",
            "description": "The name of the party (typically the exporter/seller) who is entitled to receive payment under the credit.",
        },
        {
            "name": "BENEFICIARY ADDRESS",
            "description": "The full address of the beneficiary (exporter/seller).",
        },
        {
            "name": "BENEFICIARY COUNTRY",
            "description": "The country where the beneficiary is located.",
        },
        {
            "name": "CURRENCY",
            "description": "The specific currency code (e.g., USD, EUR, INR) for the transaction amount.",
        },
        {
            "name": "AMOUNT",
            "description": "The principal monetary value of the transaction or credit.",
        },
        {
            "name": "BENEFICIARY ACCOUNT NO / IBAN",
            "description": "The beneficiary's bank account number or International Bank Account Number (IBAN) for receiving funds.",
        },
        {
            "name": "BENEFICIARY BANK",
            "description": "The name of the bank where the beneficiary holds their account.",
        },
        {
            "name": "BENEFICIARY BANK ADDRESS",
            "description": "The full address of the beneficiary's bank.",
        },
        {
            "name": "BENEFICIARY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE",
            "description": "The unique identification code of the beneficiary's bank (SWIFT/BIC for international, Sort Code for UK, BSB for Australia, IFSC for India).",
        },
        {
            "name": "STANDARD DECLARATIONS AS PER PRODUCTS",
            "description": "Any standard clauses, declarations, or compliance statements required for the specific financial product.",
        },
        {
            "name": "APPLICANT SIGNATURE",
            "description": "Indication or confirmation of the applicant's signature (may be text stating 'Signed' or an image area). Focus on confirmation text if present.",
        },
        {
            "name": "APPLICANT NAME",
            "description": "The name of the party (typically the importer/buyer) who requested the transaction or credit.",
        },
        {
            "name": "APPLICANT ADDRESS",
            "description": "The full address of the applicant (importer/buyer).",
        },
        {
            "name": "APPLICANT COUNTRY",
            "description": "The country where the applicant is located.",
        },
        {
            "name": "TRANSACTION Product Code Selection",
            "description": "A specific code identifying the type of financial product or transaction.",
        },
        {
            "name": "TRANSACTION EVENT",
            "description": "A code or description identifying the specific event within the transaction lifecycle (e.g., issuance, amendment).",
        },
        {
            "name": "VALUE DATE",
            "description": "The date on which the funds are expected to be credited or debited.",
        },
        {
            "name": "HS CODE",
            "description": "The Harmonized System code, an international standard for classifying traded goods.",
        },
        {
            "name": "TYPE OF GOODS",
            "description": "A general description of the merchandise being traded.",
        },
        {
            "name": "INCOTERM",
            "description": "The standardized trade term (e.g., FOB, CIF, EXW) defining buyer/seller responsibilities for shipping, risk, and costs.",
        },
        {
            "name": "DEBIT ACCOUNT NO",
            "description": "The applicant's account number from which funds will be debited.",
        },
        {
            "name": "FEE ACCOUNT NO",
            "description": "The account number from which transaction fees will be debited (if different from the main debit account).",
        },
        {
            "name": "LATEST SHIPMENT DATE",
            "description": "The latest date by which the goods must be shipped according to the credit terms.",
        },
        {
            "name": "DISPATCH PORT",
            "description": "The port or place from where the goods are dispatched or shipped (Port of Loading).",
        },
        {
            "name": "DELIVERY PORT",
            "description": "The port or place where the goods are to be delivered (Port of Discharge).",
        },
        {
            "name": "FB CHARGES",
            "description": "Details regarding who bears the foreign bank charges (e.g., BEN, OUR, SHA).",
        },
        {
            "name": "INTERMEDIARY BANK NAME",
            "description": "The name of any intermediary bank involved in the payment chain (if applicable).",
        },
        {
            "name": "INTERMEDIARY BANK ADDRESS",
            "description": "The address of the intermediary bank (if applicable).",
        },
        {
            "name": "INTERMEDIARY BANK COUNTRY",
            "description": "The country of the intermediary bank (if applicable).",
        },
        {
            "name": "THIRD PARTY EXPORTER NAME",
            "description": "Name of a third-party exporter involved, if different from the main beneficiary.",
        },
        {
            "name": "THIRD PARTY EXPORTER COUNTRY",
            "description": "Country of the third-party exporter, if applicable.",
        },
    ],
    "INVOICE": [
        {
            "name": "TYPE OF INVOICE - COMMERCIAL/PROFORMA/CUSTOMS/",
            "description": """The explicit classification of the invoice document. Search for titles or phrases like 'COMMERCIAL INVOICE', 'PROFORMA INVOICE', 'TAX INVOICE', 'CUSTOMS INVOICE', or 'INVOICE'.
                            If no explicit type is mentioned but it functions as a bill, infer 'COMMERCIAL' if it seems final for goods/services rendered.
                            If it's preliminary (e.g., for quotation, pre-shipment, or to open an L/C), infer 'PROFORMA'.
                            If it's specifically for customs purposes and includes details like HS codes and country of origin for declaration, infer 'CUSTOMS'.
                            Look across all pages, especially in headers or titles. Example: 'PROFORMA INVOICE'[cite: 4].""",
        },
        {
            "name": "INVOICE DATE",
            "description": """The specific date when the invoice was created or issued by the seller/issuer.
                            Look for labels like 'Invoice Date', 'Date', 'Issue Date', 'Date of Issue'.
                            It's usually found near the invoice number or seller's details. Ensure it's a clear date format (e.g., DD-MMM-YY, MM/DD/YYYY, YYYY-MM-DD). Example: '27-Sep-23'[cite: 4].""",
        },
        {
            "name": "INVOICE NO",
            "description": """The unique alphanumeric identifier assigned to this specific invoice by the seller/issuer.
                            Search for labels such as 'Invoice No.', 'Invoice #', 'Inv. No.', 'Reference #', 'Document No.'.
                            This is a critical field and is usually prominently displayed, often in the header or near the seller's information. Example: '2546049' listed under 'Reference #' [cite: 2] for a proforma invoice.""",
        },
        {
            "name": "BUYER NAME",
            "description": """The full legal name of the individual or company purchasing the goods or services.
                            Look for labels like 'Buyer', 'Bill To', 'Customer', 'Sold To', 'Consignee' (if also the buyer), 'Importer', 'To:', 'Applicant'.
                            It's often located in a distinct section detailing the recipient of the invoice. Example: 'Arrow Business Advisory Pvt. Ltd' [cite: 4] under 'BILL TO:'.""",
        },
        {
            "name": "BUYER ADDRESS",
            "description": """The complete mailing address of the buyer, including street, city, state/province, postal code, and potentially country.
                            This information is typically found directly below or adjacent to the 'BUYER NAME' under labels like 'Address', or within the 'Bill To' or 'Consignee' block.
                            Extract the full, multi-line address as a single string. Example: '159 Mittal Industrial Estate Sanjay Building No. 5/B Marol Naka, Andheri (East) Mumbai - 400 059 India'[cite: 4].""",
        },
        {
            "name": "BUYER COUNTRY",
            "description": """The country where the buyer is officially located or registered.
                            This is often the last line of the buyer's address or may be explicitly labeled as 'Country'.
                            If the address is multi-line, identify the country name. Example: 'India' [cite: 4] as part of the buyer's address.""",
        },
        {
            "name": "SELLER NAME",
            "description": """The full legal name of the individual or company selling the goods or services and issuing the invoice.
                            Look for labels like 'Seller', 'From', 'Shipper' (if also the seller), 'Exporter', 'Beneficiary', 'Invoice From', or it might be the company name in the letterhead.
                            Example: 'TRANSCENDIA, INC' [cite: 1] at the top of the document.""",
        },
        {
            "name": "SELLER ADDRESS",
            "description": """The complete mailing address of the seller, including street, city, state/province, postal code, and country.
                            Usually found near the 'SELLER NAME', often in the header or footer of the invoice, or under a 'Remit To' or 'From' section.
                            Extract the full, multi-line address as a single string. Example: '300 INDUSTRIAL PARKWAY RICHMOND, IN 47374'[cite: 1]. A more complete corporate HQ address might also be '9201 W. Belmont Avenue, Franklin Park, IL 60131'[cite: 27]. Prefer the address most clearly associated with the invoice issuance or seller identity on the primary invoice pages.""",
        },
        {
            "name": "SELLER COUNTRY",
            "description": """The country where the seller is officially located or registered.
                            This is typically the last line of the seller's address or may be explicitly labeled.
                            Based on the address 'RICHMOND, IN 47374'[cite: 1], the country is implicitly USA. For 'Franklin Park, IL 60131'[cite: 27], it's also USA. Explicitly state "USA" if inferred from state codes like IN or IL.""",
        },
        {
            "name": "INVOICE CURRENCY",  # Renamed from CURRENCY for clarity
            "description": """The specific currency in which the invoice amounts are denominated (e.g., USD, EUR, GBP, INR).
                            Look for currency symbols ($, €, £) or currency codes (USD, EUR) next to monetary values, especially the total amount.
                            Sometimes explicitly stated like 'All amounts in USD'. Example: 'USD' is appended to the amount '$135,750.00 USD'[cite: 3].""",
        },
        {
            "name": "INVOICE AMOUNT/VALUE",  # Renamed from AMOUNT for clarity
            "description": """The primary financial value of the invoice, typically the total sum of goods/services before certain taxes or after certain discounts, or the grand total if no other total is more prominent.
                            Search for terms like 'Total', 'Subtotal', 'Net Amount', 'Invoice Total', 'Grand Total'.
                            This should be a numerical value. Be careful to distinguish it from line item amounts if a clear overall total is present. Example: '$135,750.00'[cite: 3].""",
        },
        {
            "name": "INVOICE AMOUNT/VALUE IN WORDS",
            "description": """The total invoice amount written out in words (e.g., 'One Hundred Thirty-Five Thousand Seven Hundred Fifty Dollars Only').
                            This field is often found near the numerical total amount, sometimes labeled 'Amount in Words', 'Say Total', or just appearing as a textual representation of the sum.
                            This may not always be present. If not found, state null.""",
        },
        {
            "name": "BENEFICIARY ACCOUNT NO / IBAN",
            "description": """The bank account number or International Bank Account Number (IBAN) of the seller (beneficiary) where the payment should be sent.
                            Look for labels like 'Account No.', 'A/C No.', 'IBAN', 'Beneficiary Account'. Often found in a 'Bank Details' or 'Payment Instructions' section.
                            Example: 'Account #: 830769961' for both ACH and Wire[cite: 29].""",
        },
        {
            "name": "BENEFICARY BANK",  # Spelling from user
            "description": """The name of the bank where the seller (beneficiary) holds their account.
                            Search for labels such as 'Bank Name', 'Beneficiary Bank', 'Bank', 'Payable to Bank'.
                            This is usually listed in the payment instructions or bank details section. Example: 'JPMorgan Chase'[cite: 29].""",
        },
        {
            "name": "BENEFICAIRY BANK ADDRESS",  # Spelling from user
            "description": """The full mailing address of the seller's (beneficiary's) bank.
                            Look for this information near the beneficiary bank's name or within the 'Bank Details' section.
                            It should include street, city, and country. Example: 'New York, NY 10017'[cite: 29].""",
        },
        {
            "name": "BENEFICAIRY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE/ROUTING NO",  # Spelling from user, expanded
            "description": """The unique identification code for the seller's (beneficiary's) bank. This could be a SWIFT/BIC code (for international payments),
                            ABA Routing Number (for US payments), Sort Code (UK), BSB (Australia), IFSC (India), etc.
                            Look for labels like 'SWIFT Code', 'BIC', 'ABA No.', 'Routing No.', 'IFSC', 'Sort Code', 'BSB'. Example: 'Swift Code: CHASUS33' or 'ABA (Routing) #: 071000013' (for ACH) or 'Bank Routing Number: 021000021' (for Wire)[cite: 29]. Prioritize SWIFT if available for international context, or the most relevant routing for the transaction type.""",
        },
        {
            "name": "Total Invoice Amount",
            "description": """The final, definitive total monetary sum due on the invoice, inclusive of all items, charges, taxes (if applicable and included in the final sum), and less any deductions reflected in the total.
                            This is often labeled 'Grand Total', 'Total Amount Due', 'Total Invoice Value', 'Please Pay This Amount'.
                            It should be the ultimate figure the buyer is expected to pay. Example: '$135,750.00 USD' [cite: 3] (appears as the main extension and final sum in this proforma).""",
        },
        {
            "name": "Invoice Amount",  # Repeated field, ensure description helps differentiate or confirms synonymity
            "description": """This field typically refers to the primary sum of the invoice. It can be synonymous with 'Total Invoice Amount' if only one total is presented.
                            If there are multiple totals (e.g., Subtotal, Total before Tax, Grand Total), this should ideally capture the most representative invoiced amount, often the grand total.
                            Verify if it's distinct from other amounts like 'Subtotal'. In many cases, it will be the same as 'Total Invoice Amount'. Example: '$135,750.00 USD'[cite: 3].""",
        },
        {
            "name": "Beneficiary Name",  # Often the same as Seller Name
            "description": """The name of the ultimate recipient of the funds, usually the seller or exporter.
                            Look for labels like 'Beneficiary', 'Payable to', 'Beneficiary Name'. This is often the same as the 'SELLER NAME'.
                            Confirm if explicitly stated in a 'Payment Details' or 'Beneficiary Information' section. Example: 'Transcendia, Inc. - Depository'[cite: 29]. If just 'Transcendia, Inc.' is listed as seller[cite: 1], use that if more direct.""",
        },
        {
            "name": "Beneficiary Address",  # Often the same as Seller Address
            "description": """The full address of the beneficiary (seller/exporter) to whom the payment is directed.
                            This is commonly the same as the 'SELLER ADDRESS'. Check for specific 'Beneficiary Address' details if provided separately in payment instructions.
                            Example: '9201 W. Belmont Avenue Franklin Park, IL 60131' [cite: 29] associated with the beneficiary name.""",
        },
        {
            "name": "DESCRIPTION OF GOODS",
            "description": """A detailed account of the products or services being invoiced.
                            This is usually found in the main table or line items section of the invoice. It can include product names, codes, specifications, or service descriptions.
                            Extract all relevant descriptive text for each line item, or a summary if it's a very long list.
                            Example: 'HA Laminating Film 984mm X 1,829 LM Rolls'[cite: 3]. If multiple items, list them or summarize.""",
        },
        {
            "name": "QUANTITY OF GOODS",
            "description": """The amount or number of units for each item or service listed on the invoice.
                            Look for columns labeled 'Quantity', 'Qty', 'Units', 'No. of Items'.
                            Specify units if mentioned (e.g., pcs, kgs, hrs, SM). Example: '75,000 SM' (Square Meters)[cite: 3].""",
        },
        {
            "name": "PAYMENT TERMS",
            "description": """The conditions agreed upon for payment of the invoice, such as the timeframe and method.
                            Search for labels like 'Payment Terms', 'Terms of Payment', 'Terms'.
                            Examples include 'Net 30 days', 'Due Upon Receipt', '50% Advance, 50% on Delivery'.
                            Example: '50% advance and balance 50% after 60 days from the date of Bill of Lading (BL)'[cite: 3]. Also see 'Standard Payment terms - Net 30' [cite: 33] on a general info page, but prefer terms on the invoice itself.""",
        },
        {
            "name": "BENEFICIARY/SELLER'S SIGNATURE",
            "description": """The handwritten or digital signature of the authorized representative of the seller/beneficiary, or the typed name if a physical signature is replaced by it in a digital document.
                            Look for a signature line or block often labeled 'Seller's Signature', 'Authorized Signature', 'For [Seller Company Name]'.
                            This may not always be present, or could be a scanned image. Describe if present (e.g., "Signature present", "Typed name: Diana McGehee"). Example: 'Diana McGehee' typed below 'Sr Customer Service'[cite: 3], which might represent authorization.""",
        },
        {
            "name": "APPLICANT/BUYER'S SIGNATURE",
            "description": """The handwritten or digital signature of the authorized representative of the applicant/buyer, acknowledging the invoice or associated order.
                            Look for a signature line or block often labeled 'Buyer's Signature', 'Authorized Signature', 'Accepted By', 'For [Buyer Company Name]'.
                            This is less common on invoices themselves unless it's a proforma being accepted, but more common on related Purchase Orders. Example: 'Authorised Signatory' with a signature for 'For Arrow Business Advisory Private Limited' [cite: 4] at the bottom, indicating acceptance/issuance from buyer's perspective on a document they might have prepared or signed.""",
        },
        {
            "name": "MODE OF REMITTANCE",
            "description": """The method by which the payment is to be made (e.g., Wire Transfer, ACH, Cheque, Credit Card).
                            This information is often found within the 'Payment Instructions', 'Bank Details', or 'Payment Terms' sections.
                            The document shows 'ACH & Wire Transfer Instructions' [cite: 29] and mentions 'Credit cards are accepted' [cite: 32] and 'Remit to Address for Checks'[cite: 31]. List all applicable or the primary ones mentioned in context of this transaction.""",
        },
        {
            "name": "MODE OF TRANSIT",
            "description": """The method of transportation used for shipping the goods (e.g., Sea, Air, Road, Rail).
                            Look for labels like 'Ship Via', 'Mode of Shipment', 'Transport Mode', 'By'.
                            Example: 'Ship Via Ocean'[cite: 2], 'Mode of Shipment SEA'[cite: 12].""",
        },
        {
            "name": "INCO TERM",
            "description": """The Incoterm (International Commercial Term) specifies the responsibilities of buyers and sellers in international trade (e.g., EXW, FOB, CIF, DDP).
                            Look for labels like 'Incoterms', 'Terms of Sale', 'Freight Terms', or a three-letter code often followed by a location.
                            Example: 'EX-Works Richmond IN' [cite: 2] (EXW is the Incoterm).""",
        },
        {
            "name": "HS CODE",
            "description": """The Harmonized System (HS) code or HTS (Harmonized Tariff Schedule) code, which is an international nomenclature for the classification of products.
                            It allows customs authorities to identify products and apply duties and taxes. Look for labels like 'HS Code', 'HTS Code', 'Tariff Code'.
                            This is more common on Commercial or Customs Invoices. May not be present on all Proformas. If not found, state null.""",
        },
        {
            "name": "Intermediary Bank ( Field 56)",  # Existing field
            "description": """Details of any intermediary bank (correspondent bank) that is used to route the payment from the buyer's bank to the seller's (beneficiary's) bank.
                            Often referred to by 'Field 56' in SWIFT messages. Look for labels like 'Intermediary Bank', 'Correspondent Bank', or specific SWIFT field references if available.
                            This may not always be present or required. If not found, state null.""",
        },
        {
            "name": "INTERMEDIARY BANK NAME",
            "description": """The name of the intermediary bank, if one is specified in the payment instructions.
                            This would be under a section labeled 'Intermediary Bank'. If no such section or bank is named, state null.""",
        },
        {
            "name": "INTERMEDIARY BANK ADDRESS",
            "description": """The full address of the intermediary bank, if specified.
                            This information would be found along with the intermediary bank's name. If not present, state null.""",
        },
        {
            "name": "INTERMEDIARY BANK COUNTRY",
            "description": """The country where the intermediary bank is located, if specified.
                            Typically part of the intermediary bank's address. If not present, state null.""",
        },
        {
            "name": "Party Name ( Applicant )",  # Existing field
            "description": """The name of the party applying for a service related to the invoice, often the buyer/importer, especially in the context of a Letter of Credit or financing.
                            On an invoice, this is usually synonymous with the 'BUYER NAME'.
                            Look for labels like 'Applicant', or infer from the 'Buyer' or 'Bill To' details. Example: 'Arrow Business Advisory Pvt. Ltd'[cite: 4].""",
        },
        {
            "name": "Party Name ( Beneficiary )",  # Existing field
            "description": """The name of the party who is the beneficiary of the payment or transaction related to the invoice, typically the seller/exporter.
                            On an invoice, this is usually synonymous with the 'SELLER NAME' or 'BENEFICIARY NAME'.
                            Example: 'TRANSCENDIA, INC'[cite: 1], or more specifically 'Transcendia, Inc. - Depository'[cite: 29].""",
        },
        {
            "name": "Party Country ( Benefciary )",  # Spelling from user
            "description": """The country where the beneficiary (seller/exporter) is located.
                            This is typically the country of the 'SELLER ADDRESS' or 'BENEFICIARY ADDRESS'.
                            Example: USA (inferred from IN [cite: 1] or IL [cite: 29]).""",
        },
        {
            "name": "Party Type ( Beneficiary Bank )",  # Existing field
            "description": """The role or classification of the beneficiary's bank (e.g., 'Beneficiary Bank', 'Depository Bank').
                            This might be explicitly stated or inferred from its function of receiving funds for the beneficiary.
                            Example: 'Beneficiary Bank' is implied for JPMorgan Chase[cite: 29].""",
        },
        {
            "name": "Party Name (Beneficiary Bank )",  # Existing field, space before closing parenthesis
            "description": """The name of the bank that holds the account for the beneficiary (seller/exporter).
                            This is the same as 'BENEFICARY BANK'. Example: 'JPMorgan Chase'[cite: 29].""",
        },
        {
            "name": "Party Country ( Beneficiary Bank )",  # Existing field, space before closing parenthesis
            "description": """The country where the beneficiary's bank is located.
                            This is the country of the 'BENEFICAIRY BANK ADDRESS'. Example: USA (inferred from 'New York, NY' [cite: 29]).""",
        },
        {
            "name": "Drawee Address",  # Existing field
            "description": """The name and address of the party on whom a draft or bill of exchange (if applicable to the transaction) is drawn.
                            This is often the buyer or the buyer's bank. If no draft is mentioned or involved, this may not be applicable.
                            Look for terms like 'Drawee'. On a standard invoice, this might be the Buyer's address if they are the direct payer. Example: If a draft is drawn on the buyer, it would be the buyer's address '159 Mittal Industrial Estate...India'[cite: 4].""",
        },
        {
            "name": "PORT OF LOADING",  # Existing field
            "description": """The specific port or airport where the goods are loaded onto the main international transport vessel, aircraft, or vehicle for export.
                            Look for labels like 'Port of Loading', 'POL', 'From Port', 'Airport of Departure', 'Place of Receipt' (if it's the start of main carriage).
                            Example: 'Richmond IN' is mentioned under 'Freight EX-Works'[cite: 2], suggesting it's the point of origin/loading for an EXW term.""",
        },
        {
            "name": "PORT OF DISCHARGE",  # Existing field
            "description": """The specific port or airport where the goods are to be unloaded from the main international transport after arrival in the destination country.
                            Look for labels like 'Port of Discharge', 'POD', 'To Port', 'Airport of Destination', 'Place of Delivery' (if it's the end of main carriage).
                            Example: 'Nhava Sheva Port' [cite: 2] mentioned under 'Ship Via' and also as delivery location in PO[cite: 17].""",
        },
        {
            "name": "VESSEL TYPE",
            "description": """The general type of transport conveyance used for the main leg of the journey (e.g., 'Vessel', 'Aircraft', 'Truck', 'Container Ship').
                            This may be inferred from 'Mode of Transit' (e.g., if 'Ocean' or 'Sea', then 'Vessel').
                            Example: 'Ocean' implies a vessel[cite: 2]. If 'SEA' is mentioned[cite: 12], it also implies a vessel.""",
        },
        {
            "name": "VESSEL NAME",  # Existing field
            "description": """The specific name of the ship or vessel carrying the goods, or the flight number if by air, or voyage number.
                            Look for labels like 'Vessel Name', 'Voyage No.', 'Flight No.', 'Carrier Name'.
                            This is often found in shipping details sections. May not be present on a proforma invoice issued long before shipment. If not found, state null.""",
        },
        {
            "name": "THIRD PARTY EXPORTER NAME",  # Existing field
            "description": """The name of a third-party exporter, if different from the primary seller listed on the invoice.
                            This situation arises if another company handles the export formalities or is named as the exporter of record for other reasons.
                            Look for distinct fields like 'Third Party Exporter', or if the 'Exporter' field names a different entity than the 'Seller' or 'Beneficiary'. If not mentioned or not applicable, state null.""",
        },
        {
            "name": "THIRD PARTY EXPORTER COUNTRY",  # Existing field
            "description": """The country of the third-party exporter, if such a party is named.
                            This would be part of the third-party exporter's address details. If no third-party exporter is mentioned, state null.""",
        },
    ],
}

# --- Processing Configuration ---
MAX_WORKERS = 4  # Adjust based on CPU cores and API limits for parallel processing
TEMP_DIR = "temp_processing"
OUTPUT_FILENAME = "extracted_data.xlsx"

# --- Logging Configuration ---
LOG_FILE = "app_log.log"
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- NEW: Classification Prompt Template ---
CLASSIFICATION_PROMPT_TEMPLATE = """
**Task:** You are an AI Document Classification Specialist. Your objective is to meticulously analyze the provided document pages ({num_pages} pages) and accurately classify the document's primary type based on its intrinsic purpose, structural characteristics, and specific content elements. The document may consist of multiple pages that collectively form a single logical entity.

**Acceptable Document Types:**
{acceptable_types_str}

**Detailed Instructions for Classification:**

1.  **Holistic Review:** Conduct a comprehensive examination of all pages. Pay close attention to titles, headings, recurring phrases, specific keywords, data tables, and the overall layout to discern the document's fundamental function.
2.  **Content and Keyword Analysis:**
    * **INVOICE (Commercial, Proforma, Customs):**
        * **Keywords:** Look for explicit titles like "Invoice", "Commercial Invoice", "Proforma Invoice", "Tax Invoice", "Customs Invoice". Also search for related terms such as "Bill", "Statement of Charges".
        * **Core Fields:** Identify the presence of:
            * A unique "Invoice Number" or "Invoice ID".
            * Detailed "Seller" (or "Shipper", "Exporter", "Beneficiary", "From") and "Buyer" (or "Consignee", "Importer", "Bill To", "To") information (names, addresses).
            * Line items detailing goods or services, including descriptions, quantities, unit prices, and total amounts per item.
            * A "Total Amount Due", "Grand Total", or similar aggregate financial sum.
            * "Invoice Date" or "Date of Issue".
            * Payment terms, bank details for payment.
        * **Differentiation:**
            * **Commercial Invoice:** Typically used for actual billing and payment for goods/services already shipped or rendered.
            * **Proforma Invoice:** An estimate or quotation provided *before* goods are shipped or services rendered. Often states "Proforma Invoice" clearly. May lack a definitive "due date" in the same way a commercial invoice does and might be used for initiating a letter of credit.
            * **Customs Invoice:** Specifically designed for customs clearance, detailing goods for import/export, values, HS codes, country of origin. May have specific fields required by customs authorities.
    * **CRL (Customer Request Letter) / Application:**
        * **Keywords:** Search for terms like "Application for...", "Request for...", "Letter of Instruction", "To The Manager", "We request you to...".
        * **Content Focus:** Look for explicit requests made to a bank or financial institution (e.g., "issue a Letter of Credit", "process a payment", "debit our account").
        * **Parties:** Identifies an "Applicant" (the one making the request) and often a "Beneficiary" (the recipient of the transaction). Details of the transaction (amount, currency, purpose) are central.
    * **PACKING_LIST:**
        * **Keywords:** "Packing List", "Shipping List", "Delivery Note" (if it details package contents without prices).
        * **Content Focus:** Emphasizes logistics and shipping details:
            * Shipper/Consignee information.
            * Detailed list of packages, marks and numbers on packages.
            * Description of goods per package.
            * Quantities, gross weight, net weight, and measurements (dimensions/volume) of packages.
            * Typically *excludes* pricing information (unit prices, total invoice value).
    * **BL (Bill of Lading) / Air Waybill / Transport Document:**
        * **Keywords:** "Bill of Lading", "Air Waybill (AWB)", "Sea Waybill", "CMR Note", "Consignment Note".
        * **Content Focus:** Acts as a receipt for shipment and a contract of carriage.
            * Identifies Shipper, Consignee, Notify Party.
            * Details of the carrier (vessel name, voyage number, flight number).
            * Ports/places of loading and discharge/delivery.
            * Description of goods, number of packages, weight, measurements.
            * Freight terms (e.g., "Freight Prepaid", "Freight Collect").
            * Date "Shipped on Board" or dispatch date.
            * Terms and conditions of carriage, often on the reverse side.
    * **(Add similar detailed hints and differentiators for other specific document types you define)**
3.  **Primary Purpose Determination:** Based on the collective evidence from all pages and the indicators above, ascertain which single "Acceptable Document Type" most accurately represents the *overall primary purpose* of the document. Consider what action the document is intended to facilitate.
4.  **Confidence Assessment:** Assign a confidence score based on the clarity and preponderance of evidence. High confidence comes from explicit titles and a strong match of multiple key indicators. Lower confidence if the type is inferred or indicators are ambiguous or conflicting.
5.  **Output Format (Strict Adherence Required):**
    * Return ONLY a single, valid JSON object.
    * The JSON object must contain exactly three keys: `"classified_type"`, `"confidence"`, and `"reasoning"`.
    * `"classified_type"`: The determined document type string. This MUST be one of the "Acceptable Document Types". If, after thorough analysis, the document does not definitively match any acceptable type based on the provided indicators, use "UNKNOWN".
    * `"confidence"`: A numerical score between 0.0 (highly uncertain/unknown) and 1.0 (very certain).
    * `"reasoning"`: A concise but specific explanation for your classification. Reference key terms, field presence/absence, or structural elements observed across the document that led to your decision (e.g., "Document titled 'PROFORMA INVOICE' on page 1, contains seller/buyer details, itemized goods with prices, and payment terms consistent with a proforma invoice. Lacks 'Shipped on Board' date typical of a final commercial invoice post-shipment.").

**Example Output:**
\`\`\`json
{{
  "classified_type": "PROFORMA_INVOICE",
  "confidence": 0.98,
  "reasoning": "Document explicitly titled 'PROFORMA INVOICE' on page 1[cite: 4]. Contains all typical proforma invoice elements: seller (Transcendia, INC [cite: 1]), buyer (Arrow Business Advisory Pvt. Ltd [cite: 4]), detailed product description ('HA Laminating Film' [cite: 3]), quantity, unit price, total value[cite: 3], and payment terms[cite: 3]. It serves as a preliminary bill before shipment."
}}
\`\`\`
**Important:** Your response must be ONLY the valid JSON object. No greetings, apologies, or any text outside the JSON structure.
"""

EXTRACTION_PROMPT_TEMPLATE = """
**Your Role:** You are a highly meticulous and accurate AI Document Analysis Specialist. Your primary function is to extract structured data from business documents precisely according to instructions, with an extreme emphasis on the certainty of every character extracted.

**Task:** Analyze the provided {num_pages} pages, which together constitute a single logical '{doc_type}' document associated with Case ID '{case_id}'. Carefully extract the specific data fields listed below. Use the provided descriptions to understand the context and meaning of each field within this document type. Consider all pages to find the most relevant and accurate information for each field. Pay close attention to the nuanced instructions in each field's description to differentiate similar concepts and locate information that may not be explicitly labeled but can be inferred from context or common document structures.

**Fields to Extract (Name and Description):**
{field_list_str}

**Output Requirements (Strict):**

1.  **JSON Only:** You MUST return ONLY a single, valid JSON object as your response. Do NOT include any introductory text, explanations, summaries, apologies, or any other text outside of the JSON structure. The response must start directly with `{{` and end with `}}`.
2.  **JSON Structure:** The JSON object MUST have keys corresponding EXACTLY to the field **names** provided in the "Fields to Extract" list above.
3.  **Field Value Object:** Each value associated with a field key MUST be another JSON object containing the following three keys EXACTLY:
    * `"value"`: The extracted text value for the field.
        * If the field is clearly present, extract the value with absolute precision, ensuring every character is accurately represented.
        * If the field is **not found** or **not applicable** after thoroughly searching all pages and considering contextual clues as per the description, use the JSON value `null` (not the string "null").
        * If multiple potential values exist, select the one that is most pertinent to the specific context of the field's description. If ambiguity persists even after contextual evaluation, this must be reflected in a lower confidence score and explained in the reasoning.
        * For amounts, extract numerical values (e.g., "15000.75"). For dates, prefer a consistent format (e.g., YYYY-MM-DD or as it appears). Ensure no extraneous characters are included.

    * `"confidence"`: **Character-Informed Confidence Score (Strict)**
        * **Core Principle:** The overall confidence score (float, 0.00 to 1.00) for each field MUST reflect the system's certainty about **every single character** comprising the extracted value. The field's confidence is heavily influenced by the *lowest* confidence assigned to any of its critical constituent characters or segments during the OCR/interpretation process. A field cannot have high confidence if even one character is questionable.
        * **Calculation Basis:** This score integrates:
            * OCR engine's internal character-level confidence values (if available).
            * Visual clarity, print quality, and sharpness of each character in the source text segment.
            * Ambiguity checks for similar characters (e.g., '0' vs 'O', '1' vs 'l' vs 'I', '5' vs 'S', '8' vs 'B'). Each instance must be critically evaluated.
            * Legibility of handwriting (individual strokes, character formation, connections). Even if generally readable, individual poorly formed characters degrade confidence.
            * Strict adherence of every character to the expected field format and context (e.g., an alphabetic character 'O' in a purely numeric field like an account number *drastically* lowers confidence unless it's an accepted part of the format).
            * Cross-validation results where applicable (e.g., amount in words vs. numeric amount – discrepancies must lower confidence).
        * **Strict Benchmarks:**
            * **0.98 - 1.00 (Very High):** Absolute or near-absolute certainty. ALL characters are perfectly clear, sharp, unambiguous, flawlessly formed (print or ideal handwriting), and fully context-compliant. No plausible alternative interpretation exists for ANY character. This score implies that every character is deemed 100% recognizable.
            * **0.90 - 0.97 (High):** Strong confidence, but not absolute perfection for every character. All characters are clearly legible and contextually sound, but minor visual imperfections (e.g., slight pixelation, tiny ink spread that doesn't cause ambiguity) might exist for one or two characters, OR extremely low-probability alternative character interpretations were considered but definitively ruled out by strong contextual evidence.
            * **0.75 - 0.89 (Moderate):** Reasonable confidence, but with specific, identifiable uncertainties regarding one or more characters. This applies if:
                * One or two characters have moderate ambiguity that required contextual resolution (e.g., a printed '8' that is slightly broken making it look like a '3' until context confirms '8').
                * Minor OCR segmentation issues were overcome (e.g., slightly touching characters that were correctly separated but with effort).
                * Legible but somewhat challenging handwriting style for a character or two (e.g., a cursive 'e' that is not perfectly closed).
                * Slight fading or smudging on a few characters not critical to overall interpretation but preventing a "Very High" score.
            * **0.50 - 0.74 (Low):** Significant uncertainty exists regarding multiple characters or critical parts of the value. This applies if:
                * Several characters are ambiguous, poorly printed, or difficult to read.
                * Poor print quality (significant fading, widespread smudging, pixelation) affects key characters.
                * Irregular or barely legible handwriting is involved for a substantial portion of the value.
                * Contextual conflicts exist that raise doubts about the accuracy of certain characters (e.g., a date field showing '31-Feb-2023').
            * **< 0.50 (Very Low / Unreliable):** Extraction is highly speculative or impossible to perform reliably. The field value is likely incorrect, incomplete, or based on guesswork. Assign this if the text is largely illegible, completely missing, critical characters are indecipherable, or contextual validation fails insurmountably.
        * If the `"value"` is `null`, the `"confidence"` MUST be `0.0`.

    * `"reasoning"`: A concise explanation justifying the extracted `value` and `confidence` score.
        * Specify *how* the information was identified (e.g., "Directly beside explicit label 'Invoice No.'", "Inferred from the 'SHIP TO:' address block").
        * Indicate *where* the information was found (e.g., "on page 1, top right section", "page 5, under 'ACH & Wire Transfer Instructions'").
        * **Mandatory for any confidence score below 0.98 (previously 0.95, increased for stricter regime):** Briefly explain the *primary reason* for the reduced confidence, referencing specific character ambiguities (e.g., "Value 'INV-0012O'; Moderate (0.85): Final character resembles 'O' but context suggests '0'; slight blur."), handwriting issues, print quality ("Value '123 Main St'; High (0.92): Slight fading on 'St' but legible."), or contextual conflicts. If 0.98-1.00, reasoning can be "All characters perfectly clear and contextually valid."
        * If `"value"` is `null`, briefly explain *why* (e.g., "No explicit field label 'HS Code' or related tariff code information found on any page.").

**Example of Expected JSON Output Structure (Reflecting Stricter Confidence):**
(Note: The actual field names will match those provided in the 'Fields to Extract' list for the specific '{doc_type}')

\`\`\`json
{{
  "TYPE OF INVOICE - COMMERCIAL/PROFORMA/CUSTOMS/": {{
    "value": "PROFORMA INVOICE",
    "confidence": 1.00,
    "reasoning": "Extracted from explicit title 'PROFORMA INVOICE' on page 1[cite: 4]. All characters are perfectly clear, printed, and contextually valid."
  }},
  "INVOICE NO": {{
    "value": "2546049",
    "confidence": 0.99,
    "reasoning": "Extracted from the field labeled 'Reference #' on page 1, top section[cite: 2]. All digits are clearly printed and unambiguous. Confidence just below 1.00 due to general possibility of OCR misread on any character, though none observed."
  }},
  "BUYER NAME": {{
    "value": "Arrow Business Advisory Pvt. Ltd",
    "confidence": 0.98,
    "reasoning": "Extracted from 'BILL TO:' section on page 1[cite: 4]. All characters are clearly printed and well-defined. No ambiguities noted."
  }},
  "BUYER ADDRESS": {{
    "value": "159 Mittal Industrial Estate Sanjay Building No. 5/B Marol Naka, Andheri (East) Mumbai - 400 059 India",
    "confidence": 0.97,
    "reasoning": "Extracted from the 'BILL TO:' section on page 1[cite: 4]. All characters are clearly printed. Confidence slightly reduced from maximum due to the density of text and potential for any single character to have micro-imperfections not immediately obvious but considered under strict character policy."
  }},
  "BENEFICAIRY BANK SWIFT CODE / SORT CODE/ BSB / IFS CODE/ROUTING NO": {{
    "value": "CHASUS33",
    "confidence": 0.99,
    "reasoning": "Extracted from 'Swift Code: CHASUS33' under 'Wire Instructions' on page 5[cite: 29]. All characters are clearly printed and contextually valid as a SWIFT code."
  }},
  "PAYMENT TERMS": {{
    "value": "50% advance and balance 50% after 60 days from the date of Bill of Lading (BL)",
    "confidence": 0.96,
    "reasoning": "Extracted from 'Payment Terms:' section on page 1[cite: 3]. Text is clearly printed. Small font size and length of text slightly increase potential for overlooked character-level OCR nuances, hence not 0.98+."
  }},
  "HS CODE": {{
    "value": null,
    "confidence": 0.0,
    "reasoning": "No explicit field label 'HS Code' or related tariff code information found on any page of the provided proforma invoice."
  }},
  "Total Invoice Amount": {{
    "value": "135750.00",
    "confidence": 1.00,
    "reasoning": "Value '$135,750.00 USD' found as the total extension on page 1[cite: 3]. All digits, decimal point, and surrounding currency indicators are perfectly clear and printed. Numerical part extracted."
  }}
  // ... (all other requested fields for the '{doc_type}' document would follow this structure)
}}
\`\`\`
"""
