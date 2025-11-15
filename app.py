import os
import json
import logging
import hmac
import hashlib
from functools import wraps
from flask import Flask, request, jsonify
from flask import abort
import tempfile
from flask_mail import Mail, Message
from dotenv import load_dotenv
import requests
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load .env file only if this is NOT on Render
if os.getenv("RENDER") is None:
    load_dotenv()

app = Flask(__name__)

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', True)
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)

# Load credentials from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# --- LLM Configuration with Fallback ---

# The prompt is now a global constant
PROMPT = """You are an AI assistant that receives WhatsApp webhook messages and MUST OUTPUT a single valid JSON object only (no markdown, no text).

Required JSON format always (exact keys, never change or add keys):

{
  "phone": "",
  "customer_name": "",
  "message_text": "",
  "product": "",
  "specification": "",
  "quantity": "",
  "rate": "",
  "hsn_code": "",
  "email": ""
}

Rules:
1. Always output valid JSON only. Nothing else.
2. If you can extract a field, fill it. If not, the field value MUST be an empty string "".
3. Do NOT add any extra fields or metadata.
4. Phone field should contain digits only (strip +, spaces, or punctuation). If phone cannot be extracted from message text, use any phone in the webhook metadata. If still unavailable, return "".
5. If parsing fails or input is ambiguous, return the JSON with all fields set to "" (no errors or prose).
6. No surrounding text, no explanation, no code fences — raw JSON only.

Example output (if found):
{"phone":"919009000396","customer_name":"Rudra","message_text":"Send quote 110 ...","product":"5inch SS 316L sheets","specification":"5inch","quantity":"5psc","rate":"25000","hsn_code":"7219","email":"arnavbhandari1777@gmail.com"}

If unable to extract, respond:
{"phone":"","customer_name":"","message_text":"","product":"","specification":"","quantity":"","rate":"","hsn_code":"","email":""}
"""

# Candidate models in priority order.
CANDIDATE_MODELS = [
    "gemini-1.5-flash", # Using the newer, faster models first
    "gemini-pro",
]

# Safe fallback JSON to ensure parsed_data is never None.
SAFE_EMPTY_JSON = {
  "phone": "", "customer_name": "", "message_text": "", "product": "",
  "specification": "", "quantity": "", "rate": "", "hsn_code": "", "email": ""
}

# --- End LLM Configuration ---


def send_whatsapp_message(to_number, message_text):
    """Sends a text message to a WhatsApp number."""
    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_text},
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        logging.info(f"Message sent to {to_number}, status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending message to {to_number}: {e}")

def send_whatsapp_document(to_number, document_url, filename):
    """Sends a document to a WhatsApp number from a URL."""
    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "document",
        "document": {"link": document_url, "filename": filename},
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logging.info(f"Document sent to {to_number}, status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending document to {to_number}: {e}")

def call_gemini_to_parse(user_message: str) -> dict:
    """
    Calls candidate LLM models with retry logic to parse a user message.
    It uses the raw requests library for maximum control.
    Returns a structured JSON object, falling back to SAFE_EMPTY_JSON on total failure.
    """
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    full_prompt = PROMPT + f"\n\nParse the following message:\n\"{user_message}\""

    for model in CANDIDATE_MODELS:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = { "contents": [{ "parts": [{ "text": full_prompt }] }] }
        
        logging.info(f"Attempting to call model: {model}")
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
        except Exception as e:
            logging.warning(f"Model call network error for {model}: {e}")
            continue

        if resp.status_code == 200:
            try:
                body = resp.json()
                # Extract text from the response structure
                text_out = body['candidates'][0]['content']['parts'][0]['text']
                cleaned_text = text_out.strip().replace("```json", "").replace("```", "").strip()
                
                parsed = json.loads(cleaned_text)
                required = set(SAFE_EMPTY_JSON.keys())
                if isinstance(parsed, dict) and required.issubset(set(parsed.keys())):
                    logging.info(f"Successfully parsed response from {model}")
                    return parsed
                else:
                    logging.warning(f"Parsed JSON missing required keys from model {model}")
                    continue
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logging.warning(f"Failed to parse model {model} output as JSON: {e}")
                logging.warning(f"Malformed string/body was: {resp.text[:500]}")
                continue

        elif resp.status_code == 404:
            logging.info(f"Model {model} not available (404) — trying next")
            continue
        else:
            logging.warning(f"Model {model} returned HTTP {resp.status_code}: {resp.text[:300]}")
            continue

    # All models failed -> return safe empty JSON to avoid "Parsed data is None".
    logging.error("All model attempts failed — returning SAFE_EMPTY_JSON")
    return SAFE_EMPTY_JSON

def generate_quotation_document(parsed_data, phone_number):
    """
    Generates a quotation PDF from parsed data, saves it to a temporary file,
    and returns a public URL and filename.
    """
    logging.info(f"Generating a real quotation document for: {parsed_data}")

    # 1. Mock database lookup for price
    mock_price_db = {
        "TMT bars": {"Fe 550D": 55000, "Fe 500": 52000},
        "steel coils": {"IS 2062": 48000, "HR": 47500},
        "sheets": {"CRCA": 60000, "GP": 62000}
    }
    product = parsed_data.get("product") or "N/A"
    grade = parsed_data.get("specification") or "N/A" # Using 'specification' as 'grade'
    price_per_ton = mock_price_db.get(product, {}).get(grade, 50000) # Default price

    # 2. Create HTML content for the PDF
    html_content = f"""
    <html>
        <!-- (HTML content for PDF generation remains the same) -->
        <head><style>
            body {{ font-family: sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            .header {{ display: flex; justify-content: space-between; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            .company-details {{ text-align: right; }}
            .quote-details {{ margin-top: 40px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style></head>
        <body>
            <div class="header">
                <h1>Quotation</h1>
                <div class="company-details">
                    <strong>Your Steel Company Inc.</strong><br>
                    123 Steel Road, Metalburg<br>
                    sales@yoursteelco.com
                </div>
            </div>
            <div class="quote-details">
                <strong>Quote For:</strong> {phone_number}<br>
                <strong>Date:</strong> {os.getenv('CURRENT_DATE', '2023-10-27')}
            </div>
            <table>
                <tr><th>Product</th><th>Grade</th><th>Quantity</th><th>Unit Price (per Ton)</th></tr>
                <tr><td>{product}</td><td>{grade}</td><td>{parsed_data.get("quantity") or "N/A"}</td><td>INR {price_per_ton}</td></tr>
            </table>
        </body>
    </html>
    """

    # 3. Generate PDF and save to a temporary file
    # Note: pdfkit is not included in your new code, so this will fail if not installed
    import pdfkit
    temp_pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdfkit.from_string(html_content, temp_pdf_file.name)
    logging.info(f"PDF generated and saved to temporary file: {temp_pdf_file.name}")

    # --- IMPORTANT ---
    # In a real application, you would now upload `temp_pdf_file.name` to a cloud storage
    # (like S3) and get a public URL. For now, we return a dummy URL.
    public_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    filename = f"Quotation_{parsed_data.get('product', 'details')}.pdf"
    return public_url, filename

def send_internal_email_alert(phone_number, message_text, parsed_data, error_message=None):
    """Sends an internal email alert to the admin about incoming requests or errors."""
    try:
        subject = f"New Quote Request from {phone_number}"
        
        if error_message:
            subject = f"ERROR: Quote Request Failed from {phone_number}"
            body = f"""
            An error occurred while processing a quote request.
            
            Phone Number: {phone_number}
            Original Message: {message_text}
            Error: {error_message}
            Parsed Data: {parsed_data}
            
            Please investigate and follow up with the customer.
            """
        else:
            body = f"""
            A new quote request has been received.
            
            Phone Number: {phone_number}
            Message: {message_text}
            
            Parsed Data:
            - Product: {parsed_data.get('product', 'N/A')}
            - Specification: {parsed_data.get('specification', 'N/A')}
            - Quantity: {parsed_data.get('quantity', 'N/A')}
            
            A quotation document has been sent to the customer.
            """
        
        msg = Message(
            subject=subject,
            recipients=[os.getenv('ADMIN_EMAIL')],
            body=body
        )
        
        mail.send(msg)
        logging.info(f"Internal alert email sent to admin for {phone_number}")
        
    except Exception as e:
        logging.error(f"Error sending internal email alert: {e}")

@app.route("/")
def index():
    return "Server is running."

@app.route("/webhook", methods=["GET", "POST"])
def whatsapp_webhook():
    """
    Handles webhook verification and incoming messages from WhatsApp.
    """
    if request.method == "GET":
        # WhatsApp webhook verification
        mode = request.args.get("hub.mode")
        token_from_facebook = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        logging.info(f"!!! DEBUG: Token from Facebook: '{token_from_facebook}'. My token from env: '{WHATSAPP_VERIFY_TOKEN}'")

        if mode == "subscribe" and token_from_facebook == WHATSAPP_VERIFY_TOKEN:
            logging.info("INFO: Webhook verified.")
            return challenge, 200
        else:
            logging.error("ERROR: Webhook verification failed.")
            return "Forbidden", 403

    elif request.method == "POST":
        # --- NEW: SIGNATURE VALIDATION ---
        signature = request.headers.get('X-Hub-Signature-256')
        if not signature:
            logging.error("ERROR: Signature header missing!")
            abort(403)

        signature = signature.split('=')[-1]

        expected_signature = hmac.new(
            bytes(WHATSAPP_APP_SECRET, 'latin-1'),
            request.data, # Use request.data (raw body), NOT request.json
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            logging.error("ERROR: Webhook signature mismatch!")
            abort(403)

        logging.info("INFO: Webhook signature VERIFIED.")
        # --- END OF SIGNATURE VALIDATION ---

        # Now that we are secure, we can read the JSON
        body = request.json

        try:
            message = body['entry'][0]['changes'][0]['value']['messages'][0]
            user_message_text = message['text']['body']
            user_phone_number = message['from']

            logging.info(f"--- Processing message from {user_phone_number}: '{user_message_text}' ---")

            parsed_data = call_gemini_to_parse(user_message_text)
            doc_url, doc_filename = generate_quotation_document(parsed_data, user_phone_number)
            send_whatsapp_message(user_phone_number, "Here is your quote!")
            send_whatsapp_document(user_phone_number, doc_url, doc_filename)
            send_internal_email_alert(user_phone_number, user_message_text, parsed_data)
            logging.info("--- Successfully processed request and sent quotation ---")

        except Exception as e:
            logging.error(f"Error in processing pipeline for {user_phone_number}: {e}", exc_info=True)
            error_message = "I'm sorry, I ran into an error processing your request. A human will be with you shortly."
            send_whatsapp_message(user_phone_number, error_message)
            send_internal_email_alert(user_phone_number, user_message_text, None, error_message=str(e))

        return 'OK', 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host='0.0.0.0', port=port)