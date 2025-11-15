import os
import json
import logging
import hmac
import hashlib
import subprocess
import datetime
from functools import wraps
from pathlib import Path
from flask import Flask, request, jsonify
from flask import abort
import tempfile
from flask_mail import Mail, Message
from dotenv import load_dotenv
import requests
import google.generativeai as genai
from docxtpl import DocxTemplate

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

# Template path (update if Template.docx is in a different location)
TEMPLATE_PATH = os.getenv("TEMPLATE_PATH", "Template.docx")


def _coerce_number(value):
    """Try to parse an int/float from string; return None if not numeric."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    s = str(value).replace(',', '').strip()
    try:
        if '.' in s:
            return float(s)
        return int(s)
    except Exception:
        return None


def _normalize_phone(phone):
    """Extract digits only from phone number."""
    if not phone:
        return ""
    return ''.join(ch for ch in str(phone) if ch.isdigit())


def generate_quote_pdf(parsed_data: dict, template_path: str = TEMPLATE_PATH) -> str:
    """
    Fill Template.docx and convert to PDF. Returns the PDF file path.

    parsed_data: dict with keys matching your JSON (phone, customer_name, product, quantity, rate, units, hsn_code, etc.)
    template_path: path to Template.docx
    """

    # Ensure parsed_data has defaults
    data = dict(SAFE_EMPTY_JSON)
    if parsed_data:
        data.update(parsed_data)

    # Normalize and compute fields
    phone = _normalize_phone(data.get("phone", ""))
    customer_name = data.get("customer_name", "") or ""
    product = data.get("product", "") or ""
    specification = data.get("specification", "") or ""
    message_text = data.get("message_text", "") or ""
    email = data.get("email", "") or ""
    hsn = data.get("hsn_code", "") or data.get("hsn", "") or ""
    units = data.get("units", "") or data.get("units_text", "") or ""
    
    # Quantity & rate - try coercing numeric and compute line total if possible
    qty_raw = data.get("quantity", "")
    rate_raw = data.get("rate", "")
    qty = _coerce_number(qty_raw)
    rate = _coerce_number(rate_raw)
    total_amount = ""
    
    if qty is not None and rate is not None:
        try:
            total_val = qty * rate
            # format without trailing decimals if integer
            total_amount = f"{int(total_val):,}" if float(total_val).is_integer() else f"{total_val:,.2f}"
        except Exception:
            total_amount = ""
    else:
        # If quantity/rate are strings like "5psc" or "25000 per pcs", try to extract digits
        try:
            qty_digits = ''.join(ch for ch in str(qty_raw) if ch.isdigit())
            rate_digits = ''.join(ch for ch in str(rate_raw) if ch.isdigit())
            if qty_digits and rate_digits:
                total_val = int(qty_digits) * int(rate_digits)
                total_amount = f"{total_val:,}"
        except Exception:
            total_amount = ""

    # Quotation number & date
    q_no = data.get("q_no") or f"{datetime.datetime.utcnow().strftime('%y%m%d')}-{phone[-4:] if phone else '0000'}"
    today_str = datetime.datetime.utcnow().strftime("%d-%b-%Y")

    # Build context mapping matching placeholders in Template.docx
    context = {
        "q_no": q_no,
        "date": today_str,
        "company_name": data.get("company_name", "") or "",
        "customer_name": customer_name,
        "product": product or specification,
        "quantity": str(data.get("quantity", "")),
        "rate": str(data.get("rate", "")),
        "units": units,
        "hsn": hsn,
        "total": total_amount,
        "email": email,
        "phone": phone,
        "message_text": message_text
    }

    # Load docx template and render
    try:
        tpl = DocxTemplate(template_path)
        tpl.render(context)
    except FileNotFoundError:
        logging.error(f"Template file not found at {template_path}")
        raise
    except Exception as e:
        logging.error(f"Error rendering template: {e}")
        raise

    # Save temporary docx and convert to pdf
    tmp_dir = tempfile.mkdtemp(prefix="quote_")
    tmp_docx = os.path.join(tmp_dir, f"quote_{q_no}.docx")
    tmp_pdf = os.path.join(tmp_dir, f"quote_{q_no}.pdf")
    
    try:
        tpl.save(tmp_docx)
        logging.info(f"Temporary DOCX saved to {tmp_docx}")
    except Exception as e:
        logging.error(f"Error saving temporary DOCX: {e}")
        raise

    # Convert using LibreOffice (recommended on Linux/Render)
    try:
        # soffice will produce PDF in the same tmp_dir
        subprocess.run([
            "soffice", "--headless", "--convert-to", "pdf", "--outdir", tmp_dir, tmp_docx
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        
        # LibreOffice names the file with .pdf extension same basename
        if os.path.exists(tmp_pdf):
            logging.info("PDF generated and saved to temporary file: %s", tmp_pdf)
            return tmp_pdf
        else:
            # If LibreOffice produced differently, search directory
            for f in os.listdir(tmp_dir):
                if f.lower().endswith(".pdf"):
                    return os.path.join(tmp_dir, f)
            raise RuntimeError("LibreOffice conversion succeeded but PDF not found.")
    except FileNotFoundError:
        logging.warning("LibreOffice (soffice) not found. Trying docx2pdf fallback (Windows).")
    except subprocess.CalledProcessError as e:
        logging.warning("LibreOffice conversion error: %s; stderr: %s", e, e.stderr.decode(errors='ignore'))
    except Exception as e:
        logging.warning("LibreOffice conversion failed: %s", e)

    # Fallback: docx2pdf (Windows)
    try:
        from docx2pdf import convert
        convert(tmp_docx, tmp_pdf)
        if os.path.exists(tmp_pdf):
            logging.info("PDF generated (docx2pdf) and saved to: %s", tmp_pdf)
            return tmp_pdf
    except Exception as e:
        logging.error("Fallback docx->pdf failed: %s", e)

    # Last resort: return docx path so caller can decide
    logging.error("Could not convert docx to pdf. Returning DOCX path.")
    return tmp_docx

# --- End PDF Generation ---




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
            
            # Generate PDF quotation from template
            pdf_path = generate_quote_pdf(parsed_data)
            filename = f"Quotation_{parsed_data.get('product', 'quote')}.pdf"

            # --- URGENT: The generated PDF must be uploaded to a public URL first ---
            # The line below uses a placeholder. In production, you would upload pdf_path
            # to a cloud service (e.g., S3, Cloudinary) and get a real public URL.
            public_doc_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf" # Placeholder
            # -------------------------------------------------------------------------

            send_whatsapp_message(user_phone_number, "Here is your quote!")
            send_whatsapp_document(user_phone_number, public_doc_url, filename)
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