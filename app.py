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
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

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

def call_gemini_to_parse(message_text):
    """Uses Gemini to parse the user's message into structured JSON."""
    try:
        model = gemini_model
        prompt = f"""
        You are an intelligent assistant for a steel company.
        Your task is to parse a user's request for a price quote into a structured JSON object.
        The user's message is: "{message_text}"

        Extract the following fields:
        - "product": The type of steel product (e.g., 'TMT bars', 'steel coils', 'sheets').
        - "quantity": The amount, including units (e.g., '10 tons', '500 kgs').
        - "grade": The grade or specification of the steel (e.g., 'Fe 550D', 'IS 2062').

        If a field is not mentioned, use a value of null.
        Return ONLY the JSON object, with no other text or formatting.
        """
        response = model.generate_content(prompt)

        # Clean up the response to ensure it's valid JSON
        cleaned_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        logging.info(f"Gemini raw response: {response.text}")
        logging.info(f"Cleaned JSON string: {cleaned_text}")

        parsed_json = json.loads(cleaned_text)
        return parsed_json

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from Gemini: {e}")
        logging.error(f"Malformed string: {cleaned_text}")
        return None
    except Exception as e:
        logging.error(f"An error occurred calling Gemini API: {e}")
        return None

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
    product = parsed_data.get("product", "N/A")
    grade = parsed_data.get("grade", "N/A")
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
                <tr><td>{product}</td><td>{grade}</td><td>{parsed_data.get("quantity", "N/A")}</td><td>INR {price_per_ton}</td></tr>
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
            - Quantity: {parsed_data.get('quantity', 'N/A')}
            - Grade: {parsed_data.get('grade', 'N/A')}
            
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
            if not parsed_data:
                raise ValueError("Failed to parse message with Gemini. Parsed data is None.")

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