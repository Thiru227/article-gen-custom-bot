import os
import io
import base64
import requests
from flask import Flask, request, jsonify
from PIL import Image
import PyPDF2
import json
from datetime import datetime

app = Flask(__name__)

# Configuration - Set these as environment variables
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
WEBHOOK_URL = os.environ.get('WEBHOOK_URL')  # Your render.com URL

# AI Provider Keys - Add multiple keys for each provider
AI_PROVIDERS = {
    'gemini': {
        'keys': os.environ.get('GEMINI_KEYS', '').split(','),
        'endpoint': 'https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent',  # v1 instead of v1beta
        'active': True
    },
    'openrouter': {
        'keys': os.environ.get('OPENROUTER_KEYS', '').split(','),
        'endpoint': 'https://openrouter.ai/api/v1/chat/completions',
        'model': 'google/gemini-flash-1.5',
        'active': True
    },
    'groq': {
        'keys': os.environ.get('GROQ_KEYS', '').split(','),
        'endpoint': 'https://api.groq.com/openai/v1/chat/completions',
        'model': 'llama-3.2-90b-text-preview',  # Using text model with OCR approach
        'active': False  # Disabled since Groq deprecated vision models
    }
}

# Track failed keys
failed_keys = {provider: set() for provider in AI_PROVIDERS}

def send_telegram_message(chat_id, text, parse_mode='Markdown'):
    """Send message via Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': parse_mode
    }
    try:
        response = requests.post(url, json=data, timeout=10)
        return response.json()
    except Exception as e:
        print(f"Error sending message: {e}")
        return None

def download_file(file_id):
    """Download file from Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile"
    response = requests.get(url, params={'file_id': file_id})
    file_path = response.json()['result']['file_path']
    
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    file_response = requests.get(file_url)
    return file_response.content

def pdf_to_images(pdf_bytes):
    """Convert PDF first page to base64 image"""
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
        if images:
            img_byte_arr = io.BytesIO()
            images[0].save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            return base64.b64encode(img_byte_arr.read()).decode('utf-8')
    except Exception as e:
        print(f"PDF conversion error: {e}")
    return None

def image_to_base64(image_bytes):
    """Convert image to base64"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Resize if too large
        if img.width > 2000 or img.height > 2000:
            img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=90)
        img_byte_arr.seek(0)
        return base64.b64encode(img_byte_arr.read()).decode('utf-8')
    except Exception as e:
        print(f"Image processing error: {e}")
        return None

def call_gemini(base64_image, api_key):
    """Call Gemini API with v1 endpoint"""
    url = f"{AI_PROVIDERS['gemini']['endpoint']}?key={api_key}"
    
    prompt = """Analyze this event poster and extract all information in a well-formatted article.

Structure your response as:
# [Event Title]

**Location & Date:** [Extract location and dates]

[Write 2-3 paragraphs describing the event, its purpose, and significance]

## Event Details:
- **Date:** [Full dates]
- **Time:** [Time range if available]
- **Location:** [Venue details]

## Organizers and Resource Persons:
- **Organizer:** [Department/Institution]
- **Resource Person(s):** [Names and designations]
- **Convenor:** [Name and designation]
- **Co-ordinators:** [Names and designations]

Extract ALL text accurately. If information is missing, omit that section. Keep the tone professional and engaging."""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2048
        }
    }
    
    response = requests.post(url, json=payload, timeout=30)
    if response.status_code == 200:
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    else:
        print(f"Gemini response: {response.text}")
        raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

def call_openrouter(base64_image, api_key):
    """Call OpenRouter API"""
    url = AI_PROVIDERS['openrouter']['endpoint']
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://your-app.com',  # Optional but recommended
        'X-Title': 'Event Poster Bot'  # Optional but recommended
    }
    
    prompt = """Analyze this event poster and extract all information in a well-formatted article.

Structure your response as:
# [Event Title]

**Location & Date:** [Extract location and dates]

[Write 2-3 paragraphs describing the event, its purpose, and significance]

## Event Details:
- **Date:** [Full dates]
- **Time:** [Time range if available]
- **Location:** [Venue details]

## Organizers and Resource Persons:
- **Organizer:** [Department/Institution]
- **Resource Person(s):** [Names and designations]
- **Convenor:** [Name and designation]
- **Co-ordinators:** [Names and designations]

Extract ALL text accurately. If information is missing, omit that section. Keep the tone professional and engaging."""

    payload = {
        "model": AI_PROVIDERS['openrouter']['model'],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.4,
        "max_tokens": 2048
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=45)
    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        print(f"OpenRouter response: {response.text}")
        raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

def call_groq(base64_image, api_key):
    """Call Groq API - DEPRECATED, keeping for future use"""
    # Groq has deprecated all vision models as of Nov 2024
    # This function is kept for reference but not used
    raise Exception("Groq vision models are deprecated")

def extract_event_info(base64_image):
    """Try multiple AI providers with multiple keys"""
    providers_order = ['gemini', 'openrouter']  # Removed groq due to deprecation
    
    for provider_name in providers_order:
        provider = AI_PROVIDERS[provider_name]
        if not provider['active']:
            print(f"Skipping {provider_name} - disabled")
            continue
            
        for api_key in provider['keys']:
            if not api_key or api_key.strip() == '' or api_key in failed_keys[provider_name]:
                continue
                
            try:
                print(f"Trying {provider_name} with key ending in ...{api_key[-4:]}")
                
                if provider_name == 'gemini':
                    result = call_gemini(base64_image, api_key)
                elif provider_name == 'openrouter':
                    result = call_openrouter(base64_image, api_key)
                
                print(f"✓ Success with {provider_name}")
                return result
                
            except Exception as e:
                print(f"✗ Failed with {provider_name}: {e}")
                failed_keys[provider_name].add(api_key)
                continue
    
    raise Exception("❌ All AI providers failed. Please verify:\n1. API keys are valid\n2. You have credits/quota remaining\n3. Keys are properly set in environment variables")

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming Telegram updates"""
    try:
        update = request.get_json()
        
        if 'message' not in update:
            return jsonify({'ok': True})
        
        message = update['message']
        chat_id = message['chat']['id']
        
        # Handle /start command
        if 'text' in message and message['text'] == '/start':
            welcome_text = """👋 *Welcome to Event Poster to Article Bot!*

Send me an event poster (image or PDF) and I'll convert it into a well-formatted article.

*How to use:*
• Send a clear photo of the event poster
• Or send a PDF file
• Wait a few seconds for processing

*Powered by:* Google Gemini AI

Just send the image/PDF to get started! 🚀"""
            send_telegram_message(chat_id, welcome_text)
            return jsonify({'ok': True})
        
        # Handle /help command
        if 'text' in message and message['text'] == '/help':
            help_text = """*Event Poster Bot Help* 📚

*How it works:*
1️⃣ Send a clear photo of an event poster
2️⃣ Or upload a PDF file containing the poster
3️⃣ The bot extracts all information and formats it as an article

*Tips for best results:*
• Use clear, well-lit photos
• Ensure all text is readable
• Avoid blurry or distorted images
• PDF should be under 10MB

*Supported formats:*
✓ JPEG/PNG images
✓ PDF documents (first page only)

*Issues?* Make sure your image is clear and text is visible."""
            send_telegram_message(chat_id, help_text)
            return jsonify({'ok': True})
        
        base64_image = None
        
        # Handle images
        if 'photo' in message:
            send_telegram_message(chat_id, "📸 Processing your image... Please wait.")
            file_id = message['photo'][-1]['file_id']  # Get highest resolution
            image_bytes = download_file(file_id)
            base64_image = image_to_base64(image_bytes)
            
        # Handle documents (PDFs)
        elif 'document' in message:
            doc = message['document']
            if doc.get('mime_type') == 'application/pdf':
                send_telegram_message(chat_id, "📄 Processing your PDF... This may take a moment.")
                file_id = doc['file_id']
                pdf_bytes = download_file(file_id)
                base64_image = pdf_to_images(pdf_bytes)
                
                if not base64_image:
                    send_telegram_message(chat_id, "❌ Failed to convert PDF. Please try:\n• Sending it as an image instead\n• Making sure PDF is not corrupted\n• Using a clearer PDF")
                    return jsonify({'ok': True})
            else:
                send_telegram_message(chat_id, "⚠️ Please send an *image* (JPEG/PNG) or *PDF* file.")
                return jsonify({'ok': True})
        else:
            send_telegram_message(chat_id, "⚠️ Please send an event poster:\n• As an image (photo)\n• As a PDF document\n\nUse /help for more info.")
            return jsonify({'ok': True})
        
        # Extract event information
        if base64_image:
            try:
                article = extract_event_info(base64_image)
                send_telegram_message(chat_id, article)
                send_telegram_message(chat_id, "✅ *Article generated successfully!*\n\nNeed another? Just send another poster!")
            except Exception as e:
                error_msg = f"❌ *Error processing image*\n\n{str(e)}\n\nPlease try:\n• Using a clearer image\n• Checking if text is readable\n• Verifying your API quota"
                send_telegram_message(chat_id, error_msg)
                print(f"Processing error: {e}")
        
        return jsonify({'ok': True})
        
    except Exception as e:
        print(f"Webhook error: {e}")
        return jsonify({'ok': True})

@app.route('/set_webhook', methods=['GET'])
def set_webhook():
    """Set Telegram webhook"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
    webhook_url = f"{WEBHOOK_URL}/webhook"
    response = requests.post(url, json={'url': webhook_url})
    return jsonify(response.json())

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring and cron jobs"""
    active_providers = {k: v['active'] for k, v in AI_PROVIDERS.items() if v['active']}
    key_counts = {k: len([key for key in v['keys'] if key and key.strip()]) 
                  for k, v in AI_PROVIDERS.items() if v['active']}
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'providers': active_providers,
        'valid_keys': key_counts
    }), 200

@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint to keep app awake"""
    return jsonify({'status': 'ok', 'time': datetime.utcnow().isoformat()}), 200

@app.route('/keepalive', methods=['GET'])
def keepalive():
    """Keep-alive endpoint for cron jobs (prevents Render sleep)"""
    return "OK", 200

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    active_providers = [k for k, v in AI_PROVIDERS.items() if v['active']]
    
    return f"""
    <h1>🤖 Event Poster to Article Bot</h1>
    <p><strong>Status:</strong> ✅ Running</p>
    <p><strong>Active Providers:</strong> {', '.join(active_providers)}</p>
    
    <h3>Quick Links:</h3>
    <ul>
        <li><a href="/health">Health Check</a> - View system status</li>
        <li><a href="/set_webhook">Set Webhook</a> - Configure Telegram webhook</li>
    </ul>
    
    <h3>How to Use:</h3>
    <ol>
        <li>Open Telegram and find your bot</li>
        <li>Send /start to begin</li>
        <li>Send an event poster image or PDF</li>
        <li>Receive formatted article instantly!</li>
    </ol>
    
    <p><em>Powered by Google Gemini AI</em></p>
    """

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
