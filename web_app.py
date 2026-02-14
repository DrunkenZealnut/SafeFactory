#!/usr/bin/env python3
"""
Pinecone Agent Web Interface
Flask-based web UI for Pinecone vector database operations with RAG support.
"""

# SSL Certificate configuration (must be done before importing httpx/urllib3)
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import unicodedata
from flask import Flask, render_template, send_from_directory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24).hex())

# Import shared configuration
from services.domain_config import DOCUMENTS_PATH, DOMAIN_CONFIG

# Register API blueprints and CORS
from api import init_api
init_api(app)


# ========================================
# HTML Page Routes
# ========================================

@app.route('/')
def home():
    """Home page with domain selection."""
    return render_template('home.html')


@app.route('/semiconductor')
def semiconductor():
    """Semiconductor domain page."""
    config = DOMAIN_CONFIG['semiconductor']
    return render_template('domain.html', domain='semiconductor', config=config)


@app.route('/laborlaw')
def laborlaw():
    """Labor law domain page."""
    config = DOMAIN_CONFIG['laborlaw']
    return render_template('domain.html', domain='laborlaw', config=config)


@app.route('/field-training')
def field_training():
    """Field training safety domain page."""
    config = DOMAIN_CONFIG['field-training']
    return render_template('domain.html', domain='field-training', config=config)


@app.route('/msds')
def msds():
    """MSDS chemical information page."""
    config = DOMAIN_CONFIG['msds']
    return render_template('msds.html', domain='msds', config=config)


@app.route('/documents/<path:filepath>')
def serve_document(filepath):
    """Serve document images and files."""
    # Normalize unicode for macOS (NFD filesystem) compatibility
    filepath = unicodedata.normalize('NFC', filepath)
    full_path = DOCUMENTS_PATH / filepath
    if not full_path.exists():
        filepath_nfd = unicodedata.normalize('NFD', filepath)
        full_path_nfd = DOCUMENTS_PATH / filepath_nfd
        if full_path_nfd.exists():
            filepath = filepath_nfd
    return send_from_directory(DOCUMENTS_PATH, filepath)


if __name__ == '__main__':
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        exit(1)
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY not set")
        exit(1)

    print("🚀 Pinecone Agent Web Interface")
    print("=" * 40)
    print(f"Index: {os.getenv('PINECONE_INDEX_NAME', 'document-index')}")
    print("=" * 40)
    print("\n🌐 http://localhost:5001 에서 접속하세요\n")

    app.run(debug=True, host='0.0.0.0', port=5001)
