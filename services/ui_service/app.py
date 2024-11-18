from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Create templates directory if it doesn't exist
os.makedirs('services/ui_service/templates', exist_ok=True)

# Basic routes
@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "service": "PhishDefender",
        "endpoints": {
            "health": "/health",
            "scan": "/scan",
            "results": "/results"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "services": {
            "kafka": "connected",
            "model": "running",
            "analytics": "running"
        }
    })

@app.route('/scan', methods=['POST'])
def scan_email():
    data = request.get_json()
    email = data.get('email', '')
    
    # Here you would typically:
    # 1. Send the email to Kafka
    # 2. Trigger the analysis process
    # 3. Return a job ID
    
    return jsonify({
        "status": "processing",
        "message": "Email submitted for scanning",
        "job_id": "sample_job_123"  # You should generate a real job ID
    })

@app.route('/results/<job_id>')
def get_results(job_id):
    # Here you would typically:
    # 1. Look up results for the given job_id
    # 2. Return the analysis results
    
    return jsonify({
        "job_id": job_id,
        "status": "completed",
        "results": {
            "is_phishing": False,
            "confidence": 0.95,
            "analysis_timestamp": "2024-03-19T12:00:00Z"
        }
    })

if __name__ == '__main__':
    # Changed port from 5000 to 8000
    app.run(host='0.0.0.0', port=8000, debug=True)