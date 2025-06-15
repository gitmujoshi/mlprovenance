from flask import Flask, render_template, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import json
from datetime import datetime
import sys
import os

# Add the parent directory to the Python path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from safety_features.safety import SafetyChecker, SafetyConfig
from safety_features.config import AgeRating

app = Flask(__name__)

# Initialize model and tokenizer
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path))
    return model

def load_latest_model():
    # Find the latest model in the artifacts directory
    artifacts_dir = Path('artifacts/provenance')
    model_files = list(artifacts_dir.glob('**/trained_model.pt'))
    if not model_files:
        return None, None
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    model = load_model(latest_model)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer

# Initialize safety checker
safety_config = SafetyConfig(
    min_age_rating=AgeRating.TEEN,
    content_filters=["bad_word", "inappropriate"],
    max_input_length=512,
    max_output_length=100,
    block_sensitive_topics=True,
    require_content_warning=True
)
safety_checker = SafetyChecker(safety_config)

# Load model and tokenizer
model, tokenizer = load_latest_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if model is None or tokenizer is None:
        return jsonify({
            'error': 'No trained model found. Please train a model first.'
        }), 404

    data = request.get_json()
    prompt = data.get('prompt', '')
    
    # Safety check on input
    safety_result = safety_checker.check_text(prompt)
    if not safety_result['passes_checks']:
        return jsonify({
            'error': 'Input failed safety checks',
            'warnings': safety_result['warnings'],
            'violations': safety_result['violations']
        }), 400

    # Generate text
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Safety check on output
    output_safety = safety_checker.check_text(generated_text)
    
    return jsonify({
        'generated_text': generated_text,
        'safety_result': output_safety
    })

@app.route('/model-info')
def model_info():
    if model is None:
        return jsonify({
            'error': 'No trained model found'
        }), 404

    # Get model information
    model_path = Path('artifacts/provenance')
    model_files = list(model_path.glob('**/trained_model.pt'))
    if not model_files:
        return jsonify({
            'error': 'No model files found'
        }), 404

    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    model_dir = latest_model.parent

    # Try to load provenance report
    try:
        provenance_files = list(model_dir.glob('provenance_report_*.json'))
        if provenance_files:
            with open(max(provenance_files, key=lambda x: x.stat().st_mtime), 'r') as f:
                provenance = json.load(f)
        else:
            provenance = None
    except:
        provenance = None

    return jsonify({
        'model_path': str(latest_model),
        'last_modified': datetime.fromtimestamp(latest_model.stat().st_mtime).isoformat(),
        'provenance': provenance or {}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001) 