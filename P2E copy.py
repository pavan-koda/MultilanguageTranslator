from flask import Flask, render_template, request, jsonify
import argostranslate.package
import argostranslate.translate
import os

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

app = Flask(__name__)

# Global stores for IndicTrans2
indic_tokenizers = {}
indic_models = {}

# Initialize translation
def initialize_translator():
    """Download and install translation packages if not already installed"""
    print("Initializing Argos Translate for Polish...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    
    required_pairs = [("pl", "en"), ("en", "pl")]
    installed_packages = argostranslate.package.get_installed_packages()
    installed_pairs = [(pkg.from_code, pkg.to_code) for pkg in installed_packages]
    
    for from_code, to_code in required_pairs:
        if (from_code, to_code) not in installed_pairs:
            package_to_install = next(
                (pkg for pkg in available_packages if pkg.from_code == from_code and pkg.to_code == to_code), 
                None
            )
            if package_to_install:
                print(f"Downloading {from_code}-{to_code} translation package...")
                argostranslate.package.install_from_path(package_to_install.download())
                print(f"Package {from_code}-{to_code} installed successfully!")
            else:
                print(f"{from_code}-{to_code} package not found in available packages")
        else:
            print(f"{from_code}-{to_code} translation package already installed")

    # Initialize IndicTrans2 for Hindi
    if TRANSFORMERS_AVAILABLE:
        print("\nInitializing IndicTrans2 for Hindi (this may take some time)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        pairs = {
            "en-hi": "ai4bharat/indictrans2-en-indic-1B",
            "hi-en": "ai4bharat/indictrans2-indic-en-1B"
        }
        
        for direction, model_name in pairs.items():
            if direction not in indic_models:
                print(f"Loading {model_name} for {direction}...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
                    model = model.to(device)
                    model.eval()
                    
                    indic_tokenizers[direction] = tokenizer
                    indic_models[direction] = model
                    print(f"Successfully loaded {direction} model.")
                except Exception as e:
                    print(f"Failed to load {direction} model: {e}")
    else:
        print("\nTransformers or Torch not installed. Skipping IndicTrans2 Hindi setup.")
        print("To enable Hindi, run: pip install transformers sentencepiece torch indic-nlp-library")

def translate_text(text, from_lang="pl", to_lang="en"):
    """Translate text between specified languages"""
    direction = f"{from_lang}-{to_lang}"
    
    # Route Hindi through IndicTrans2
    if "hi" in direction:
        if not TRANSFORMERS_AVAILABLE or direction not in indic_models:
        if not TRANSFORMERS_AVAILABLE:
            return "Error: IndicTrans2 not loaded. Make sure transformers is installed."
        
        try:
            # Map the direction to the correct model name
            model_map = {
                "en-hi": "ai4bharat/indictrans2-en-indic-1B",
                "hi-en": "ai4bharat/indictrans2-indic-en-1B",
                "en-hi-200m": "ai4bharat/indictrans2-en-indic-dist-200M",
                "hi-en-200m": "ai4bharat/indictrans2-indic-en-dist-200M"
            }
            
            model_name = model_map.get(direction)
            if not model_name:
                return f"Error: Model not mapped for direction {direction}"
                
            if direction not in indic_models:
                print(f"Lazy loading {model_name} for {direction}... This will take a moment.")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
                model = model.to(device)
                model.eval()
                indic_tokenizers[direction] = tokenizer
                indic_models[direction] = model

            tokenizer = indic_tokenizers[direction]
            model = indic_models[direction]
            device = model.device
            
            src_lang = "eng_Latn" if from_lang == "en" else "hin_Deva"
            tgt_lang = "hin_Deva" if to_lang == "hi" else "eng_Latn"
            base_from = from_lang.split('-')[0]
            base_to = to_lang.split('-')[0]
            src_lang = "eng_Latn" if base_from == "en" else "hin_Deva"
            tgt_lang = "hin_Deva" if base_to == "hi" else "eng_Latn"
            
            try:
                inputs = tokenizer(text, src_lang=src_lang, tgt_lang=tgt_lang, return_tensors="pt", padding=True, truncation=True)
            except TypeError:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                
            inputs = inputs.to(device)
            
            with torch.no_grad():
                if hasattr(tokenizer, "lang_code_to_id") and tgt_lang in tokenizer.lang_code_to_id:
                    outputs = model.generate(**inputs, max_new_tokens=256, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
                else:
                    outputs = model.generate(**inputs, max_new_tokens=256)
                    
            translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return translated_text
            
        except Exception as e:
            return f"IndicTrans2 Translation error: {str(e)}"

    # Route Polish (and others) through Argos Translate
    try:
        # Get installed languages
        installed_languages = argostranslate.translate.get_installed_languages()
        
        # Find source and target languages
        from_language = next((lang for lang in installed_languages if lang.code == from_lang), None)
        to_language = next((lang for lang in installed_languages if lang.code == to_lang), None)
        
        if not from_language or not to_language:
            return f"Error: Language pair not found. Please ensure the {from_lang}-{to_lang} package is installed."
        
        # Get translation
        translation = from_language.get_translation(to_language)
        if not translation:
            return "Error: Translation not available"
        
        return translation.translate(text)
    except Exception as e:
        return f"Translation error: {str(e)}"

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """Handle translation requests"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        direction = data.get('direction', 'pl-en')  # e.g., pl-en, en-pl, hi-en, en-hi
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if '-' in direction:
            from_lang, to_lang = direction.split('-', 1)
        else:
            from_lang, to_lang = 'pl', 'en'
            
        result = translate_text(text, from_lang=from_lang, to_lang=to_lang)
        
        return jsonify({'translation': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize translator on startup
    print("Initializing translator...")
    initialize_translator()
    
    # Run the Flask app
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

    