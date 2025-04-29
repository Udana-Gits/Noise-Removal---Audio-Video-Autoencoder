from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import torch
import tempfile
from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio
import numpy as np
import logging
import shutil
from pathlib import Path

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create absolute paths for temp directories
base_temp_dir = os.path.join(tempfile.gettempdir(), "audio_denoiser")
UPLOAD_FOLDER = os.path.join(base_temp_dir, "uploads")
OUTPUT_FOLDER = os.path.join(base_temp_dir, "outputs")

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Log the actual paths for debugging
logger.info(f"Upload folder path: {UPLOAD_FOLDER}")
logger.info(f"Output folder path: {OUTPUT_FOLDER}")

# Configure Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize model globally
model = None


# Load Demucs model
def load_model_instance():
    global model
    try:
        model = get_model("htdemucs")
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        logger.info("Demucs model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Demucs model: {e}")
        raise


# Load model at startup
with app.app_context():
    load_model_instance()

# Noise type mapping to Demucs stems
NOISE_TYPE_TO_STEM = {
    "white_noise": "noise",
    "background": "other",
    "chatter": "vocals",
    "traffic_noise": "other",
    "mechanical_noise": "other",
    "babble_noise": "vocals",
    "music_background": "music"
}


def get_stem_targets(noise_type):
    """Map noise type to stems that should be removed"""
    if noise_type not in NOISE_TYPE_TO_STEM:
        return ["noise", "other"]  # Default targets

    stem = NOISE_TYPE_TO_STEM[noise_type]
    if stem == "noise":
        return ["noise"]
    elif stem == "other":
        return ["other"]
    elif stem == "vocals":
        return ["vocals"]
    elif stem == "music":
        return ["music"]
    else:
        return ["noise", "other"]  # Default fallback


def separate_sources(audio_path, noise_type):
    """
    Separate audio sources using Demucs and return the denoised audio
    by removing the noise sources
    """
    global model

    # Check if model is loaded
    if model is None:
        load_model_instance()

    # Verify the file exists before processing
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Input audio file not found at path: {audio_path}")

    # Load audio file
    wav = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)

    # Convert to tensor and normalize
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()

    # Add batch dimension
    wav = wav.unsqueeze(0)

    # Move to GPU if available
    if torch.cuda.is_available():
        wav = wav.cuda()

    # Separate sources
    with torch.no_grad():
        sources = model(wav)

    # Sources shape is [batch, source, channel, time]
    if torch.cuda.is_available():
        sources = sources.cpu()

    # Get all source names from model
    source_names = model.sources

    # Determine which sources to remove
    stems_to_remove = get_stem_targets(noise_type)

    # Create a mask for sources to keep (1) and remove (0)
    mask = torch.ones_like(sources)
    for stem in stems_to_remove:
        if stem in source_names:
            idx = source_names.index(stem)
            mask[:, idx] = 0

    # Apply mask to keep only desired sources
    filtered_sources = sources * mask

    # Mix down to a single source (denoised audio)
    denoised = filtered_sources.sum(dim=1)

    # Re-normalize
    denoised = denoised * ref.std() + ref.mean()

    return denoised.squeeze(0).numpy()


@app.route('/api/denoice', methods=['POST'])
def denoice_audio():
    input_path = None
    output_path = None

    try:
        # Check if audio file and noise type are provided
        if 'audio' not in request.files or 'noise_type' not in request.form:
            logger.error("Missing audio file or noise type")
            return jsonify({'error': 'Missing audio file or noise type'}), 400

        audio_file = request.files['audio']
        noise_type = request.form['noise_type']

        # Validate file
        if audio_file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        # Generate a unique filename
        filename = secure_filename(audio_file.filename)
        unique_id = str(uuid.uuid4())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")

        # Log full path for debugging
        logger.info(f"Saving audio file to: {input_path}")

        # Make sure the directory exists
        os.makedirs(os.path.dirname(input_path), exist_ok=True)

        # Save uploaded file
        audio_file.save(input_path)

        # Verify file was saved successfully
        if not os.path.exists(input_path):
            logger.error(f"Failed to save file at {input_path}")
            return jsonify({'error': 'Failed to save uploaded file'}), 500

        logger.info(f"Saved audio file: {input_path}")

        # Process the audio file
        denoised_audio = separate_sources(input_path, noise_type)

        # Create output path
        output_path = os.path.join(OUTPUT_FOLDER, f"denoised_{unique_id}_{filename}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Log full output path for debugging
        logger.info(f"Saving denoised audio to: {output_path}")

        # Save denoised audio
        save_audio(denoised_audio, output_path, model.samplerate)

        # Verify output file was created
        if not os.path.exists(output_path):
            logger.error(f"Failed to save denoised audio at {output_path}")
            return jsonify({'error': 'Failed to save denoised audio'}), 500

        logger.info(f"Saved denoised audio: {output_path}")

        # Return the processed file
        return send_file(output_path,
                         mimetype='audio/wav',
                         as_attachment=True,
                         download_name=f"denoised_{filename}")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded file
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
                logger.info(f"Removed temporary input file: {input_path}")
        except Exception as e:
            logger.warning(f"Error removing temporary input file: {str(e)}")

        # Keep the output file until it's downloaded, then clean it up
        # We'll rely on periodic temp folder cleanup instead


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


# Clean up any leftover temporary files
def cleanup_temp_files():
    try:
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
            logger.info(f"Cleaned up upload folder: {UPLOAD_FOLDER}")
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)
            logger.info(f"Cleaned up output folder: {OUTPUT_FOLDER}")
    except Exception as e:
        logger.warning(f"Error cleaning up temp files: {e}")


# Register cleanup function
import atexit

atexit.register(cleanup_temp_files)


# Add CORS headers to allow cross-origin requests
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response


# Handle OPTIONS preflight requests
@app.route('/api/denoice', methods=['OPTIONS'])
def handle_options():
    return '', 200


# Add a root endpoint for testing
@app.route('/', methods=['GET'])
def root():
    # Check if temp directories are accessible
    upload_accessible = os.access(UPLOAD_FOLDER, os.W_OK)
    output_accessible = os.access(OUTPUT_FOLDER, os.W_OK)

    return jsonify({
        'message': 'Audio Denoiser API is running',
        'temp_directories': {
            'upload_folder': UPLOAD_FOLDER,
            'upload_writable': upload_accessible,
            'output_folder': OUTPUT_FOLDER,
            'output_writable': output_accessible
        }
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)