from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import torch
from torch import nn
import tempfile
from demucs.pretrained import get_model
from demucs.audio import AudioFile, save_audio
import logging
import shutil
from pathlib import Path
import time
from threading import Lock

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
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Increased to 32MB max upload size
app.config['SUPPORTED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'flac'}

# Initialize model globally
model = None
model_lock = Lock()  # Lock for thread-safe model access

# Noise type mapping to Demucs stems
NOISE_TYPE_TO_STEM = {
    "white_noise": "noise",
    "background": "other",
    "chatter": "vocals",
    "traffic_noise": "other",
    "mechanical_noise": "other",
    "babble_noise": "vocals",
    "music_background": "music",
    "all_noise": "all"  # Special case to target all noise sources
}


def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['SUPPORTED_EXTENSIONS']


# Load Demucs model
def load_model_instance():
    global model
    try:
        with model_lock:
            if model is None:
                model = get_model("htdemucs")
                model.eval()
                # Force model to CPU mode
                model = model.cpu()
                logger.info("Demucs model loaded successfully on CPU")
    except Exception as e:
        logger.error(f"Failed to load Demucs model: {e}")
        raise


def get_stem_targets(noise_type):
    """Map noise type to stems that should be removed"""
    if noise_type == "all_noise":
        return ["noise", "other", "vocals"]  # Remove all noise sources

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


def separate_sources(audio_path, noise_type, strength=1.0):
    """
    Separate audio sources using Demucs and return the denoised audio
    by removing the noise sources with a variable strength parameter
    """
    global model

    # Check if model is loaded
    load_model_instance()

    # Verify the file exists before processing
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Input audio file not found at path: {audio_path}")

    # Process with model access lock to ensure thread safety
    with model_lock:
        try:
            # Load audio file
            wav = AudioFile(audio_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)

            # Convert to tensor and normalize
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()

            # Add batch dimension
            wav = wav.unsqueeze(0)

            # Separate sources
            with torch.no_grad():
                sources = model(wav)

            # Sources shape is [batch, source, channel, time]
            source_names = model.sources

            # Determine which sources to remove
            stems_to_remove = get_stem_targets(noise_type)

            # Apply variable strength removal
            mask = torch.ones_like(sources)
            for stem in stems_to_remove:
                if stem in source_names:
                    idx = source_names.index(stem)
                    mask[:, idx] = 1.0 - min(1.0, max(0.0, float(strength)))

            # Apply mask to keep only desired sources
            filtered_sources = sources * mask

            # Mix down to a single source (denoised audio)
            denoised = filtered_sources.sum(dim=1)

            # Re-normalize
            denoised = denoised * ref.std() + ref.mean()

            return denoised.squeeze(0).numpy()

        except Exception as e:
            logger.error(f"Error in audio processing: {str(e)}")
            raise


@app.route('/api/denoice', methods=['POST'])
def denoice_audio():
    start_time = time.time()
    input_path = None
    output_path = None

    try:
        # Check if audio file is provided
        if 'audio' not in request.files:
            logger.error("Missing audio file")
            return jsonify({'error': 'Missing audio file'}), 400

        audio_file = request.files['audio']

        # Get noise type with default fallback
        noise_type = request.form.get('noise_type', 'background')

        # Get optional strength parameter (0.0-1.0)
        try:
            strength = float(request.form.get('strength', 1.0))
            strength = max(0.0, min(1.0, strength))  # Clamp between 0 and 1
        except ValueError:
            strength = 1.0
            logger.warning("Invalid strength value, using default 1.0")

        # Validate file
        if audio_file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        if not allowed_file(audio_file.filename):
            logger.error(f"Unsupported file format: {audio_file.filename}")
            return jsonify({'error': 'Unsupported file format. Allowed formats: wav, mp3, ogg, flac'}), 400

        # Generate a unique filename
        filename = secure_filename(audio_file.filename)
        unique_id = str(uuid.uuid4())
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")

        # Make sure the directory exists
        os.makedirs(os.path.dirname(input_path), exist_ok=True)

        # Save uploaded file
        audio_file.save(input_path)

        # Verify file was saved successfully
        if not os.path.exists(input_path):
            logger.error(f"Failed to save file at {input_path}")
            return jsonify({'error': 'Failed to save uploaded file'}), 500

        logger.info(f"Processing audio file: {input_path} with noise type: {noise_type}, strength: {strength}")

        # Process the audio file
        denoised_audio = separate_sources(input_path, noise_type, strength)

        # Create output path
        output_path = os.path.join(OUTPUT_FOLDER, f"denoised_{unique_id}_{filename}")
        output_filename = f"denoised_{filename}"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save denoised audio
        save_audio(denoised_audio, output_path, model.samplerate)

        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"Audio processing completed in {processing_time:.2f} seconds")

        # Return the processed file
        return send_file(output_path,
                         mimetype='audio/wav',
                         as_attachment=True,
                         download_name=output_filename)

    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

    finally:
        # Clean up uploaded file
        try:
            if input_path and os.path.exists(input_path):
                os.remove(input_path)
                logger.info(f"Removed temporary input file: {input_path}")
        except Exception as e:
            logger.warning(f"Error removing temporary input file: {str(e)}")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint with model status"""
    global model

    # Check if temp directories are accessible
    upload_accessible = os.access(UPLOAD_FOLDER, os.W_OK)
    output_accessible = os.access(OUTPUT_FOLDER, os.W_OK)

    # Check model status
    model_loaded = model is not None

    return jsonify({
        'status': 'healthy' if upload_accessible and output_accessible and model_loaded else 'degraded',
        'model_loaded': model_loaded,
        'temp_directories': {
            'upload_folder': UPLOAD_FOLDER,
            'upload_writable': upload_accessible,
            'output_folder': OUTPUT_FOLDER,
            'output_writable': output_accessible
        },
        'supported_formats': list(app.config['SUPPORTED_EXTENSIONS']),
        'noise_types': list(NOISE_TYPE_TO_STEM.keys()),
    }), 200


# Automated cleanup of old temporary files (files older than 1 hour)
def cleanup_old_files():
    """Remove temporary files older than 1 hour"""
    try:
        current_time = time.time()
        one_hour_in_seconds = 3600

        # Clean up old upload files
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > one_hour_in_seconds:
                    os.remove(file_path)
                    logger.info(f"Removed old upload file: {file_path}")

        # Clean up old output files
        for filename in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > one_hour_in_seconds:
                    os.remove(file_path)
                    logger.info(f"Removed old output file: {file_path}")
    except Exception as e:
        logger.warning(f"Error cleaning up old files: {str(e)}")


# Clean up ALL temporary files at exit
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


@app.route('/', methods=['GET'])
def root():
    """Enhanced root endpoint with more system information"""
    # Try to load the model
    load_model_instance()

    # Get version information
    torch_version = torch.__version__

    # Get memory usage
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return jsonify({
        'message': 'Audio Denoiser API is running',
        'status': 'ready',
        'version': '1.1.0',
        'system_info': {
            'torch_version': torch_version,
            'memory_usage_mb': memory_info.rss / (1024 * 1024),
            'cpu_count': os.cpu_count()
        },
        'temp_directories': {
            'upload_folder': UPLOAD_FOLDER,
            'output_folder': OUTPUT_FOLDER
        },
        'supported_formats': list(app.config['SUPPORTED_EXTENSIONS']),
        'noise_types': list(NOISE_TYPE_TO_STEM.keys()),
    }), 200


# Schedule cleanup of old files periodically
@app.before_first_request
def setup_scheduled_tasks():
    from threading import Timer

    def run_cleanup():
        cleanup_old_files()
        # Schedule the next cleanup in 15 minutes
        Timer(15 * 60, run_cleanup).start()

    # Initial cleanup schedule
    Timer(15 * 60, run_cleanup).start()


if __name__ == '__main__':
    # Load model at startup
    load_model_instance()
    app.run(debug=False, host='0.0.0.0', port=5000)