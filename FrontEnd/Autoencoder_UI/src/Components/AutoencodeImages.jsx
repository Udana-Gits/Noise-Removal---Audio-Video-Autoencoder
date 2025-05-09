import React, { useState } from 'react';
import '../CSS/AutoencodeImages.css';

function AutoencodeImages() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [denoisedImages, setDenoisedImages] = useState(null);
  const [error, setError] = useState(null);

  // API endpoint - change this to match your backend server address
  const API_URL = 'http://localhost:5000';

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setSelectedImage(file);
      setPreviewUrl(url);
      setDenoisedImages(null); // Reset denoised images when a new image is selected
      setError(null); // Reset any errors
    }
  };

  const handleDenoise = async () => {
    if (!selectedImage) {
      setError("Please select an image first");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Create form data
      const formData = new FormData();
      formData.append('image', selectedImage);

      // Make API request
      const response = await fetch(`${API_URL}/denoise`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setDenoisedImages(data);
    } catch (err) {
      console.error("Error denoising image:", err);
      setError(`Failed to denoise image: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadImage = (base64Data, filename) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${base64Data}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="aei-split-container">
      <div className="aei-left-pane">
        <h1 className="aei-page-title">Images Autoencoder</h1>

        <div className="aei-input-container">
          <label htmlFor="aei-image-input" className="aei-input-label">
            Select Image
          </label>
          <input
            type="file"
            id="aei-image-input"
            className="aei-file-input"
            accept="image/*"
            onChange={handleImageChange}
          />

          <button 
            className="aei-clean-button"
            onClick={handleDenoise}
            disabled={isLoading || !selectedImage}
          >
            {isLoading ? 'Processing...' : 'Denoise'}
          </button>
        </div>

        {/* Original Noisy Image */}
        {previewUrl && (
          <div className="aei-file-info">
            <h3 className="aei-small-title">Original Noisy</h3>
            <img src={previewUrl} alt="Original" className="aei-image-preview" />
          </div>
        )}

        {/* Error message */}
        {error && <div className="aei-error-message">{error}</div>}
      </div>

      <div className="aei-right-pane">
        {isLoading ? (
          <div className="aei-loading">
            <p>Processing image... This may take a few moments.</p>
          </div>
        ) : denoisedImages ? (
          <div className="aei-multiple-preview-container">
            <h2 className="aei-preview-title">Comparison: Original vs Denoised</h2>

            <div className="aei-preview-grid">
              {/* Denoised Gaussian */}
              {denoisedImages.gaussian && (
                <div className="aei-preview-card">
                  <h3 className="aei-small-title">Gaussian Denoised</h3>
                  <img 
                    src={`data:image/png;base64,${denoisedImages.gaussian.base64}`} 
                    alt="Gaussian Denoised" 
                    className="aei-image-preview" 
                  />
                  <button 
                    className="aei-download-button"
                    onClick={() => downloadImage(denoisedImages.gaussian.base64, 'Gaussian_Denoised.png')}
                  >
                    Download
                  </button>
                </div>
              )}

              {/* Denoised Speckle */}
              {denoisedImages.speckle && (
                <div className="aei-preview-card">
                  <h3 className="aei-small-title">Speckle Denoised</h3>
                  <img 
                    src={`data:image/png;base64,${denoisedImages.speckle.base64}`} 
                    alt="Speckle Denoised" 
                    className="aei-image-preview" 
                  />
                  <button 
                    className="aei-download-button"
                    onClick={() => downloadImage(denoisedImages.speckle.base64, 'Speckle_Denoised.png')}
                  >
                    Download
                  </button>
                </div>
              )}

              {/* Denoised Salt-Pepper */}
              {denoisedImages.salt_pepper && (
                <div className="aei-preview-card">
                  <h3 className="aei-small-title">Salt-Pepper Denoised</h3>
                  <img 
                    src={`data:image/png;base64,${denoisedImages.salt_pepper.base64}`} 
                    alt="Salt Pepper Denoised" 
                    className="aei-image-preview" 
                  />
                  <button 
                    className="aei-download-button"
                    onClick={() => downloadImage(denoisedImages.salt_pepper.base64, 'SaltPepper_Denoised.png')}
                  >
                    Download
                  </button>
                </div>
              )}

              {/* RealESRGAN Denoised */}
              {denoisedImages.realesrgan && (
                <div className="aei-preview-card">
                  <h3 className="aei-small-title">RealESRGAN Denoised</h3>
                  <img 
                    src={`data:image/png;base64,${denoisedImages.realesrgan.base64}`} 
                    alt="RealESRGAN Denoised" 
                    className="aei-image-preview" 
                  />
                  <button 
                    className="aei-download-button"
                    onClick={() => downloadImage(denoisedImages.realesrgan.base64, 'RealESRGAN_Denoised.png')}
                  >
                    Download
                  </button>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="aei-empty-state">
            <p>Select an image and click Denoise</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default AutoencodeImages;