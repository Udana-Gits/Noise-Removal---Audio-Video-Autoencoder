import React, { useState } from 'react';
import '../CSS/AutoencodeImages.css';

function AutoencodeImages() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [showPreviews, setShowPreviews] = useState(false);  // <-- New state

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setSelectedImage(file);
      setPreviewUrl(url);
      setShowPreviews(false); // <-- Reset previews until denoise clicked
    }
  };

  const handleDenoise = () => {
    if (selectedImage) {
      setShowPreviews(true);
    }
  };

  const downloadImage = (url, filename) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="aei-split-container">
      <div className="aei-left-pane">
        <h1 className="aei-page-title">Autoencoder Images</h1>

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
          >
            Denoise
          </button>
        </div>

        {selectedImage && (
          <div className="aei-file-info">
            <p><strong>File Name:</strong> {selectedImage.name}</p>
            <p><strong>File Size:</strong> {(selectedImage.size / 1024).toFixed(2)} KB</p>
            <p><strong>File Type:</strong> {selectedImage.type}</p>
          </div>
        )}
      </div>

      <div className="aei-right-pane">
        {showPreviews && previewUrl ? (
          <div className="aei-multiple-preview-container">
            <h2 className="aei-preview-title">Comparison: Original vs Denoised</h2>

            <div className="aei-preview-grid">

              {/* Original Noisy Image */}
              <div className="aei-preview-card">
                <h3 className="aei-small-title">Original Noisy</h3>
                <img src={previewUrl} alt="Original" className="aei-image-preview" />
                {/* No download button here */}
              </div>

              {/* Denoised Gaussian */}
              <div className="aei-preview-card">
                <h3 className="aei-small-title">Gaussian Denoised</h3>
                <img src={previewUrl} alt="Gaussian Denoised" className="aei-image-preview" />
                <button 
                  className="aei-download-button"
                  onClick={() => downloadImage(previewUrl, 'Gaussian_Denoised.png')}
                >
                  Download Gaussian
                </button>
              </div>

              {/* Denoised Speckle */}
              <div className="aei-preview-card">
                <h3 className="aei-small-title">Speckle Denoised</h3>
                <img src={previewUrl} alt="Speckle Denoised" className="aei-image-preview" />
                <button 
                  className="aei-download-button"
                  onClick={() => downloadImage(previewUrl, 'Speckle_Denoised.png')}
                >
                  Download Speckle
                </button>
              </div>

              {/* Denoised Salt-Pepper */}
              <div className="aei-preview-card">
                <h3 className="aei-small-title">Salt-Pepper Denoised</h3>
                <img src={previewUrl} alt="Salt Pepper Denoised" className="aei-image-preview" />
                <button 
                  className="aei-download-button"
                  onClick={() => downloadImage(previewUrl, 'SaltPepper_Denoised.png')}
                >
                  Download Salt-Pepper
                </button>
              </div>

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
