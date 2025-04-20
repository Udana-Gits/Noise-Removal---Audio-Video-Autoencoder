import React, { useState } from 'react';
import '../CSS/AutoencodeImages.css';

function AutoencodeImages() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleClean = () => {
    setSelectedImage(null);
    setPreviewUrl(null);
    // Reset file input
    const fileInput = document.getElementById('aei-image-input');
    if (fileInput) fileInput.value = '';
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
            onClick={handleClean}
          >
            Clean
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
        {previewUrl ? (
          <div className="aei-preview-container">
            <h2 className="aei-preview-title">Image Preview</h2>
            <img 
              src={previewUrl} 
              alt="Preview" 
              className="aei-image-preview" 
            />
          </div>
        ) : (
          <div className="aei-empty-state">
            <p>Select an image to preview</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default AutoencodeImages;