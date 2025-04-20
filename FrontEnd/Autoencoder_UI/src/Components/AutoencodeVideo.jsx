import React, { useState } from 'react';
import './CSS/AutoencodeVideo.css';

function AutoencodeVideo() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleVideoChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedVideo(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleClean = () => {
    setSelectedVideo(null);
    setPreviewUrl(null);
    // Reset file input
    const fileInput = document.getElementById('aev-video-input');
    if (fileInput) fileInput.value = '';
  };

  return (
    <div className="aev-split-container">
      <div className="aev-left-pane">
        <h1 className="aev-page-title">Autoencoder Video</h1>
        
        <div className="aev-input-container">
          <label htmlFor="aev-video-input" className="aev-input-label">
            Select Video
          </label>
          <input
            type="file"
            id="aev-video-input"
            className="aev-file-input"
            accept="video/*"
            onChange={handleVideoChange}
          />
          
          <button 
            className="aev-clean-button"
            onClick={handleClean}
          >
            Clean
          </button>
        </div>
        
        {selectedVideo && (
          <div className="aev-file-info">
            <p><strong>File Name:</strong> {selectedVideo.name}</p>
            <p><strong>File Size:</strong> {(selectedVideo.size / 1024).toFixed(2)} KB</p>
            <p><strong>File Type:</strong> {selectedVideo.type}</p>
          </div>
        )}
      </div>
      
      <div className="aev-right-pane">
        {previewUrl ? (
          <div className="aev-preview-container">
            <h2 className="aev-preview-title">Video Preview</h2>
            <video 
              controls
              src={previewUrl}
              className="aev-video-preview"
            />
          </div>
        ) : (
          <div className="aev-empty-state">
            <p>Select a video to preview</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default AutoencodeVideo;