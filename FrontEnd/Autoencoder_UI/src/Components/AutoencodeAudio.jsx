import React, { useState } from 'react';
import '../CSS/AutoencodeAudio.css';

function AutoencodeAudio() {
  const [selectedAudio, setSelectedAudio] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleAudioChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedAudio(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleClean = () => {
    setSelectedAudio(null);
    setPreviewUrl(null);
    // Reset file input
    const fileInput = document.getElementById('aea-audio-input');
    if (fileInput) fileInput.value = '';
  };

  return (
    <div className="aea-split-container">
      <div className="aea-left-pane">
        <h1 className="aea-page-title">Autoencoder Audio</h1>
        
        <div className="aea-input-container">
          <label htmlFor="aea-audio-input" className="aea-input-label">
            Select Audio
          </label>
          <input
            type="file"
            id="aea-audio-input"
            className="aea-file-input"
            accept="audio/*"
            onChange={handleAudioChange}
          />
          
          <button 
            className="aea-clean-button"
            onClick={handleClean}
          >
            Clean
          </button>
        </div>
        
        {selectedAudio && (
          <div className="aea-file-info">
            <p><strong>File Name:</strong> {selectedAudio.name}</p>
            <p><strong>File Size:</strong> {(selectedAudio.size / 1024).toFixed(2)} KB</p>
            <p><strong>File Type:</strong> {selectedAudio.type}</p>
          </div>
        )}
      </div>
      
      <div className="aea-right-pane">
        {previewUrl ? (
          <div className="aea-preview-container">
            <h2 className="aea-preview-title">Audio Preview</h2>
            <audio 
              controls
              src={previewUrl}
              className="aea-audio-preview"
            />
          </div>
        ) : (
          <div className="aea-empty-state">
            <p>Select an audio file to preview</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default AutoencodeAudio;