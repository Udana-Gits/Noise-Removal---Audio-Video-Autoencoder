import React, { useState, useEffect } from 'react';
import '../CSS/AutoencodeAudio.css';

function AutoencodeAudio() {
  const [selectedAudio, setSelectedAudio] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [noiseType, setNoiseType] = useState('');
  const [denoicedUrl, setDenoicedUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingError, setProcessingError] = useState(null);
  const [serverStatus, setServerStatus] = useState('checking'); // 'checking', 'online', 'offline'
  const [serverDetails, setServerDetails] = useState(null);

  // Check server status on component mount
  useEffect(() => {
    checkServerStatus();
  }, []);

  const checkServerStatus = async () => {
    try {
      setServerStatus('checking');
      
      // First check the root endpoint to get server details
      try {
        const rootResponse = await fetch('/');
        if (rootResponse.ok) {
          const data = await rootResponse.json();
          setServerDetails(data);
        }
      } catch (error) {
        console.log('Could not fetch server details');
      }
      
      // Now check the health endpoint
      const response = await fetch('/api/health');
      if (response.ok) {
        setServerStatus('online');
      } else {
        setServerStatus('offline');
        console.error('Server health check failed:', response.status);
      }
    } catch (error) {
      setServerStatus('offline');
      console.error('Cannot connect to server:', error);
    }
  };

  const handleAudioChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate file size (limit to 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setProcessingError("File size exceeds 10MB limit. Please select a smaller file.");
        return;
      }
      
      setSelectedAudio(file);
      setPreviewUrl(URL.createObjectURL(file));
      setDenoicedUrl(null); // Reset denoiced audio when new file selected
      setProcessingError(null);
    }
  };

  const handleNoiseTypeChange = (e) => {
    setNoiseType(e.target.value);
    setDenoicedUrl(null); // Reset denoiced audio when noise type changes
    setProcessingError(null);
  };

  const handleDenoice = async () => {
    if (serverStatus !== 'online') {
      await checkServerStatus();
      if (serverStatus !== 'online') {
        setProcessingError("Cannot connect to audio processing server. Please make sure the server is running.");
        return;
      }
    }

    if (!selectedAudio || !noiseType) {
      setProcessingError("Please select both an audio file and noise type.");
      return;
    }

    setIsProcessing(true);
    setProcessingError(null);

    const formData = new FormData();
    formData.append('audio', selectedAudio);
    formData.append('noise_type', noiseType);

    try {
      const response = await fetch('/api/denoice', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // Try to get detailed error message from response
        let errorMessage = `Server error: ${response.status}`;
        try {
          const errorData = await response.json();
          if (errorData && errorData.error) {
            errorMessage = errorData.error;
          }
        } catch (e) {
          // Couldn't parse JSON, use default error message
        }
        throw new Error(errorMessage);
      }

      // Check if response is a valid audio file
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('audio/')) {
        throw new Error('Server returned invalid audio format');
      }

      const blob = await response.blob();
      if (blob.size === 0) {
        throw new Error('Server returned empty audio file');
      }
      
      const url = URL.createObjectURL(blob);
      setDenoicedUrl(url);
    } catch (error) {
      console.error('Error processing audio:', error);
      setProcessingError(error.message || "Failed to process audio. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setSelectedAudio(null);
    setPreviewUrl(null);
    setDenoicedUrl(null);
    setNoiseType('');
    setProcessingError(null);
    // Reset file input
    const fileInput = document.getElementById('aea-audio-input');
    if (fileInput) fileInput.value = '';
  };

  const handleDownload = () => {
    if (!denoicedUrl) return;
    
    const link = document.createElement('a');
    link.href = denoicedUrl;
    link.download = `denoiced_${selectedAudio.name}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="aea-split-container">
      <div className="aea-left-pane">
        <h1 className="aea-page-title">Autoencoder Audio</h1>
        
        {serverStatus === 'checking' && (
          <div className="aea-server-status aea-checking">
            Checking server status...
          </div>
        )}
        
        {serverStatus === 'offline' && (
          <div className="aea-server-status aea-offline">
            <span>
              Server is offline. Make sure the Flask backend is running at http://localhost:5000.
            </span>
            <button className="aea-retry-button" onClick={checkServerStatus}>
              Retry Connection
            </button>
          </div>
        )}
        
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
            disabled={isProcessing || serverStatus === 'offline'}
          />
          
          <label htmlFor="aea-noise-type" className="aea-input-label">
            Noise Removal Type
          </label>
          <select
            id="aea-noise-type"
            className="aea-noise-select"
            value={noiseType}
            onChange={handleNoiseTypeChange}
            disabled={isProcessing || serverStatus === 'offline'}
          >
            <option value="">Select noise type to remove</option>
            <option value="white_noise">White noise</option>
            <option value="background">Background</option>
            <option value="chatter">Chatter</option>
            <option value="traffic_noise">Traffic noise</option>
            <option value="mechanical_noise">Mechanical noise</option>
            <option value="babble_noise">Babble noise</option>
            <option value="music_background">Music background</option>
          </select>
          
          <button 
            className="aea-denoice-button"
            onClick={handleDenoice}
            disabled={!selectedAudio || !noiseType || isProcessing || serverStatus === 'offline'}
          >
            {isProcessing ? 'Processing...' : 'Denoice'}
          </button>
          
          <button 
            className="aea-clean-button"
            onClick={handleReset}
            disabled={isProcessing}
          >
            Reset All
          </button>
        </div>
        
        {processingError && (
          <div className="aea-error-message">
            {processingError}
          </div>
        )}
        
        {selectedAudio && (
          <div className="aea-file-info">
            <p><strong>File Name:</strong> {selectedAudio.name}</p>
            <p><strong>File Size:</strong> {(selectedAudio.size / 1024).toFixed(2)} KB</p>
            <p><strong>File Type:</strong> {selectedAudio.type}</p>
            {noiseType && (
              <p><strong>Noise Removal:</strong> {noiseType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</p>
            )}
          </div>
        )}
        
        {serverDetails && serverStatus === 'online' && (
          <div className="aea-advanced-info">
            <details>
              <summary>Server Details</summary>
              <div className="aea-server-details">
                <p><strong>Upload Folder:</strong> {serverDetails.temp_directories?.upload_folder || 'N/A'}</p>
                <p><strong>Upload Writable:</strong> {serverDetails.temp_directories?.upload_writable ? '✅' : '❌'}</p>
                <p><strong>Output Folder:</strong> {serverDetails.temp_directories?.output_folder || 'N/A'}</p>
                <p><strong>Output Writable:</strong> {serverDetails.temp_directories?.output_writable ? '✅' : '❌'}</p>
              </div>
            </details>
          </div>
        )}
      </div>
      
      <div className="aea-right-pane">
        {isProcessing ? (
          <div className="aea-processing">
            <div className="aea-spinner"></div>
            <p>Processing your audio...</p>
          </div>
        ) : (
          <div className="aea-preview-container">
            {previewUrl && (
              <>
                <h2 className="aea-preview-title">Audio Preview</h2>
                <audio 
                  controls
                  src={previewUrl}
                  className="aea-audio-preview"
                />
              </>
            )}
            
            {denoicedUrl && (
              <>
                <h2 className="aea-preview-title">Denoiced Audio</h2>
                <audio 
                  controls
                  src={denoicedUrl}
                  className="aea-audio-preview"
                />
                <button 
                  className="aea-download-button"
                  onClick={handleDownload}
                >
                  Download Denoiced Audio
                </button>
              </>
            )}
            
            {!previewUrl && !denoicedUrl && (
              <div className="aea-empty-state">
                <p>Select an audio file to preview</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default AutoencodeAudio;