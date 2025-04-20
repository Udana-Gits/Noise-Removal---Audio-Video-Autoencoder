import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import '../CSS/mainPage.css'
import leftbackground from '../Images/t7.png'


function MainPage() {
  const navigate = useNavigate()

  function autoencodeimages(){
    navigate('/autoencodeimages')
  }
  function autoencodeaudio(){
    navigate('/autoencodeaudio')
  }
  function autoencodevideo(){
    navigate('/autoencodevideo')
  }
  

  return (
    <>
      <div className="background-img-mainpage">
        <div className="ams-split-container-mainpage">
          <div className="ams-left-pane-mainpage">
            <img src={leftbackground} alt="autoencoder System Background" />
          </div>
          <div className="ams-right-pane-mainpage">
            <div className="ams-button-container-mainpage">
              <button className="ams-action-button-mainpage" onClick={autoencodeimages}>Auto-Encode Images</button>
              <button className="ams-action-button-mainpage" onClick={autoencodeaudio}>Auto-Encode Audio</button>
              <button className="ams-action-button-mainpage" onClick={autoencodevideo}>Auto-Encode Video</button>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default MainPage