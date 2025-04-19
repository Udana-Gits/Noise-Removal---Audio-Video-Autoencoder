import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

function AutoencodeAudio() {
  const navigate = useNavigate()

 

  return (
    <>
      <div className="background-img-AutoencodeAudio">
        <div className="ams-split-container-AutoencodeAudio">
          <div className="ams-left-pane-AutoencodeAudio">
          </div>
          <div className="ams-right-pane-AutoencodeAudio">
            <div className="ams-button-container-AutoencodeAudio">
              <button className="ams-action-button-AutoencodeAudio" >Take Attendance</button>
              <button className="ams-action-button-AutoencodeAudio" >Register Students</button>
              <button className="ams-action-button-AutoencodeAudio" >Remove Students</button>
              <button className="ams-action-button-AutoencodeAudio" >Analyze Attendance</button>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default AutoencodeAudio