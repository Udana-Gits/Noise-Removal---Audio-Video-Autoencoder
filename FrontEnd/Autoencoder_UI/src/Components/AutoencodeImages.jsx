import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

function AutoencodeImages() {
  const navigate = useNavigate()

 

  return (
    <>
      <div className="background-img-AutoencodeImages">
        <div className="ams-split-container-AutoencodeImages">
          <div className="ams-left-pane-AutoencodeImages">
          </div>
          <div className="ams-right-pane-AutoencodeImages">
            <div className="ams-button-container-AutoencodeImages">
              <button className="ams-action-button-AutoencodeImages" >Take Attendance</button>
              <button className="ams-action-button-AutoencodeImages" >Register Students</button>
              <button className="ams-action-button-AutoencodeImages" >Remove Students</button>
              <button className="ams-action-button-AutoencodeImages" >Analyze Attendance</button>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default AutoencodeImages