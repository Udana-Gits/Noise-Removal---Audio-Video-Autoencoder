import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

function AutoencodeVideo() {
  const navigate = useNavigate()

 

  return (
    <>
      <div className="background-img-AutoencodeVideo">
        <div className="ams-split-container-AutoencodeVideo">
          <div className="ams-left-pane-AutoencodeVideo">
          </div>
          <div className="ams-right-pane-AutoencodeVideo">
            <div className="ams-button-container-AutoencodeVideo">
              <button className="ams-action-button-AutoencodeVideo" >Take Attendance</button>
              <button className="ams-action-button-AutoencodeVideo" >Register Students</button>
              <button className="ams-action-button-AutoencodeVideo" >Remove Students</button>
              <button className="ams-action-button-AutoencodeVideo" >Analyze Attendance</button>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

export default AutoencodeVideo