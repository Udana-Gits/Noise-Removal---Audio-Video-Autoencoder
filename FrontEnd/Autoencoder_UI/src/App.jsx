import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './App.css'
import MainPage from './Components/MainPage'
import AutoencodeImages from './Components/AutoencodeImages'
import AutoencodeAudio from './Components/AutoencodeAudio'
import AutoencodeVideo from './Components/AutoencodeVideo'




function App() {
  

  return (
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<MainPage />} />
        <Route path='/autoencodeimages' element={<AutoencodeImages />} />
        <Route path='/autoencodeaudio' element={<AutoencodeAudio />} />
        <Route path='/autoencodevideo' element={<AutoencodeVideo />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
