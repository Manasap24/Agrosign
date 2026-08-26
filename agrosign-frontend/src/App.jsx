import { Routes, Route } from "react-router-dom";
import { useState } from "react";
import Navbar from "./components/Navbar/Navbar";
import Home from "./pages/Home/Home";
import TextToSign from "./pages/TextToSign/TextToSign";
import SpeechToSign from "./pages/SpeechToSign/SpeechToSign";
import PdfToSign from "./pages/PDFtoSign/PDFtoSign";
import Results from "./pages/Results/Results";
import "./App.scss";

function App() {
  const [darkMode, setDarkMode] = useState(
    localStorage.getItem("darkMode") === "true",
  );

  const toggleDarkMode = () => {
    const newMode = !darkMode;
    setDarkMode(newMode);
    localStorage.setItem("darkMode", newMode);
  };

  return (
    <div className={darkMode ? "app dark-mode" : "app"}>
      <Navbar darkMode={darkMode} toggleDarkMode={toggleDarkMode} />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/text" element={<TextToSign />} />
        <Route path="/speech" element={<SpeechToSign />} />
        <Route path="/pdf" element={<PdfToSign />} />
        <Route path="/results" element={<Results />} />
      </Routes>
    </div>
  );
}

export default App;
