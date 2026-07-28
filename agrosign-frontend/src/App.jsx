import Navbar from "./components/Navbar/Navbar";
import { Routes, Route } from "react-router-dom";
import "./App.scss";
// import Home from "./pages/Home";
// import About from "./pages/About";
// import Contact from "./pages/Contact";
import TextToSign from "./pages/TextToSign/TextToSign";
import PDFToSign from "./pages/PDFToSign/PDFToSign"; 
import Results from "./pages/Results/Results";
import SpeechToSign from "./pages/SpeechToSign/SpeechToSign";
function App() {
  return (
    <>
      <Navbar />

      <div style={{ paddingTop: "90px" }}>
        <Routes>
           <Route path="/text" element={<TextToSign />} />
             <Route path="/speech" element={<SpeechToSign />} />
   <Route path="/pdf" element={<PDFToSign />} />
   <Route path="/results" element={<Results />} />

          {/* <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} /> */}
        </Routes>
      </div>
    </>
  );
}

export default App;