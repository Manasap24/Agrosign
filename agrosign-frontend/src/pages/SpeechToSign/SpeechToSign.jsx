

import "./SpeechToSign.scss";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  FiMic,
  FiStopCircle,
  FiUpload,
  FiArrowRight,
  FiGlobe,
} from "react-icons/fi";

const API_BASE = "http://127.0.0.1:8000";

function SpeechToSign() {
  const navigate = useNavigate();

  const [language, setLanguage] = useState("english");
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [audioFile, setAudioFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const [transcript, setTranscript] = useState("");
  const [translation, setTranslation] = useState("");
  const [resultLanguage, setResultLanguage] = useState("");

  const [signLoading, setSignLoading] = useState(false);

  useEffect(() => {
    let interval;

    if (isRecording) {
      interval = setInterval(() => {
        setSeconds((prev) => prev + 1);
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [isRecording]);

  const formatTime = () => {
    const mins = String(Math.floor(seconds / 60)).padStart(2, "0");
    const secs = String(seconds % 60).padStart(2, "0");
    return `${mins}:${secs}`;
  };

  const startRecording = () => {
    setIsRecording(true);
    setSeconds(0);
    // MediaRecorder logic later
  };

  const stopRecording = () => {
    setIsRecording(false);
  };

  const handleFile = (e) => {
    setAudioFile(e.target.files[0]);
  };

  // The English text handed off to the sign-video pipeline:
  // - Hindi selected   -> use the translated English text
  // - English selected -> use the transcript directly
  const getEnglishText = () => {
    if (resultLanguage === "hindi") return translation;
    if (resultLanguage === "english") return transcript;
    return "";
  };

  const convertSpeech = async () => {
    if (!audioFile) {
      alert("Please upload an audio or video file.");
      return;
    }

    setLoading(true);
    setTranscript("");
    setTranslation("");
    setResultLanguage("");

    try {
      const formData = new FormData();
      formData.append("file", audioFile);
      formData.append("language", language);

      const response = await fetch(`${API_BASE}/speech-to-text`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Speech-to-text request failed");
      }

      const data = await response.json();

      setTranscript(data.transcript || "");
      setTranslation(data.translation || "");
      setResultLanguage(data.language || "");

      console.log(data);
    } catch (error) {
      console.error(error);
      alert("Failed to convert speech.");
    } finally {
      setLoading(false);
    }
  };

  const convertToSign = async () => {
    const englishText = getEnglishText();

    if (!englishText) {
      alert("Please translate the speech first.");
      return;
    }

    setSignLoading(true);

    try {
      const response = await fetch(`${API_BASE}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: englishText,
          language: "english",
        }),
      });

      if (!response.ok) {
        throw new Error("Sign conversion request failed");
      }

      const data = await response.json();

      // Hand the full response off to the Results page via router state
      navigate("/results", { state: data });
    } catch (error) {
      console.error(error);
      alert("Failed to convert to sign.");
    } finally {
      setSignLoading(false);
    }
  };

  return (
    <section className="speech-to-sign">
      <div className="speech-card">
        <h2>Speech to Sign</h2>

        <div className="record-box">
          <div className="mic-circle">
            <FiMic />
          </div>

          <h3>
            {isRecording
              ? "Recording..."
              : "Click the button to start recording"}
          </h3>

          <p>{formatTime()}</p>

          <div className="record-buttons">
            <button
              className="start"
              onClick={startRecording}
              disabled={isRecording}
            >
              <FiMic />
              Start Recording
            </button>

            <button
              className="stop"
              onClick={stopRecording}
              disabled={!isRecording}
            >
              <FiStopCircle />
              Stop
            </button>
          </div>
        </div>

        <div className="upload">
          <label htmlFor="audioUpload">
            <FiUpload />
            {audioFile ? audioFile.name : "Upload Audio or Video File"}
          </label>

          <input
            id="audioUpload"
            type="file"
            accept="audio/*,video/*"
            onChange={handleFile}
          />
        </div>

        <div className="language">
          <label>Select Language</label>

          <div className="select-box">
            <FiGlobe />

            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
            >
              <option value="english">English</option>
              <option value="hindi">Hindi</option>
              <option value="kannada">Kannada</option>
            </select>
          </div>
        </div>

        <div className="action-buttons">
          <button
            className="translate"
            onClick={convertToSign}
            disabled={signLoading || !transcript}
          >
            {signLoading ? "Converting..." : "Convert to Sign"}
          </button>

          <button className="convert" onClick={convertSpeech} disabled={loading}>
            {loading ? "Processing..." : "Translate"}
            <FiArrowRight />
          </button>
        </div>

        {/* Hindi selected: show both Hindi transcript and English translation */}
        {resultLanguage === "hindi" && transcript && (
          <div className="result">
            <h3>Hindi Transcript</h3>
            <p>{transcript}</p>
          </div>
        )}

        {resultLanguage === "hindi" && translation && (
          <div className="result">
            <h3>English Translation</h3>
            <p>{translation}</p>
          </div>
        )}

        {/* English selected: show only the English text */}
        {resultLanguage === "english" && transcript && (
          <div className="result">
            <h3>English Text</h3>
            <p>{transcript}</p>
          </div>
        )}
      </div>
    </section>
  );
}

export default SpeechToSign;