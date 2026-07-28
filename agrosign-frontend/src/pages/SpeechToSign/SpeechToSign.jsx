import "./SpeechToSign.scss";
import { useState, useEffect } from "react";
import {
  FiMic,
  FiStopCircle,
  FiUpload,
  FiArrowRight,
  FiGlobe,
} from "react-icons/fi";

function SpeechToSign() {
  const [language, setLanguage] = useState("en");

  const [isRecording, setIsRecording] = useState(false);

  const [seconds, setSeconds] = useState(0);

  const [audioFile, setAudioFile] = useState(null);

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

    // Backend recording logic later
  };

  const stopRecording = () => {
    setIsRecording(false);

    // Stop MediaRecorder later
  };

  const handleFile = (e) => {
    setAudioFile(e.target.files[0]);
  };

  const convertSpeech = () => {
    console.log({
      language,
      audioFile,
      recordingTime: seconds,
    });

    // Call FastAPI later
  };

  return (
    <section className="speech">

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
  <label>Or Upload Audio File</label>

  <input
    type="file"
    accept=".mp3,.wav,.m4a"
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

     

        <button
          className="convert"
          onClick={convertSpeech}
        >

          Convert to Sign

          <FiArrowRight />

        </button>

      </div>

    </section>
  );
}

export default SpeechToSign;