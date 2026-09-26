



import "./SpeechToSign.scss";
import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  FiMic,
  FiStopCircle,
  FiUpload,
  FiArrowRight,
  FiGlobe,
  FiTrash2,
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

  const [liveHindi, setLiveHindi] = useState("");
  const [liveKannada, setLiveKannada] = useState("");
  const [liveEnglish, setLiveEnglish] = useState("");

  const [partialHindi, setPartialHindi] = useState("");
  const [partialKannada, setPartialKannada] = useState("");

  const socketRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const processorRef = useRef(null);
  const sourceRef = useRef(null);

  const hindiBoxRef = useRef(null);
  const kannadaBoxRef = useRef(null);
  const englishBoxRef = useRef(null);

  const LIVE_LANGUAGES = ["hindi", "english", "kannada"];

  useEffect(() => {
    let interval;

    if (isRecording) {
      interval = setInterval(() => {
        setSeconds((prev) => prev + 1);
      }, 1000);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isRecording]);

  useEffect(() => {
    return () => {
      stopLiveTranscription();
    };
  }, []);

  useEffect(() => {
    if (hindiBoxRef.current) {
      hindiBoxRef.current.scrollTop =
        hindiBoxRef.current.scrollHeight;
    }
  }, [liveHindi, partialHindi]);

  useEffect(() => {
    if (kannadaBoxRef.current) {
      kannadaBoxRef.current.scrollTop =
        kannadaBoxRef.current.scrollHeight;
    }
  }, [liveKannada, partialKannada]);

  useEffect(() => {
    if (englishBoxRef.current) {
      englishBoxRef.current.scrollTop =
        englishBoxRef.current.scrollHeight;
    }
  }, [liveEnglish]);

  const formatTime = () => {
    const mins = String(
      Math.floor(seconds / 60)
    ).padStart(2, "0");

    const secs = String(
      seconds % 60
    ).padStart(2, "0");

    return `${mins}:${secs}`;
  };

  const clearLiveResults = () => {
    setLiveHindi("");
    setLiveKannada("");
    setLiveEnglish("");
    setPartialHindi("");
    setPartialKannada("");
  };

  const startRecording = async () => {
    if (LIVE_LANGUAGES.includes(language)) {
      await startLiveTranscription(language);
      return;
    }

    alert(
      "Live recording isn't available for this language yet."
    );
  };

  const stopRecording = () => {
    if (LIVE_LANGUAGES.includes(language)) {
      stopLiveTranscription();
      return;
    }

    setIsRecording(false);
  };

  const handleFile = (e) => {
    setAudioFile(e.target.files[0]);
  };

  const getEnglishText = () => {
    if (
      resultLanguage === "hindi" ||
      resultLanguage === "kannada"
    ) {
      return translation;
    }

    if (resultLanguage === "english") {
      return transcript;
    }

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

      const response = await fetch(
        `${API_BASE}/speech-to-text`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(
          "Speech-to-text request failed"
        );
      }

      const data = await response.json();

      setTranscript(data.transcript || "");
      setTranslation(data.translation || "");
      setResultLanguage(data.language || "");

      console.log("Recorded speech result:", data);

    } catch (error) {
      console.error(error);
      alert("Failed to convert speech.");
    } finally {
      setLoading(false);
    }
  };

  const convertToSign = async () => {
    let englishText = "";

    if (language === "hindi") {
      englishText =
        liveEnglish || translation || "";
    } else if (language === "kannada") {
      englishText =
        liveEnglish || translation || "";
    } else if (language === "english") {
      englishText =
        liveEnglish || transcript || "";
    }

    if (!englishText) {
      alert("Please translate the speech first.");
      return;
    }

    setSignLoading(true);

    try {
      const response = await fetch(
        `${API_BASE}/translate`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: englishText,
            language: "english",
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          "Sign conversion request failed"
        );
      }

      const data = await response.json();

      navigate("/results", {
        state: data,
      });

    } catch (error) {
      console.error(error);
      alert("Failed to convert to sign.");
    } finally {
      setSignLoading(false);
    }
  };

  const floatTo16BitPCM = (input) => {
    const output = new Int16Array(
      input.length
    );

    for (let i = 0; i < input.length; i++) {
      const sample = Math.max(
        -1,
        Math.min(1, input[i])
      );

      output[i] =
        sample < 0
          ? sample * 0x8000
          : sample * 0x7fff;
    }

    return output;
  };

  const downsampleBuffer = (
    buffer,
    inputSampleRate,
    outputSampleRate
  ) => {
    if (
      outputSampleRate ===
      inputSampleRate
    ) {
      return buffer;
    }

    if (
      outputSampleRate >
      inputSampleRate
    ) {
      throw new Error(
        "Output sample rate must be lower than input sample rate."
      );
    }

    const ratio =
      inputSampleRate /
      outputSampleRate;

    const newLength =
      Math.round(
        buffer.length / ratio
      );

    const result =
      new Float32Array(newLength);

    let offsetResult = 0;
    let offsetBuffer = 0;

    while (
      offsetResult < result.length
    ) {
      const nextOffsetBuffer =
        Math.round(
          (offsetResult + 1) *
            ratio
        );

      let accum = 0;
      let count = 0;

      for (
        let i = offsetBuffer;
        i < nextOffsetBuffer &&
        i < buffer.length;
        i++
      ) {
        accum += buffer[i];
        count++;
      }

      result[offsetResult] =
        count > 0
          ? accum / count
          : 0;

      offsetResult++;
      offsetBuffer =
        nextOffsetBuffer;
    }

    return result;
  };

  const startLiveTranscription = async (
    lang
  ) => {
    if (socketRef.current) {
      return;
    }

    let endpoint = "";

    if (lang === "hindi") {
      endpoint = "/live-hindi";
    } else if (lang === "kannada") {
      endpoint = "/live-kannada";
    } else {
      endpoint = "/live-english";
    }

    try {
      clearLiveResults();

      setSeconds(0);

      const stream =
        await navigator.mediaDevices.getUserMedia(
          {
            audio: true,
          }
        );

      streamRef.current =
        stream;

      const socket =
        new WebSocket(
          `${API_BASE.replace(
            "http",
            "ws"
          )}${endpoint}`
        );

      socketRef.current =
        socket;

      socket.onopen = async () => {
        console.log(
          `Live ${lang} WebSocket connected`
        );

        setIsRecording(true);

        const audioContext =
          new AudioContext();

        audioContextRef.current =
          audioContext;

        const source =
          audioContext.createMediaStreamSource(
            stream
          );

        sourceRef.current =
          source;

        const processor =
          audioContext.createScriptProcessor(
            4096,
            1,
            1
          );

        processorRef.current =
          processor;

        processor.onaudioprocess =
          (event) => {
            if (
              !socketRef.current ||
              socketRef.current
                .readyState !==
                WebSocket.OPEN
            ) {
              return;
            }

            const input =
              event.inputBuffer.getChannelData(
                0
              );

            const downsampled =
              downsampleBuffer(
                input,
                audioContext.sampleRate,
                16000
              );

            const pcm =
              floatTo16BitPCM(
                downsampled
              );

            socket.send(
              pcm.buffer
            );
          };

        source.connect(
          processor
        );

        processor.connect(
          audioContext.destination
        );
      };

      socket.onmessage = (
        event
      ) => {
        try {
          const data =
            JSON.parse(
              event.data
            );

          console.log(
            "Backend:",
            data
          );

          if (
            data.type ===
            "final"
          ) {
            if (
              data.hindi
            ) {
              setLiveHindi(
                (prev) =>
                  prev
                    ? `${prev} ${data.hindi}`
                    : data.hindi
              );

              setPartialHindi("");
            }

            if (
              data.kannada
            ) {
              setLiveKannada(
                (prev) =>
                  prev
                    ? `${prev} ${data.kannada}`
                    : data.kannada
              );

              setPartialKannada("");
            }

            if (
              data.english
            ) {
              setLiveEnglish(
                (prev) =>
                  prev
                    ? `${prev} ${data.english}`
                    : data.english
              );
            }
          }

          if (
            data.type ===
            "partial"
          ) {
            if (
              lang ===
              "hindi"
            ) {
              setPartialHindi(
                data.hindi ||
                  ""
              );
            }

            if (
              lang ===
              "kannada"
            ) {
              setPartialKannada(
                data.kannada ||
                  ""
              );
            }
          }

          if (
            data.type ===
            "error"
          ) {
            console.error(
              data.error
            );
          }

        } catch (error) {
          console.error(
            "Message parsing error:",
            error
          );
        }
      };

      socket.onerror =
        (error) => {
          console.error(
            "WebSocket error:",
            error
          );

          alert(
            `Live ${lang} connection failed.`
          );

          stopLiveTranscription();
        };

      socket.onclose =
        () => {
          console.log(
            `Live ${lang} WebSocket closed`
          );

          setIsRecording(
            false
          );

          socketRef.current =
            null;
        };

    } catch (error) {
      console.error(
        "Microphone error:",
        error
      );

      alert(
        "Please allow microphone access."
      );

      setIsRecording(
        false
      );
    }
  };

  const stopLiveTranscription = () => {
    if (
      processorRef.current
    ) {
      processorRef.current.disconnect();

      processorRef.current =
        null;
    }

    if (
      sourceRef.current
    ) {
      sourceRef.current.disconnect();

      sourceRef.current =
        null;
    }

    if (
      audioContextRef.current
    ) {
      audioContextRef.current.close();

      audioContextRef.current =
        null;
    }

    if (
      streamRef.current
    ) {
      streamRef.current
        .getTracks()
        .forEach(
          (track) => {
            track.stop();
          }
        );

      streamRef.current =
        null;
    }

    if (
      socketRef.current
    ) {
      socketRef.current.close();

      socketRef.current =
        null;
    }

    setIsRecording(
      false
    );

    setPartialHindi("");
    setPartialKannada("");
  };

  const showLiveResults =
    LIVE_LANGUAGES.includes(
      language
    ) &&
    (
      liveHindi ||
      liveKannada ||
      liveEnglish ||
      partialHindi ||
      partialKannada ||
      isRecording
    );

  return (
    <section className="speech-to-sign">

      <div className="speech-card">

        <h2>
          Speech to Sign
        </h2>

        <div className="record-box">

          <div className="mic-circle">
            <FiMic />
          </div>

          <h3>
            {isRecording
              ? "Listening..."
              : "Click the button to start recording"}
          </h3>

          <p>
            {formatTime()}
          </p>

          <div className="record-buttons">

            <button
              className="start"
              onClick={
                startRecording
              }
              disabled={
                isRecording
              }
            >
              <FiMic />
              Start Recording
            </button>

            <button
              className="stop"
              onClick={
                stopRecording
              }
              disabled={
                !isRecording
              }
            >
              <FiStopCircle />
              Stop
            </button>

          </div>

        </div>

        <div className="upload">

          <label
            htmlFor="audioUpload"
          >
            <FiUpload />

            {audioFile
              ? audioFile.name
              : "Upload Audio or Video File"}
          </label>

          <input
            id="audioUpload"
            type="file"
            accept="audio/*,video/*"
            onChange={
              handleFile
            }
          />

        </div>

        <div className="language">

          <label>
            Select Language
          </label>

          <div className="select-box">

            <FiGlobe />

            <select
              value={language}
              onChange={(e) => {

                if (
                  isRecording &&
                  LIVE_LANGUAGES.includes(
                    language
                  )
                ) {
                  stopLiveTranscription();
                }

                setLanguage(
                  e.target.value
                );

                clearLiveResults();

                setTranscript("");
                setTranslation("");
                setResultLanguage("");

              }}
            >

              <option value="english">
                English
              </option>

              <option value="hindi">
                Hindi
              </option>

              <option value="kannada">
                Kannada
              </option>

            </select>

          </div>

        </div>

        <div className="action-buttons">

          <button
            className="translate"
            onClick={
              convertToSign
            }
            disabled={
              signLoading ||
              (
                LIVE_LANGUAGES.includes(
                  language
                )
                  ? !liveEnglish &&
                    !translation &&
                    !transcript
                  : !transcript
              )
            }
          >
            {signLoading
              ? "Converting..."
              : "Convert to Sign"}
          </button>

          <button
            className="convert"
            onClick={
              convertSpeech
            }
            disabled={
              loading
            }
          >
            {loading
              ? "Processing..."
              : "Translate"}

            <FiArrowRight />

          </button>

        </div>

        {showLiveResults && (

          <div className="live-results">

            <div className="live-results-header">

              <h3>
                Live Transcript
              </h3>

              {(
                liveHindi ||
                liveKannada ||
                liveEnglish
              ) && (

                <button
                  type="button"
                  className="clear-live"
                  onClick={
                    clearLiveResults
                  }
                  title="Clear live transcript"
                >
                  <FiTrash2 />
                  Clear
                </button>

              )}

            </div>

            {language ===
              "hindi" && (

              <div className="result">

                <h3>
                  Hindi (live)
                </h3>

                <div
                  className="live-transcript-box"
                  ref={
                    hindiBoxRef
                  }
                >

                  <p>

                    {liveHindi}

                    {partialHindi && (
                      <span className="partial-text">
                        {liveHindi
                          ? " "
                          : ""}
                        {partialHindi}
                      </span>
                    )}

                    {!liveHindi &&
                      !partialHindi && (
                        <span className="placeholder">
                          Listening for speech…
                        </span>
                      )}

                  </p>

                </div>

              </div>
            )}

            {language ===
              "hindi" &&
              liveEnglish && (

              <div className="result">

                <h3>
                  English (live translation)
                </h3>

                <div
                  className="live-transcript-box"
                  ref={
                    englishBoxRef
                  }
                >

                  <p>
                    {liveEnglish}
                  </p>

                </div>

              </div>
            )}

            {language ===
              "kannada" && (

              <div className="result">

                <h3>
                  Kannada (live)
                </h3>

                <div
                  className="live-transcript-box"
                  ref={
                    kannadaBoxRef
                  }
                >

                  <p>

                    {liveKannada}

                    {partialKannada && (
                      <span className="partial-text">
                        {liveKannada
                          ? " "
                          : ""}
                        {partialKannada}
                      </span>
                    )}

                    {!liveKannada &&
                      !partialKannada && (
                        <span className="placeholder">
                          Listening for speech…
                        </span>
                      )}

                  </p>

                </div>

              </div>
            )}

            {language ===
              "kannada" &&
              liveEnglish && (

              <div className="result">

                <h3>
                  English (live translation)
                </h3>

                <div
                  className="live-transcript-box"
                  ref={
                    englishBoxRef
                  }
                >

                  <p>
                    {liveEnglish}
                  </p>

                </div>

              </div>
            )}

            {language ===
              "english" && (

              <div className="result">

                <h3>
                  English (live)
                </h3>

                <div
                  className="live-transcript-box"
                  ref={
                    englishBoxRef
                  }
                >

                  <p>

                    {liveEnglish || (
                      <span className="placeholder">
                        Listening for speech…
                      </span>
                    )}

                  </p>

                </div>

              </div>
            )}

          </div>
        )}

        {resultLanguage ===
          "hindi" &&
          transcript && (

          <div className="result">

            <h3>
              Hindi Transcript
            </h3>

            <p>
              {transcript}
            </p>

          </div>
        )}

        {resultLanguage ===
          "hindi" &&
          translation && (

          <div className="result">

            <h3>
              English Translation
            </h3>

            <p>
              {translation}
            </p>

          </div>
        )}

        {resultLanguage ===
          "kannada" &&
          transcript && (

          <div className="result">

            <h3>
              Kannada Transcript
            </h3>

            <p>
              {transcript}
            </p>

          </div>
        )}

        {resultLanguage ===
          "kannada" &&
          translation && (

          <div className="result">

            <h3>
              English Translation
            </h3>

            <p>
              {translation}
            </p>

          </div>
        )}

        {resultLanguage ===
          "english" &&
          transcript && (

          <div className="result">

            <h3>
              English Text
            </h3>

            <p>
              {transcript}
            </p>

          </div>
        )}

      </div>

    </section>
  );
}

export default SpeechToSign;