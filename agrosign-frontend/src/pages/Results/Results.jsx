import "./Results.scss";
import { useState, useRef } from "react";
import { useLocation } from "react-router-dom";
import { FiGlobe, FiActivity, FiPlay, FiDownload } from "react-icons/fi";

function Results() {
  const location = useLocation();
  const videoRef = useRef(null);

  const data = location.state;

  if (!data) {
    return (
      <div style={{ padding: "40px", textAlign: "center" }}>
        <h2>No Translation Found</h2>
        <p>Please convert some text first.</p>
      </div>
    );
  }

  // Original Text
  const originalText = data.translations.map((item) => item.sentence).join(" ");

  // Detected Language
  const language = "English";

  // Main Detected Process
  const process = data.translations[0]?.process_name || "Unknown";

  // All detected processes received from backend
  const processList = data.process_sequence || [];

  // Video URLs received from backend
  const videoList = data.complete_video_sequence || [];

  // Playlist
  // Display keyword from video filename
  const playlist = videoList.map((video) => {
    const filename = video.split("/").pop().split("\\").pop();

    const keyword = filename.replace(/\.[^/.]+$/, "");

    return {
      keyword: keyword,
      video: video,
    };
  });

  const [currentIndex, setCurrentIndex] = useState(0);

  const [currentVideo, setCurrentVideo] = useState(
    playlist.length > 0 ? playlist[0].video : "",
  );

  const [autoPlay, setAutoPlay] = useState(true);

  // Play selected video
  const playVideo = (index) => {
    setCurrentIndex(index);
    setCurrentVideo(playlist[index].video);
  };

  // Play playlist from beginning
  const playAll = () => {
    if (playlist.length > 0) {
      playVideo(0);
    }
  };

  // Play next video
  const playNext = () => {
    if (!autoPlay) return;

    if (currentIndex < playlist.length - 1) {
      const next = currentIndex + 1;

      setCurrentIndex(next);
      setCurrentVideo(playlist[next].video);
    }
  };

  return (
    <section className="results">
      <div className="results-card">
        <h2>Conversion Results</h2>

        {/* Original Text */}
        <div className="original">
          <label>Original Text</label>

          <div className="text-box">{originalText}</div>
        </div>

        {/* Information */}
        <div className="info-box">
          <div className="info">
            <FiGlobe className="icon" />

            <div>
              <span>Detected Language</span>
              <h4>{language}</h4>
            </div>
          </div>
        </div>

        {/* Detected Processes */}
        <div className="keywords">
          <h4>Detected Processes</h4>

          <div className="tags">
            {processList.map((item, index) => (
              <span key={index}>{item}</span>
            ))}
          </div>
        </div>

        {/* Video Section */}
        <div className="video-section">
          {/* Video Player */}
          <div className="video-player">
            <h4>Video Player</h4>

            {currentVideo ? (
              <video
                ref={videoRef}
                controls
                autoPlay
                width="100%"
                key={currentVideo}
                onEnded={playNext}
              >
                <source src={currentVideo} type="video/mp4" />
                Your browser does not support video.
              </video>
            ) : (
              <p>No video available.</p>
            )}
          </div>

          {/* Playlist */}
          <div className="playlist">
            <h4>Playlist ({playlist.length})</h4>

            <ul>
              {playlist.map((item, index) => (
                <li
                  key={index}
                  className={currentIndex === index ? "active" : ""}
                  onClick={() => playVideo(index)}
                >
                  {index + 1}. {item.keyword}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Buttons */}
        <div className="buttons">
          <button className="play" onClick={playAll}>
            <FiPlay />
            Play All
          </button>

          <button className="download">
            <FiDownload />
            Download Playlist
          </button>

          <button className="auto-play" onClick={() => setAutoPlay(!autoPlay)}>
            {autoPlay ? "🟢 Auto Play ON" : "⚪ Auto Play OFF"}
          </button>
        </div>
      </div>
    </section>
  );
}

export default Results;
