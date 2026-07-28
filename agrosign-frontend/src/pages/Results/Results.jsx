// import "./Results.scss";
// import { useState, useRef } from "react";
// import {
//   FiGlobe,
//   FiActivity,
//   FiPlay,
//   FiDownload,
// } from "react-icons/fi";

// function Results() {
//   const videoRef = useRef(null);

//   const [originalText] = useState(
//     "The farmer uses a harvester to harvest the crops."
//   );

//   const [language] = useState("English");

//   const [process] = useState("Harvesting");

//   const [keywords] = useState([
//     "farmer",
//     "uses",
//     "harvester",
//     "harvest",
//     "crops",
//   ]);

//   const [playlist] = useState([
//     {
//       word: "Farmer",
//       video: "/videos/farmer.mp4",
//     },
//     {
//       word: "Harvester",
//       video: "/videos/harvester.mp4",
//     },
//     {
//       word: "Crops",
//       video: "/videos/crops.mp4",
//     },
//   ]);

//   const [currentIndex, setCurrentIndex] = useState(0);

//   const [currentVideo, setCurrentVideo] = useState(
//     "/videos/farmer.mp4"
//   );

//   const [autoPlay, setAutoPlay] = useState(true);

//   const playVideo = (index) => {
//     setCurrentIndex(index);
//     setCurrentVideo(playlist[index].video);
//   };

//   const playAll = () => {
//     playVideo(0);
//   };

//   const playNext = () => {
//     if (!autoPlay) return;

//     if (currentIndex < playlist.length - 1) {
//       const next = currentIndex + 1;

//       setCurrentIndex(next);
//       setCurrentVideo(playlist[next].video);
//     }
//   };

//   return (
//     <section className="results">
//       <div className="results-card">

//         <h2>Conversion Results</h2>

//         {/* Original Text */}

//         <div className="original">

//           <label>Original Text</label>

//           <div className="text-box">
//             {originalText}
//           </div>

//         </div>

//         {/* Info */}

//         <div className="info-box">

//           <div className="info">

//             <FiGlobe className="icon" />

//             <div>

//               <span>Detected Language</span>

//               <h4>{language}</h4>

//             </div>

//           </div>

//           <div className="info">

//             <FiActivity className="icon" />

//             <div>

//               <span>Detected Process</span>

//               <h4>{process}</h4>

//             </div>

//           </div>

//         </div>

//         {/* Keywords */}

//         <div className="keywords">

//           <h4>Detected Keywords</h4>

//           <div className="tags">

//             {keywords.map((item, index) => (

//               <span key={index}>{item}</span>

//             ))}

//           </div>

//         </div>

//         {/* Video Section */}

//         <div className="video-section">

//           <div className="video-player">

//             <h4>Video Player</h4>

//             <video
//               ref={videoRef}
//               controls
//               autoPlay
//               key={currentVideo}
//               width="100%"
//               onEnded={playNext}
//             >
//               <source src={currentVideo} type="video/mp4" />

//               Your browser does not support video.
//             </video>

//           </div>

//           <div className="playlist">

//             <h4>Playlist ({playlist.length})</h4>

//             <ul>

//               {playlist.map((item, index) => (

//                 <li
//                   key={index}
//                   className={currentIndex === index ? "active" : ""}
//                   onClick={() => playVideo(index)}
//                 >
//                   {index + 1}. {item.word}
//                 </li>

//               ))}

//             </ul>

//           </div>

//         </div>

//         {/* Buttons */}

//         <div className="buttons">



//           <button
//             className="play"
//             onClick={playAll}
//           >
//             <FiPlay />

//             Play All

//           </button>

//           <button className="download">

//             <FiDownload />

//             Download Playlist

//           </button>

//                     <button
//             className="auto-play"
//             onClick={() => setAutoPlay(!autoPlay)}
//           >
//             {autoPlay ? "🟢 Auto Play ON" : "⚪ Auto Play OFF"}
//           </button>

//         </div>

//       </div>
//     </section>
//   );
// }

// export default Results;

import "./Results.scss";
import { useState, useRef } from "react";
import { useLocation } from "react-router-dom";
import {
  FiGlobe,
  FiActivity,
  FiPlay,
  FiDownload,
} from "react-icons/fi";

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

  // const translation = data.translations[0];

  // const originalText = translation.sentence;

const originalText = data.translations
  .map((item) => item.sentence)
  .join(" ");

const language = "English";
const process = data.translations[0].process_name;

  const playlist = data.complete_video_sequence.map((video) => ({
    word: video.split("/").pop().replace(".mp4", ""),
    video,
  }));

  const keywords = playlist.map((item) => item.word);

  const [currentIndex, setCurrentIndex] = useState(0);

  const [currentVideo, setCurrentVideo] = useState(
    playlist.length > 0 ? playlist[0].video : ""
  );

  const [autoPlay, setAutoPlay] = useState(true);

  const playVideo = (index) => {
    setCurrentIndex(index);
    setCurrentVideo(playlist[index].video);
  };

  const playAll = () => {
    playVideo(0);
  };

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

          <div className="text-box">
            {originalText}
          </div>

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

          <div className="info">

            <FiActivity className="icon" />

            <div>

              <span>Detected Process</span>

              <h4>{process}</h4>

            </div>

          </div>

        </div>

        {/* Keywords */}

        <div className="keywords">

          <h4>Detected Keywords</h4>

          <div className="tags">

            {keywords.map((item, index) => (

              <span key={index}>{item}</span>

            ))}

          </div>

        </div>

        {/* Video Player */}

        <div className="video-section">

          <div className="video-player">

            <h4>Video Player</h4>

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
                  {index + 1}. {item.word}
                </li>

              ))}

            </ul>

          </div>

        </div>

        {/* Buttons */}

        <div className="buttons">

          <button
            className="play"
            onClick={playAll}
          >

            <FiPlay />

            Play All

          </button>

          <button className="download">

            <FiDownload />

            Download Playlist

          </button>

          <button
            className="auto-play"
            onClick={() => setAutoPlay(!autoPlay)}
          >

            {autoPlay
              ? "🟢 Auto Play ON"
              : "⚪ Auto Play OFF"}

          </button>

        </div>

      </div>
    </section>
  );
}

export default Results;