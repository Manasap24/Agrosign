// import "./TextToSign.scss";
// import { useState } from "react";
// import {
//   FiGlobe,
//   FiArrowRight,
//   FiFileText
// } from "react-icons/fi";

// function TextToSign() {

//   const [text, setText] = useState("");
//   const [language, setLanguage] = useState("english");

//   const examples = [
//     "Farmer irrigates the field.",
//     "Use pesticide carefully.",
//     "Harvest crops.",
//     "Tractor helps in plowing."
//   ];

//   const handleConvert = () => {
//     console.log(text);
//   };

//   return (
//     <section className="text-sign">

//       <div className="text-card">

//         <h2>Text to Sign</h2>

//         <div className="input-group">

//           <label>Enter your text</label>

//           <div className="textarea-box">

//             <textarea
//               placeholder="Type agricultural text..."
//               maxLength={500}
//               value={text}
//               onChange={(e) => setText(e.target.value)}
//             />

//             <span>{text.length}/500</span>

//           </div>

//         </div>

//         <div className="input-group">

//           <label>Select Language</label>

//           <div className="select-box">

//             <FiGlobe />

//             <select value={language} onChange={(e) => setLanguage(e.target.value)}>

//               <option value="english">English</option>
//               <option value="hindi">Hindi</option>
//               <option value="kannada">Kannada</option>
              

//             </select>

//           </div>

//         </div>

//        {/* <div className="samples">

//           <h4>Sample Inputs</h4>

//           <div className="chips">

//             {examples.map((item, index) => (

//               <button
//                 key={index}
//                 onClick={() => setText(item)}
//               >
//                 {item}
//               </button>

//             ))}

//           </div>

//         </div> */}

//         <div className="button-area">

//           <button onClick={handleConvert}>

//             <FiFileText />

//             Convert to Sign

//             <FiArrowRight />

//           </button>

//         </div>

//       </div>

//     </section>
//   );
// }

// export default TextToSign;


import "./TextToSign.scss";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  FiGlobe,
  FiArrowRight,
  FiFileText,
} from "react-icons/fi";

function TextToSign() {
  const navigate = useNavigate();

  const [text, setText] = useState("");
  const [language, setLanguage] = useState("english");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const examples = [
    "Farmer irrigates the field.",
    "Use pesticide carefully.",
    "Harvest crops.",
    "Tractor helps in plowing.",
  ];

  const handleConvert = async () => {
    if (!text.trim()) {
      setError("Please enter some text.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:8000/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: text,
          language: language,
        }),
      });

      if (!response.ok) {
        throw new Error("Backend Error");
      }

      const result = await response.json();

      console.log(result);

      navigate("/results", {
        state: result,
      });
    } catch (err) {
      console.error(err);
      setError("Unable to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="text-sign">
      <div className="text-card">
        <h2>Text to Sign</h2>

        <div className="input-group">
          <label>Enter your text</label>

          <div className="textarea-box">
            <textarea
              placeholder="Type agricultural text..."
              maxLength={500}
              value={text}
              onChange={(e) => setText(e.target.value)}
            />

            <span>{text.length}/500</span>
          </div>
        </div>

        <div className="input-group">
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

        {/* Sample Inputs */}

        {/* 
        <div className="samples">
          <h4>Sample Inputs</h4>

          <div className="chips">
            {examples.map((item, index) => (
              <button
                key={index}
                onClick={() => setText(item)}
              >
                {item}
              </button>
            ))}
          </div>
        </div>
        */}

        <div className="button-area">
          <button onClick={handleConvert} disabled={loading}>
            <FiFileText />

            {loading ? "Converting..." : "Convert to Sign"}

            <FiArrowRight />
          </button>
        </div>

        {error && (
          <p
            style={{
              color: "red",
              marginTop: "15px",
              textAlign: "center",
            }}
          >
            {error}
          </p>
        )}
      </div>
    </section>
  );
}

export default TextToSign;