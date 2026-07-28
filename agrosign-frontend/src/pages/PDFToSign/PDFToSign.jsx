import "./PDFToSign.scss";
import { useRef, useState } from "react";
import { FiUploadCloud, FiArrowRight, FiFileText } from "react-icons/fi";

function PDFToSign() {
  const inputRef = useRef(null);

  const [fileName, setFileName] = useState("");
  const [previewText, setPreviewText] = useState("");

  const handleFile = (e) => {
    const file = e.target.files[0];

    if (!file) return;

    setFileName(file.name);

    // Temporary Preview
    setPreviewText(
      "Modern agricultural practices help increase crop yield.\n\nFarmers should use fertilizers and water in the correct proportion.\n\nPest management is important for healthy crops."
    );
  };

  const handleConvert = () => {
    console.log("Convert PDF");
  };

  return (
    <section className="pdf-page">

      <div className="pdf-card">

        <h2>PDF to Sign</h2>

        <div
          className="upload-box"
          onClick={() => inputRef.current.click()}
        >
          <FiUploadCloud className="upload-icon" />

          <h3>Drag & Drop your PDF here</h3>

          <p>or</p>

          <button>Choose PDF</button>

          <small>Supports PDF up to 20MB</small>

          {fileName && (
            <span className="filename">
              <FiFileText />
              {fileName}
            </span>
          )}

          <input
            type="file"
            accept=".pdf"
            hidden
            ref={inputRef}
            onChange={handleFile}
          />
        </div>

        <div className="preview">

          <label>Extracted Text Preview</label>

          <textarea
            value={previewText}
            readOnly
          />

        </div>

        <div className="button-area">

          <button onClick={handleConvert}>

            Convert to Sign

            <FiArrowRight />

          </button>

        </div>

      </div>

    </section>
  );
}

export default PDFToSign;