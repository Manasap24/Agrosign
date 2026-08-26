import { useNavigate } from "react-router-dom";
import farmerHero from "../../assets/farmer-hero.png";
import "./Home.scss";

function Home() {
  const navigate = useNavigate();

  return (
    <div className="home-page">
      <section className="hero-section">
        <div className="hero-content">
          <span className="hero-tag">AI-Powered Agriculture</span>

          <h1>
            Bridging Agriculture
            <br />
            and Accessibility
          </h1>

          <p>
            AgroSign AI converts agricultural information into Indian Sign
            Language using intelligent language understanding and semantic
            matching.
          </p>

          <button onClick={() => navigate("/text")}>
            Convert Text to Sign →
          </button>
        </div>

        <div className="hero-visual">
          <img src={farmerHero} alt="Farmer using AgroSign AI" />
        </div>
      </section>

      <section className="features-section">
        <h2>Choose Your Input</h2>

        <div className="feature-grid">
          <div className="feature-card" onClick={() => navigate("/text")}>
            <h3>Text to Sign</h3>
            <p>Convert agricultural text into a sequence of sign videos.</p>
            <span>Try now →</span>
          </div>

          <div className="feature-card" onClick={() => navigate("/speech")}>
            <h3>Speech to Sign</h3>
            <p>Convert spoken agricultural information into sign language.</p>
            <span>Try now →</span>
          </div>

          <div className="feature-card" onClick={() => navigate("/pdf")}>
            <h3>PDF to Sign</h3>
            <p>
              Extract agricultural content from PDFs and convert it to signs.
            </p>
            <span>Try now →</span>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Home;
