import { useState } from "react";
import "./Navbar.scss";
import { NavLink } from "react-router-dom";
import {
  FiHome,
  FiMic,
  FiBookOpen,
  FiInfo,
  FiMail,
  FiFileText,
  FiGlobe,
  FiMoon,
  FiArrowRight,
  FiMenu,
  FiX,
  FiChevronDown
} from "react-icons/fi";

function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [translateOpen, setTranslateOpen] = useState(false);

  const linkClass = ({ isActive }) => (isActive ? "active" : "");
  const closeMenu = () => {
    setMenuOpen(false);
    setTranslateOpen(false);
  };

  return (
    <header className="navbar-wrapper">
      <div className="navbar">
        <div className="logo">
          <h2>AgroSign AI</h2>
        </div>

        {/* Desktop nav — hidden below 992px via CSS */}
        <nav className="nav-links desktop-only">
          <NavLink to="/" end className={linkClass}>
            <FiHome />
            Home
          </NavLink>
          <NavLink to="/text" className={linkClass}>
            <FiFileText />
            Text to Sign
          </NavLink>
          <NavLink to="/speech" className={linkClass}>
            <FiMic />
            Speech to Sign
          </NavLink>
          <NavLink to="/pdf" className={linkClass}>
            <FiFileText />
            PDF to Sign
          </NavLink>
          <NavLink to="/results" className={linkClass}>
            <FiBookOpen />
            Results
          </NavLink>
          <NavLink to="/about" className={linkClass}>
            <FiInfo />
            About
          </NavLink>
          {/* <NavLink to="/contact" className={linkClass}>
            <FiMail />
            Contact
          </NavLink> */}
        </nav>

        <div className="right-section">
          {/* <button className="language" type="button">
            <FiGlobe />
            <span>English</span>
          </button> */}

          <button className="theme" type="button" aria-label="Toggle theme">
            <FiMoon />
          </button>

          <button className="start-btn desktop-only" type="button">
            <span>Get Started</span>
            <FiArrowRight />
          </button>

          {/* Hamburger — only visible below 992px */}
          <button
            className="menu-toggle"
            type="button"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((open) => !open)}
          >
            {menuOpen ? <FiX /> : <FiMenu />}
          </button>
        </div>
      </div>

      {/* Mobile slide-down menu */}
      <nav className={`mobile-menu ${menuOpen ? "open" : ""}`}>
        <NavLink to="/" end className={linkClass} onClick={closeMenu}>
          <FiHome />
          Home
        </NavLink>

        {/* Translate dropdown — groups Text / Speech / PDF to Sign */}
        <div className={`mobile-dropdown ${translateOpen ? "open" : ""}`}>
          <button
            type="button"
            className="mobile-dropdown__trigger"
            aria-expanded={translateOpen}
            onClick={() => setTranslateOpen((open) => !open)}
          >
            <span>
              <FiFileText />
              Translate
            </span>
            <FiChevronDown className="chevron" />
          </button>

          <div className="mobile-dropdown__panel">
            <NavLink to="/text" className={linkClass} onClick={closeMenu}>
              <FiFileText />
              Text to Sign
            </NavLink>
            <NavLink to="/speech" className={linkClass} onClick={closeMenu}>
              <FiMic />
              Speech to Sign
            </NavLink>
            <NavLink to="/pdf" className={linkClass} onClick={closeMenu}>
              <FiFileText />
              PDF to Sign
            </NavLink>
          </div>
        </div>

        <NavLink to="/dictionary" className={linkClass} onClick={closeMenu}>
          <FiBookOpen />
          Dictionary
        </NavLink>
        <NavLink to="/about" className={linkClass} onClick={closeMenu}>
          <FiInfo />
          About
        </NavLink>
        <NavLink to="/contact" className={linkClass} onClick={closeMenu}>
          <FiMail />
          Contact
        </NavLink>

        <button className="start-btn mobile-start" type="button" onClick={closeMenu}>
          Get Started
          <FiArrowRight />
        </button>
      </nav>
    </header>
  );
}

export default Navbar;