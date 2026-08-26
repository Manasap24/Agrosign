import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  FiHome,
  FiMic,
  FiBookOpen,
  FiInfo,
  FiFileText,
  FiMoon,
  FiSun,
  FiMenu,
  FiX,
  FiChevronDown,
} from "react-icons/fi";
import "./Navbar.scss";

function Navbar({ darkMode, toggleDarkMode }) {
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
        <NavLink to="/" className="logo" onClick={closeMenu}>
          AgroSign AI
        </NavLink>

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
        </nav>

        <div className="right-section">
          <button
            className="theme-toggle"
            type="button"
            onClick={toggleDarkMode}
            aria-label="Toggle dark mode"
          >
            {darkMode ? <FiSun /> : <FiMoon />}
          </button>

          <button
            className="menu-toggle"
            type="button"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Toggle menu"
          >
            {menuOpen ? <FiX /> : <FiMenu />}
          </button>

          <NavLink to="/text" className="get-started desktop-only">
            Get Started
            <span>→</span>
          </NavLink>
        </div>
      </div>

      {menuOpen && (
        <nav className="mobile-menu">
          <NavLink to="/" end className={linkClass} onClick={closeMenu}>
            <FiHome />
            Home
          </NavLink>

          <button
            type="button"
            className="mobile-translate"
            onClick={() => setTranslateOpen(!translateOpen)}
          >
            <span>
              <FiFileText />
              Translate
            </span>

            <FiChevronDown className={translateOpen ? "rotate" : ""} />
          </button>

          {translateOpen && (
            <div className="mobile-submenu">
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
          )}

          <NavLink to="/results" className={linkClass} onClick={closeMenu}>
            <FiBookOpen />
            Results
          </NavLink>

          <NavLink to="/about" className={linkClass} onClick={closeMenu}>
            <FiInfo />
            About
          </NavLink>
        </nav>
      )}
    </header>
  );
}

export default Navbar;
