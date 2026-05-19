import { useState } from "react";

import { Link, useLocation } from "react-router-dom";

import { FiSettings } from "react-icons/fi";

import { useTheme } from "../context/ThemeContext";
import { useLanguage } from "../context/LanguageContext";

import { useTranslation } from "react-i18next";

function Navbar() {
  const location = useLocation();

  const { theme, setTheme } = useTheme();

  const { language, setLanguage } = useLanguage();

  const { t } = useTranslation();

  const [open, setOpen] = useState(false);

  const isActive = (path) => location.pathname === path;

  return (
    <div className="navbar">
      <div className="navbar-top">
        <h2>{t("appTitle")}</h2>

        <div className="settings-container">
          <button className="settings-btn" onClick={() => setOpen(!open)}>
            <FiSettings size={22} />
          </button>

          {open && (
            <div className="settings-dropdown">
              <div className="dropdown-section">
                <label>{t("theme")}</label>

                <select
                  value={theme}
                  onChange={(e) => setTheme(e.target.value)}
                >
                  <option value="system">{t("system")}</option>

                  <option value="dark">{t("dark")}</option>

                  <option value="light">{t("light")}</option>
                </select>
              </div>

              <div className="dropdown-section">
                <label>{t("language")}</label>

                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                >
                  <option value="en">English</option>

                  <option value="hi">हिन्दी</option>

                  <option value="kn">ಕನ್ನಡ</option>
                </select>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="nav-links">
        <Link className={isActive("/train") ? "active" : ""} to="/train">
          {t("train")}
        </Link>

        <Link className={isActive("/evaluate") ? "active" : ""} to="/evaluate">
          {t("evaluate")}
        </Link>

        <Link className={isActive("/detect") ? "active" : ""} to="/detect">
          {t("detect")}
        </Link>
      </div>
    </div>
  );
}

export default Navbar;
