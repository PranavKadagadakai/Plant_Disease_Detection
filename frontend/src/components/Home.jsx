import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";

function Home() {
  const { t } = useTranslation();

  return (
    <div className="home-page">
      <section className="hero-section">
        <div className="hero-content">
          <span className="hero-badge">{t("heroBadge")}</span>

          <h1>{t("homeTitle")}</h1>

          <p className="hero-description">{t("homeDescription")}</p>

          <div className="hero-actions">
            <Link to="/detect" className="primary-btn">
              {t("startDetection")}
            </Link>

            <Link to="/evaluate" className="secondary-btn">
              {t("viewEvaluation")}
            </Link>
          </div>
        </div>

        <div className="hero-card-grid">
          <div className="feature-card">
            <h3>{t("featureOneTitle")}</h3>
            <p>{t("featureOneDescription")}</p>
          </div>

          <div className="feature-card">
            <h3>{t("featureTwoTitle")}</h3>
            <p>{t("featureTwoDescription")}</p>
          </div>

          <div className="feature-card">
            <h3>{t("featureThreeTitle")}</h3>
            <p>{t("featureThreeDescription")}</p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Home;
