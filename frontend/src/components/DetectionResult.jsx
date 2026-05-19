import ConfidenceBar from "./ConfidenceBar";

import TreatmentCard from "./TreatmentCard";

import { useLanguage } from "../context/LanguageContext";

import { useTranslation } from "react-i18next";

function DetectionResult({ result }) {
  const { language } = useLanguage();

  const { t } = useTranslation();

  if (result.error) {
    return (
      <div className="result-container">
        <div className="error-box">
          <p>{result.error}</p>
        </div>
      </div>
    );
  }

  const diseaseName =
    result.display_names?.[language] || result.display_names?.en;

  const treatment = result.treatments[language];

  return (
    <div className="result-container">
      {/* Detection Info */}
      <div className="result-section">
        <h4>{t("detectionResult")}</h4>

        <div className="metric-item">
          <strong>{t("classIndex")}:</strong>

          <span>{result.class_index}</span>
        </div>

        <div className="metric-item">
          <strong>{t("rawClass")}:</strong>

          <span>{result.normalized_class_name}</span>
        </div>

        <div className="metric-item">
          <strong>{t("confidence")}:</strong>

          <span>{(result.confidence * 100).toFixed(2)}%</span>
        </div>

        <ConfidenceBar confidence={result.confidence} />

        {result.low_confidence_flag && result.advisory && (
          <div className="warning-box">
            <h4>⚠ {t("lowConfidenceWarning")}</h4>

            <p>{result.advisory.message[language]}</p>

            <div className="contact-list">
              {result.advisory.contacts.map((contact, index) => (
                <div key={index} className="contact-item">
                  <strong>{contact.name}</strong>

                  <span>{contact.value}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Disease Name */}
      <div className="result-section">
        <h4>{t("diseaseName")}</h4>

        <div className="language-card">
          <p>{diseaseName}</p>
        </div>
      </div>

      {/* Top Predictions */}
      {result.top_predictions && (
        <div className="result-section">
          <h4>{t("topPredictions")}</h4>

          {result.top_predictions.map((item, index) => (
            <div key={index} className="metric-item">
              <span>{item.display_names[language]}</span>

              <span>{(item.confidence * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      )}

      {/* Treatments */}
      <div className="result-section">
        <h4>{t("treatmentRecommendations")}</h4>

        <div className="treatment-grid">
          <TreatmentCard language={language} treatment={treatment} />
        </div>
      </div>
    </div>
  );
}

export default DetectionResult;
