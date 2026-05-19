import { useState } from "react";

import { evaluateModel } from "../api/api";

import { useTranslation } from "react-i18next";

function Evaluate() {
  const [metrics, setMetrics] = useState(null);

  const [loading, setLoading] = useState(false);

  const { t } = useTranslation();

  const handleEvaluate = async () => {
    setLoading(true);

    try {
      const res = await evaluateModel();
      setMetrics(res.data);
    } catch (err) {
      setMetrics({
        error: t("evaluationFailed"),
      });
    }

    setLoading(false);
  };

  return (
    <div className="evaluate-page">
      <div className="card">
        <h3>{t("evaluateModel")}</h3>

        <button onClick={handleEvaluate} disabled={loading}>
          {loading ? t("evaluating") : t("evaluate")}
        </button>

        {metrics && (
          <div className="metrics">
            {metrics.error ? (
              <p className="error">{metrics.error}</p>
            ) : (
              Object.entries(metrics).map(([key, value]) => {
                if (key === "confusion_matrix_image") return null;

                return (
                  <div key={key} className="metric-item">
                    <strong>{key}:</strong>

                    {typeof value === "number"
                      ? value.toFixed(4)
                      : value.toString()}
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>

      {metrics && metrics.confusion_matrix_image && (
        <div className="cm-full-section">
          <h3>{t("confusionMatrix")}</h3>

          <button
            onClick={() => {
              const link = document.createElement("a");

              link.href = `data:image/png;base64,${metrics.confusion_matrix_image}`;

              link.download = "cm.png";

              link.click();
            }}
          >
            {t("download")}
          </button>

          <div className="cm-full-container">
            <img
              src={`data:image/png;base64,${metrics.confusion_matrix_image}`}
              alt="Confusion Matrix"
              className="cm-full-image"
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default Evaluate;
