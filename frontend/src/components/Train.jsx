import { useState } from "react";

import { trainModel } from "../api/api";

import { useTranslation } from "react-i18next";

function Train() {
  const [loading, setLoading] = useState(false);

  const [result, setResult] = useState(null);

  const { t } = useTranslation();

  const handleTrain = async () => {
    setLoading(true);
    setResult(null);

    try {
      const res = await trainModel();
      setResult(res.data);
    } catch (err) {
      setResult({
        error: t("errorTraining"),
      });
    }

    setLoading(false);
  };

  return (
    <div className="card">
      <h3>{t("trainModel")}</h3>

      <button onClick={handleTrain} disabled={loading}>
        {loading ? t("training") : t("train")}
      </button>

      {result && (
        <div className="metrics">
          {result.error ? (
            <p>{result.error}</p>
          ) : (
            Object.entries(result).map(([key, value]) => (
              <div key={key} className="metric-item">
                <strong>{key}:</strong>

                {typeof value === "number"
                  ? value.toFixed(4)
                  : value.toString()}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default Train;
