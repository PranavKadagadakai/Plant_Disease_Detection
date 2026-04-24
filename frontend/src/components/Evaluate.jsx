import { useState } from "react";
import { evaluateModel } from "../api/api";

function Evaluate() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleEvaluate = async () => {
    setLoading(true);
    try {
      const res = await evaluateModel();
      setMetrics(res.data);
    } catch (err) {
      setMetrics({ error: "Evaluation failed" });
    }
    setLoading(false);
  };

  return (
    <div className="evaluate-page">
      {/* Centered Card */}
      <div className="card">
        <h3>Evaluate Model</h3>

        <button onClick={handleEvaluate} disabled={loading}>
          {loading ? "Evaluating..." : "Evaluate"}
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
                    <strong>{key}:</strong>{" "}
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

      {/* Full-width Confusion Matrix */}
      {metrics && metrics.confusion_matrix_image && (
        <div className="cm-full-section">
          <h3>Confusion Matrix</h3>

          <button
            onClick={() => {
              const link = document.createElement("a");
              link.href = `data:image/png;base64,${metrics.confusion_matrix_image}`;
              link.download = "cm.png";
              link.click();
            }}
          >
            Download
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
