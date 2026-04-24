import { useState } from "react";
import { trainModel } from "../api/api";

function Train() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleTrain = async () => {
    setLoading(true);
    setResult(null);

    try {
      const res = await trainModel();
      setResult(res.data);
    } catch (err) {
      setResult({ error: "Error during training" });
    }

    setLoading(false);
  };

  return (
    <div className="card">
      <h3>Train Model</h3>

      <button onClick={handleTrain} disabled={loading}>
        {loading ? "Training..." : "Train"}
      </button>

      {result && (
        <div className="metrics">
          {result.error ? (
            <p>{result.error}</p>
          ) : (
            Object.entries(result).map(([key, value]) => (
              <div key={key} className="metric-item">
                <strong>{key}:</strong>{" "}
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
