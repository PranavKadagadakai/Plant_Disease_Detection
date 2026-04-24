import { useState } from "react";
import { predictImage } from "../api/api";

function Predict() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);

    if (selected) {
      setPreview(URL.createObjectURL(selected));
    }
  };

  const handlePredict = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setResult(null);

    try {
      const res = await predictImage(formData);
      setResult(res.data);
    } catch (err) {
      setResult({ error: `Prediction failed: ${err.message}` });
    }

    setLoading(false);
  };

  return (
    <div className="card">
      <h3>Predict Disease</h3>

      <input type="file" onChange={handleFileChange} />

      {preview && <img src={preview} alt="preview" className="preview" />}

      <button onClick={handlePredict} disabled={loading}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      {result && (
        <div className="metrics">
          {result.error ? (
            <p>{result.error}</p>
          ) : (
            Object.entries(result).map(([key, value]) => (
              <div key={key} className="metric-item">
                <strong>{key}:</strong>{" "}
                {typeof value === "number" ? value.toString() : value}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default Predict;
