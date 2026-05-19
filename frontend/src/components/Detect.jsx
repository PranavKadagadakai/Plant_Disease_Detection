import { useState } from "react";
import { detectDisease } from "../api/api";
import { useTranslation } from "react-i18next";

import DetectionResult from "./DetectionResult";

function Detect() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const { t } = useTranslation();

  const handleFileChange = (e) => {
    const selected = e.target.files[0];

    if (!selected) return;

    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
  };

  const handlePredict = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setResult(null);

    try {
      const res = await detectDisease(formData);
      setResult(res.data);
    } catch (err) {
      setResult({
        error: err?.response?.data?.detail || err.message || "Detection failed",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card detect-card">
      <h3>{t("plantDiseaseDetection")}</h3>

      <input type="file" accept="image/*" onChange={handleFileChange} />

      {preview && (
        <div className="preview-container">
          <img src={preview} alt="preview" className="preview" />
        </div>
      )}

      <button onClick={handlePredict} disabled={loading || !file}>
        {loading ? t("detecting") : t("detectDisease")}
      </button>

      {result && <DetectionResult result={result} />}
    </div>
  );
}

export default Detect;
