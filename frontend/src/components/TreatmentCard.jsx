import { useTranslation } from "react-i18next";

function TreatmentCard({ language, treatment }) {
  const { t } = useTranslation();

  return (
    <div className="treatment-language-card">
      <h5>{language.toUpperCase()}</h5>

      <div className="treatment-item">
        <strong>{t("organic")}:</strong>

        <p>{treatment.organic}</p>
      </div>

      <div className="treatment-item">
        <strong>{t("chemical")}:</strong>

        <p>{treatment.chemical}</p>
      </div>

      <div className="treatment-item">
        <strong>{t("cultural")}:</strong>

        <p>{treatment.cultural}</p>
      </div>
    </div>
  );
}

export default TreatmentCard;
