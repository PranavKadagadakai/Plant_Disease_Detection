function LanguageCard({ language, text }) {
  return (
    <div className="language-card">
      <strong>{language}</strong>

      <p>{text}</p>
    </div>
  );
}

export default LanguageCard;
