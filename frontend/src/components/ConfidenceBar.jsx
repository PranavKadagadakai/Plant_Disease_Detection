function ConfidenceBar({ confidence }) {
  return (
    <div className="confidence-bar-wrapper">
      <div
        className="confidence-bar"
        style={{
          width: `${confidence * 100}%`,
        }}
      />
    </div>
  );
}

export default ConfidenceBar;
