import { Link, useLocation } from "react-router-dom";

function Navbar() {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <div className="navbar">
      <h2>Plant Disease Detection</h2>

      <div className="nav-links">
        <Link className={isActive("/train") ? "active" : ""} to="/train">
          Train
        </Link>
        <Link className={isActive("/evaluate") ? "active" : ""} to="/evaluate">
          Evaluate
        </Link>
        <Link className={isActive("/predict") ? "active" : ""} to="/predict">
          Predict
        </Link>
      </div>
    </div>
  );
}

export default Navbar;
