import Navbar from "./components/Navbar";
import { Routes, Route } from "react-router-dom";

import Train from "./components/Train";
import Evaluate from "./components/Evaluate";
import Predict from "./components/Predict";

function App() {
  return (
    <div className="app">
      <Navbar />
      <div className="page">
        <Routes>
          <Route path="/" element={<Train />} />
          <Route path="/train" element={<Train />} />
          <Route path="/evaluate" element={<Evaluate />} />
          <Route path="/predict" element={<Predict />} />
        </Routes>
      </div>
    </div>
  );
}

export default App;
