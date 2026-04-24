import Train from "../components/Train";
import Evaluate from "../components/Evaluate";
import Predict from "../components/Predict";

function Dashboard() {
  return (
    <div className="container">
      <Train />
      <Evaluate />
      <Predict />
    </div>
  );
}

export default Dashboard;
