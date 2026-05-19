import Train from "../components/Train";
import Evaluate from "../components/Evaluate";
import Detect from "../components/Detect";

function Dashboard() {
  return (
    <div className="container">
      <Train />
      <Evaluate />
      <Detect />
    </div>
  );
}

export default Dashboard;
