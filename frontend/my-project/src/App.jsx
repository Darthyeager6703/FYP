import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import InputForm from "./components/InputForm"; // Import the form component
import ResultsPage from "./components/ResultsPage";
import EvaluationMetrics from "./components/EvaluationMetrics";
import Home from "./components/Home";
import Hardware from "./components/Hardware"; // Import the new Hardware component
import "./App.css"; // Ensure CSS is applied

const Navbar = () => {
  return (
    <nav>
      <Link to="/" className="nav-home">HOME</Link>
      <div className="nav-links">
        <Link to="/hardware">Hardware</Link>
        <Link to="/results">Results</Link>
      </div>
    </nav>
  );
};

const Placeholder = ({ text }) => (
  <div className="flex items-center justify-center h-screen">
    <h1 className="text-5xl font-extrabold text-[#f7e9b6]">{text}</h1>
  </div>
);

const App = () => {
  return (
    <Router>
      <div className="h-screen flex flex-col bg-[#0f0e15] text-[#f7e9b6]">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home/>} />
          <Route path="/evaluation" element={<EvaluationMetrics/>} />
          <Route path="/hardware" element={<Hardware />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
