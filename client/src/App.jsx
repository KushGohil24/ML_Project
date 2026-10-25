// App.jsx - Root component with routing and navigation
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import {
  FiShield,
  FiHome,
  FiSearch,
  FiDatabase,
  FiTrendingUp,
  FiClock,
} from "react-icons/fi";

import Home        from "./pages/Home";
import Predict     from "./pages/Predict";
import History     from "./pages/History";
import EDA         from "./pages/EDA";
import Insights    from "./pages/Insights";

import "./index.css";

function Navbar() {
  const navItems = [
    { to: "/",         label: "Overview",   Icon: FiHome },
    { to: "/predict",  label: "Detect",     Icon: FiSearch },
    { to: "/eda",      label: "Dataset",    Icon: FiDatabase },
    { to: "/insights", label: "Insights",   Icon: FiTrendingUp },
    { to: "/history",  label: "History",    Icon: FiClock },
  ];

  return (
    <nav className="navbar">
      <NavLink to="/" className="navbar-brand">
        <div className="brand-icon">
          <FiShield size={18} />
        </div>
        <span>FakeJobDetector</span>
      </NavLink>
      <ul className="navbar-links">
        {navItems.map(({ to, label, Icon }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={to === "/"}
              className={({ isActive }) => (isActive ? "active" : "")}
            >
              <Icon size={15} />
              <span>{label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function Footer() {
  return (
    <footer className="footer">
      <p>
        Fake Job Posting Detection System &nbsp;·&nbsp; College ML Project &nbsp;·&nbsp;
        MERN Stack + Python ML Service (SVM · TF-IDF · SMOTE)
      </p>
    </footer>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/"         element={<Home />} />
            <Route path="/predict"  element={<Predict />} />
            <Route path="/eda"      element={<EDA />} />
            <Route path="/insights" element={<Insights />} />
            <Route path="/history"  element={<History />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}
