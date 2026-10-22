import { Routes, Route, NavLink } from 'react-router-dom'
import EditorPage from './pages/EditorPage'
import ExperimentPage from './pages/ExperimentPage'
import LiveDemoPage from './pages/LiveDemoPage'

const link = ({ isActive }) => isActive ? 'topnav-link active' : 'topnav-link'

export default function App() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <nav className="topnav">
        <span className="topnav-brand">VDOC 2.0</span>
        <NavLink to="/"           className={link}>Live Demo</NavLink>
        <NavLink to="/editor"     className={link}>Editor</NavLink>
        <NavLink to="/experiment" className={link}>Experiment</NavLink>
      </nav>
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <Routes>
          <Route path="/"           element={<LiveDemoPage />} />
          <Route path="/editor"     element={<EditorPage />} />
          <Route path="/experiment" element={<ExperimentPage />} />
        </Routes>
      </div>
    </div>
  )
}
