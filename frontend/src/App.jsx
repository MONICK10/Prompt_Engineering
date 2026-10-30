import { Routes, Route, NavLink } from 'react-router-dom'
import EditorPage from './pages/EditorPage'
import ExperimentPage from './pages/ExperimentPage'
import LiveDemoPage from './pages/LiveDemoPage'
import HardCasesPage from './pages/HardCasesPage'
import PromptEngineeringPage from './pages/PromptEngineeringPage'
import StudioPage from './pages/StudioPage'

const link = ({ isActive }) => isActive ? 'topnav-link active' : 'topnav-link'

export default function App() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <nav className="topnav">
        <span className="topnav-brand">VDOC 2.0</span>
        <NavLink to="/"           className={link}>Live Demo</NavLink>
        <NavLink to="/hard-cases" className={link}>Hard Cases</NavLink>
        <NavLink to="/editor"     className={link}>Editor</NavLink>
        <NavLink to="/experiment" className={link}>Experiment</NavLink>
        <NavLink to="/prompts"    className={link}>Prompt Engineering</NavLink>
        <NavLink to="/studio"     className={link}>VDOC Studio</NavLink>
      </nav>
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <Routes>
          <Route path="/"           element={<LiveDemoPage />} />
          <Route path="/hard-cases" element={<HardCasesPage />} />
          <Route path="/editor"     element={<EditorPage />} />
          <Route path="/experiment" element={<ExperimentPage />} />
          <Route path="/prompts"    element={<PromptEngineeringPage />} />
          <Route path="/studio"     element={<StudioPage />} />
        </Routes>
      </div>
    </div>
  )
}
