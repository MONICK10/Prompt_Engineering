import { Routes, Route, NavLink } from 'react-router-dom'
import EditorPage from './pages/EditorPage'
import ExperimentPage from './pages/ExperimentPage'

export default function App() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <nav className="topnav">
        <span className="topnav-brand">VDOC</span>
        <NavLink to="/"          className={({ isActive }) => isActive ? 'topnav-link active' : 'topnav-link'}>Editor</NavLink>
        <NavLink to="/experiment" className={({ isActive }) => isActive ? 'topnav-link active' : 'topnav-link'}>Experiment</NavLink>
      </nav>
      <div style={{ flex: 1, overflow: 'hidden' }}>
        <Routes>
          <Route path="/"           element={<EditorPage />} />
          <Route path="/experiment" element={<ExperimentPage />} />
        </Routes>
      </div>
    </div>
  )
}
