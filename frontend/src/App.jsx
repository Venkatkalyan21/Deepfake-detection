import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import History from './pages/History'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-dark-900 overflow-hidden relative">
        {/* Forensic Scanline & Background Effects */}
        <div className="fixed inset-0 pointer-events-none z-0 opacity-20">
          <div className="absolute inset-0 bg-grid-pattern" />
          <div className="w-full h-1 bg-primary-400/50 shadow-[0_0_20px_#00f0ff] animate-scanline absolute top-0" />
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-primary-500/5 to-transparent animate-scanline" />
        </div>

        <Navbar />

        <main className="relative z-10">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
