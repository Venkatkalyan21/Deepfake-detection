import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import History from './pages/History'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-dark-900">
        {/* Ambient background blobs */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
          <div className="absolute -top-40 -left-40 w-96 h-96 bg-primary-600/20 rounded-full blur-3xl animate-pulse-slow" />
          <div className="absolute top-1/2 -right-40 w-80 h-80 bg-purple-600/15 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1.5s' }} />
          <div className="absolute -bottom-20 left-1/3 w-72 h-72 bg-pink-600/10 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '3s' }} />
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
