import { NavLink } from 'react-router-dom'
import { Shield, Clock, Activity } from 'lucide-react'

export default function Navbar() {
  const linkClass = ({ isActive }) =>
    `flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
      isActive
        ? 'bg-primary-600/20 text-primary-400 border border-primary-500/30'
        : 'text-white/60 hover:text-white hover:bg-white/5'
    }`

  return (
    <nav className="sticky top-0 z-50 border-b border-white/5 bg-dark-900/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-9 h-9 bg-gradient-to-br from-primary-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg shadow-primary-500/30">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-green-400 rounded-full border-2 border-dark-900 animate-pulse" />
            </div>
            <div>
              <span className="text-lg font-bold text-gradient">DeepShield</span>
              <p className="text-xs text-white/40 leading-none">AI Deepfake Detection</p>
            </div>
          </div>

          {/* Nav links */}
          <div className="flex items-center gap-1">
            <NavLink to="/" className={linkClass}>
              <Activity className="w-4 h-4" />
              <span className="hidden sm:inline">Detection</span>
            </NavLink>
            <NavLink to="/history" className={linkClass}>
              <Clock className="w-4 h-4" />
              <span className="hidden sm:inline">History</span>
            </NavLink>
          </div>

          {/* Status badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-xs text-green-400 font-medium hidden sm:inline">System Online</span>
          </div>
        </div>
      </div>
    </nav>
  )
}
