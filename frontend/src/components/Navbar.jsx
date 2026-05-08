import { NavLink } from 'react-router-dom'
import { Cpu, Terminal, Activity } from 'lucide-react'

export default function Navbar() {
  const linkClass = ({ isActive }) =>
    `flex items-center gap-2 px-4 py-2 text-xs font-mono tracking-widest uppercase transition-all ${
      isActive
        ? 'text-primary-400 border-b-2 border-primary-400 bg-primary-500/5'
        : 'text-white/50 hover:text-white hover:bg-white/5'
    }`

  return (
    <nav className="sticky top-0 z-50 border-b border-primary-500/20 bg-dark-950/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          
          {/* Logo / System ID */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Cpu className="w-5 h-5 text-primary-400" />
              <div className="flex flex-col">
                <span className="text-sm font-mono font-bold text-white tracking-widest leading-none">
                  DEEPSHIELD
                </span>
                <span className="text-[9px] font-mono text-primary-400/60 tracking-[0.2em] mt-1">
                  SYS.ID: DS-CORE-V2
                </span>
              </div>
            </div>
            
            {/* Status Pulse */}
            <div className="hidden sm:flex items-center gap-2 ml-4 pl-4 border-l border-white/10">
              <div className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-success-500"></span>
              </div>
              <span className="text-[10px] font-mono text-success-400">NETWORK ACTIVE</span>
            </div>
          </div>

          {/* Navigation Links */}
          <div className="flex items-center gap-2">
            <NavLink to="/" className={linkClass}>
              <Activity className="w-3 h-3" />
              <span className="hidden sm:inline">Telemetry</span>
            </NavLink>
            <NavLink to="/history" className={linkClass}>
              <Terminal className="w-3 h-3" />
              <span className="hidden sm:inline">Logs</span>
            </NavLink>
          </div>

        </div>
      </div>
    </nav>
  )
}
