import { CircularProgressbar, buildStyles } from 'react-circular-progressbar'
import 'react-circular-progressbar/dist/styles.css'
import { AlertTriangle, CheckCircle, HelpCircle, Activity } from 'lucide-react'

export default function ResultPanel({ result, loading }) {
  if (loading) {
    return (
      <div className="glass p-8 rounded-2xl flex flex-col items-center justify-center min-h-[400px] animate-pulse">
        <div className="relative">
          <div className="w-24 h-24 rounded-full border-4 border-white/5 border-t-primary-500 animate-spin" />
          <div className="absolute inset-0 flex items-center justify-center">
            <Activity className="w-8 h-8 text-primary-500 animate-pulse" />
          </div>
        </div>
        <h3 className="text-xl font-bold mt-6 text-white">Analyzing Media</h3>
        <p className="text-white/50 text-sm mt-2 text-center max-w-xs">
          DeepShield is scanning for AI manipulation, texture artifacts, and facial inconsistencies…
        </p>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="glass p-8 rounded-2xl flex flex-col items-center justify-center min-h-[400px] border-dashed border-2 border-white/5">
        <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6">
          <Shield className="w-10 h-10 text-white/20" />
        </div>
        <h3 className="text-xl font-bold text-white/40">Ready for Analysis</h3>
        <p className="text-white/30 text-sm mt-2 text-center">
          Upload a file to see the deepfake detection results here.
        </p>
      </div>
    )
  }

  const { verdict, confidence, modality, details } = result
  const isFake = verdict?.toUpperCase() === 'FAKE'
  const isReal = verdict?.toUpperCase() === 'REAL'
  
  const color = isFake ? '#ef4444' : isReal ? '#22c55e' : '#eab308'
  const Icon = isFake ? AlertTriangle : isReal ? CheckCircle : HelpCircle
  const percentage = Math.round((confidence || 0) * 100)

  return (
    <div className="glass p-8 rounded-2xl animate-slide-up relative overflow-hidden">
      {/* Background glow based on verdict */}
      <div 
        className="absolute -top-40 -right-40 w-96 h-96 rounded-full blur-3xl opacity-10 pointer-events-none"
        style={{ backgroundColor: color }}
      />

      <div className="flex flex-col md:flex-row items-center gap-8 relative z-10">
        {/* Gauge */}
        <div className="w-48 h-48 shrink-0 relative">
          <CircularProgressbar
            value={percentage}
            text={`${percentage}%`}
            styles={buildStyles({
              pathColor: color,
              textColor: '#fff',
              trailColor: 'rgba(255,255,255,0.05)',
              pathTransitionDuration: 1.5,
              textSize: '18px',
            })}
          />
          <p className="text-center text-xs text-white/50 mt-4 uppercase tracking-wider font-semibold">
            Confidence Score
          </p>
        </div>

        {/* Details */}
        <div className="flex-1 text-center md:text-left">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-xs font-semibold uppercase tracking-wider text-white/70 mb-4">
            <span className="w-2 h-2 rounded-full bg-primary-500" />
            {modality} Analysis
          </div>

          <h2 className="text-4xl font-black text-white mb-2 flex items-center justify-center md:justify-start gap-3">
            <Icon className="w-10 h-10" style={{ color }} />
            <span style={{ color }}>{verdict}</span>
          </h2>
          
          <p className="text-white/60 mb-6 max-w-md">
            {isFake 
              ? "High probability of AI manipulation detected. The media exhibits characteristics consistent with deepfake generation."
              : isReal
              ? "No significant AI manipulation detected. The media appears to be authentic."
              : "Analysis inconclusive. Unable to make a definitive determination."}
          </p>

          {/* Breakdown / Details */}
          {details && (
            <div className="grid grid-cols-2 gap-3">
              {Object.entries(details).map(([key, val]) => (
                <div key={key} className="bg-white/5 border border-white/5 rounded-xl p-3">
                  <p className="text-xs text-white/40 uppercase tracking-wider mb-1">{key.replace(/_/g, ' ')}</p>
                  <p className="text-sm font-medium text-white">{val}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function Shield({ className }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
    </svg>
  )
}
