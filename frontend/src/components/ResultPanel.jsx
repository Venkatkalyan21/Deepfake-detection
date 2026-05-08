import { CircularProgressbar, buildStyles } from 'react-circular-progressbar'
import 'react-circular-progressbar/dist/styles.css'
import { AlertTriangle, CheckCircle, HelpCircle, Activity, Film } from 'lucide-react'

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
    <div className="glass p-6 animate-slide-up relative overflow-hidden flex flex-col h-full justify-between">
      {/* Background glow based on verdict */}
      <div 
        className="absolute -top-40 -right-40 w-96 h-96 rounded-full blur-3xl opacity-5 pointer-events-none"
        style={{ backgroundColor: color }}
      />

      <div className="flex flex-col md:flex-row gap-6 relative z-10">
        
        {/* Gauge / Radar Area */}
        <div className="w-40 h-40 shrink-0 relative flex flex-col items-center justify-center border border-white/5 bg-dark-950 p-4">
          <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-white/20" />
          <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-white/20" />
          
          <CircularProgressbar
            value={percentage}
            text={`${percentage}%`}
            styles={buildStyles({
              pathColor: color,
              textColor: color,
              trailColor: 'rgba(255,255,255,0.05)',
              pathTransitionDuration: 1.5,
              textSize: '18px',
            })}
          />
          <p className="text-[10px] font-mono text-white/40 mt-3 tracking-widest text-center">
            CONFIDENCE_IDX
          </p>
        </div>

        {/* Data Readout Area */}
        <div className="flex-1 flex flex-col">
          <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/10">
            <div className="inline-flex items-center gap-2 px-2 py-1 bg-primary-500/10 border border-primary-500/20 text-[10px] font-mono text-primary-400 uppercase tracking-widest">
              <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-pulse" />
              MODALITY: {modality}
            </div>
            <div className="text-[10px] font-mono text-white/30">
              ID: {result.session_id ? result.session_id.split('-')[0] : 'N/A'}
            </div>
          </div>

          <h2 className="text-3xl font-mono font-bold flex items-center gap-3 mt-2" style={{ color }}>
            <Icon className="w-8 h-8" />
            {verdict}
          </h2>
          
          <p className="text-[11px] font-mono text-white/50 mt-3 max-w-md leading-relaxed border-l-2 border-white/10 pl-3">
            {isFake 
              ? "CRITICAL: Neural artifacts and frequency anomalies detected. Metadata indicates synthetic generation via deep learning architecture."
              : isReal
              ? "CLEAR: Media entropy levels normal. No significant adversarial perturbations or facial manipulation artifacts detected."
              : "UNKNOWN: Insufficient tensor data to compute conclusive confidence matrix."}
          </p>

          {/* Detailed Telemetry Grid */}
          {details && (
            <div className="grid grid-cols-2 gap-2 mt-4">
              {Object.entries(details).map(([key, val]) => (
                <div key={key} className="bg-dark-950 border border-white/5 p-2 flex justify-between items-center">
                  <span className="text-[9px] font-mono text-white/40 uppercase">{key.replace(/_/g, ' ')}</span>
                  <span className="text-[10px] font-mono text-white font-bold">{val}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Frame-by-frame breakdown (Videos only) */}
      {result.frame_scores && result.frame_scores.length > 0 && (
        <div className="mt-10 pt-8 border-t border-white/5 relative z-10">
          <h3 className="text-sm font-semibold text-white/70 uppercase tracking-wider mb-4 flex items-center gap-2">
            <Film className="w-4 h-4" />
            Frame Analysis Timeline
          </h3>
          
          <div className="flex gap-4 overflow-x-auto pb-4 scrollbar-thin">
            {result.frame_scores.map((frame, idx) => {
              const isFrameFake = frame.verdict === 'FAKE'
              const frameColor = isFrameFake ? 'text-red-400' : 'text-green-400'
              const frameBg = isFrameFake ? 'bg-red-500/20 border-red-500/30' : 'bg-green-500/20 border-green-500/30'
              const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'
              
              return (
                <div key={idx} className="shrink-0 w-32 glass rounded-xl overflow-hidden flex flex-col group">
                  {/* Grad-CAM Heatmap */}
                  <div className="h-24 bg-dark-900 relative overflow-hidden">
                    <img 
                      src={`${API_URL}${frame.cam_path}`} 
                      alt={`Frame ${frame.frame_idx}`}
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500 opacity-80"
                      onError={(e) => { e.target.style.display = 'none' }}
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-dark-900 via-transparent to-transparent opacity-80" />
                    <span className="absolute bottom-1 left-2 text-[10px] font-mono text-white/70">
                      FRM {frame.frame_idx}
                    </span>
                  </div>
                  
                  {/* Frame Stats */}
                  <div className={`p-2 border-t flex flex-col items-center justify-center ${frameBg}`}>
                    <span className={`text-xs font-bold ${frameColor}`}>
                      {frame.verdict}
                    </span>
                    <span className="text-[10px] font-mono text-white/50 mt-0.5">
                      {Math.round(frame.fake_prob * 100)}% FAKE
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}
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
