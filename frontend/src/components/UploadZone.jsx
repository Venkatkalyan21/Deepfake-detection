import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Crosshair, Film, Image as ImageIcon, Mic, X } from 'lucide-react'

const MODES = [
  { id: 'video', label: 'VIDEO', icon: Film, accept: { 'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'] } },
  { id: 'image', label: 'IMAGE', icon: ImageIcon, accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] } },
  { id: 'audio', label: 'AUDIO', icon: Mic, accept: { 'audio/*': ['.wav', '.mp3', '.flac', '.ogg', '.m4a'] } },
]

export default function UploadZone({ onAnalyze, loading }) {
  const [activeMode, setActiveMode] = useState('video')
  const [file, setFile] = useState(null)
  const [uploadPct, setUploadPct] = useState(0)

  const mode = MODES.find(m => m.id === activeMode)

  const onDrop = useCallback(accepted => {
    if (accepted[0]) setFile(accepted[0])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: mode.accept,
    maxFiles: 1,
    disabled: loading,
  })

  return (
    <div className="glass p-5 flex flex-col h-full animate-fade-in relative">
      <div className="absolute top-0 right-0 p-2 text-[10px] font-mono text-primary-400/30">
        TARGET_ACQUISITION_MOD
      </div>

      <h2 className="text-sm font-mono font-bold text-white mb-4 uppercase tracking-widest border-b border-primary-500/20 pb-2">
        Data Stream Input
      </h2>

      {/* Mode selection */}
      <div className="grid grid-cols-3 gap-2 mb-6">
        {MODES.map(m => {
          const Icon = m.icon
          const active = m.id === activeMode
          return (
            <button
              key={m.id}
              onClick={() => { setActiveMode(m.id); setFile(null) }}
              disabled={loading}
              className={`flex flex-col items-center justify-center gap-2 p-3 border font-mono text-[10px] transition-all
                ${active 
                  ? 'bg-primary-500/20 border-primary-400 text-primary-400 shadow-[0_0_10px_rgba(0,240,255,0.2)]' 
                  : 'bg-dark-900 border-white/5 text-white/40 hover:border-primary-500/30'}`}
            >
              <Icon className="w-4 h-4" />
              {m.label}
            </button>
          )
        })}
      </div>

      {/* Target Zone */}
      <div
        {...getRootProps()}
        className={`flex-1 relative border border-dashed flex flex-col items-center justify-center p-6 text-center transition-all cursor-pointer min-h-[200px]
          ${isDragActive ? 'border-primary-400 bg-primary-500/5' : 'border-white/20 hover:border-primary-500/50 bg-dark-950'}
          ${loading ? 'opacity-50 pointer-events-none' : ''}`}
      >
        <input {...getInputProps()} />

        {/* Crosshair accents */}
        <div className="absolute top-2 left-2 w-3 h-3 border-t border-l border-primary-500/50" />
        <div className="absolute top-2 right-2 w-3 h-3 border-t border-r border-primary-500/50" />
        <div className="absolute bottom-2 left-2 w-3 h-3 border-b border-l border-primary-500/50" />
        <div className="absolute bottom-2 right-2 w-3 h-3 border-b border-r border-primary-500/50" />

        {file ? (
          <div className="relative z-10 animate-fade-in flex flex-col items-center">
            <Crosshair className="w-8 h-8 text-primary-400 mb-3 animate-spin-slow" />
            <p className="font-mono text-xs text-white truncate max-w-full px-4">{file.name}</p>
            <p className="font-mono text-[10px] text-primary-400/60 mt-1">{(file.size/1024/1024).toFixed(2)} MB</p>
            {!loading && (
              <button 
                onClick={e => { e.stopPropagation(); setFile(null) }}
                className="mt-4 px-3 py-1 bg-danger-500/10 text-danger-400 border border-danger-500/30 text-[10px] font-mono hover:bg-danger-500/20"
              >
                ABORT
              </button>
            )}
          </div>
        ) : (
          <div className="relative z-10 flex flex-col items-center opacity-60">
            <Crosshair className="w-8 h-8 text-white mb-3" />
            <p className="font-mono text-xs text-white uppercase tracking-wider">Awaiting Target Payload</p>
            <p className="font-mono text-[9px] text-white/50 mt-2">DRAG & DROP OR CLICK TO LOCATE</p>
          </div>
        )}
      </div>

      <button
        onClick={() => file && !loading && onAnalyze(file, activeMode, setUploadPct)}
        disabled={!file || loading}
        className={`mt-6 w-full py-3 font-mono text-xs tracking-widest font-bold uppercase transition-all
          ${file && !loading
            ? 'bg-primary-500/20 text-primary-400 border border-primary-500 hover:bg-primary-500/30 hover:shadow-[0_0_15px_rgba(0,240,255,0.4)]'
            : 'bg-dark-950 border border-white/10 text-white/20 cursor-not-allowed'}`}
      >
        {loading ? 'EXECUTING ANALYSIS...' : 'ENGAGE DETECTOR'}
      </button>
    </div>
  )
}
