import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Film, Image, Mic, X, FileVideo, FileImage, FileAudio } from 'lucide-react'

const MODES = [
  { id: 'video', label: 'Video', icon: Film, accept: { 'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'] }, color: 'from-blue-500 to-cyan-500', bg: 'bg-blue-500/10', border: 'border-blue-500/30' },
  { id: 'image', label: 'Image', icon: Image, accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp', '.bmp'] }, color: 'from-purple-500 to-pink-500', bg: 'bg-purple-500/10', border: 'border-purple-500/30' },
  { id: 'audio', label: 'Audio', icon: Mic,   accept: { 'audio/*': ['.wav', '.mp3', '.flac', '.ogg', '.m4a'] }, color: 'from-amber-500 to-orange-500', bg: 'bg-amber-500/10', border: 'border-amber-500/30' },
]

const FileIcon = ({ mode }) => {
  if (mode === 'video') return <FileVideo className="w-10 h-10 text-blue-400" />
  if (mode === 'image') return <FileImage className="w-10 h-10 text-purple-400" />
  return <FileAudio className="w-10 h-10 text-amber-400" />
}

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

  const handleAnalyze = () => {
    if (file) onAnalyze(file, activeMode, setUploadPct)
  }

  const formatSize = bytes => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="glass p-6 rounded-2xl animate-fade-in">
      {/* Mode tabs */}
      <div className="flex gap-2 mb-6">
        {MODES.map(m => {
          const Icon = m.icon
          const active = m.id === activeMode
          return (
            <button
              key={m.id}
              onClick={() => { setActiveMode(m.id); setFile(null) }}
              disabled={loading}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 flex-1 justify-center
                ${active
                  ? `bg-gradient-to-r ${m.color} text-white shadow-lg`
                  : 'bg-white/5 text-white/50 hover:text-white hover:bg-white/10'
                }`}
            >
              <Icon className="w-4 h-4" />
              {m.label}
            </button>
          )
        })}
      </div>

      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={`relative border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-300
          ${isDragActive ? `${mode.bg} ${mode.border} scale-[1.02]` : 'border-white/10 hover:border-white/25 hover:bg-white/3'}
          ${loading ? 'opacity-50 cursor-not-allowed' : ''}
          ${file ? 'bg-white/3' : ''}`}
      >
        <input {...getInputProps()} />

        {file ? (
          <div className="flex flex-col items-center gap-3 animate-fade-in">
            <FileIcon mode={activeMode} />
            <div>
              <p className="text-white font-medium text-sm">{file.name}</p>
              <p className="text-white/40 text-xs mt-0.5">{formatSize(file.size)}</p>
            </div>
            {!loading && (
              <button
                onClick={e => { e.stopPropagation(); setFile(null); setUploadPct(0) }}
                className="absolute top-3 right-3 p-1.5 rounded-lg bg-white/5 hover:bg-red-500/20 text-white/40 hover:text-red-400 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3 text-white/40">
            <div className={`w-16 h-16 rounded-2xl ${mode.bg} ${mode.border} border flex items-center justify-center`}>
              <Upload className="w-7 h-7 text-white/60" />
            </div>
            <div>
              <p className="text-white/70 font-medium">
                {isDragActive ? 'Drop your file here' : `Drag & drop your ${mode.label.toLowerCase()}`}
              </p>
              <p className="text-xs mt-1">or click to browse</p>
            </div>
            <p className="text-xs text-white/25">
              {Object.values(mode.accept)[0].join(', ')}
            </p>
          </div>
        )}
      </div>

      {/* Upload progress */}
      {loading && uploadPct > 0 && uploadPct < 100 && (
        <div className="mt-4">
          <div className="flex justify-between text-xs text-white/50 mb-1">
            <span>Uploading…</span><span>{uploadPct}%</span>
          </div>
          <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div
              className={`h-full bg-gradient-to-r ${mode.color} rounded-full transition-all duration-300`}
              style={{ width: `${uploadPct}%` }}
            />
          </div>
        </div>
      )}

      {/* Analyze button */}
      <button
        onClick={handleAnalyze}
        disabled={!file || loading}
        className={`mt-4 w-full py-3.5 rounded-xl font-semibold text-sm transition-all duration-200
          ${file && !loading
            ? `bg-gradient-to-r ${mode.color} text-white hover:opacity-90 hover:shadow-lg active:scale-[0.98] shadow-lg`
            : 'bg-white/5 text-white/30 cursor-not-allowed'
          }`}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
            </svg>
            Analyzing…
          </span>
        ) : `Analyze ${mode.label}`}
      </button>
    </div>
  )
}
