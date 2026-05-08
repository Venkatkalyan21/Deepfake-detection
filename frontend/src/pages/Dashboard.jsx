import { useState, useEffect } from 'react'
import UploadZone from '../components/UploadZone'
import ResultPanel from '../components/ResultPanel'
import StatsCards from '../components/StatsCards'
import { detectVideo, detectImage, detectAudio, getHistory } from '../api/deepshield'

export default function Dashboard() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [stats, setStats] = useState(null)
  const [logs, setLogs] = useState([])

  useEffect(() => {
    getHistory(100).then(data => {
      if (data && data.history) {
        const total_scans = data.total
        const total_fakes = data.history.filter(h => h.verdict === 'FAKE').length
        setStats({ total_scans, total_fakes })
      }
    }).catch(console.error)
  }, [result])

  const addLog = (msg) => {
    setLogs(prev => [...prev.slice(-4), `[${new Date().toISOString().split('T')[1].slice(0,-1)}] ${msg}`])
  }

  const handleAnalyze = async (file, mode, onProgress) => {
    setLoading(true)
    setResult(null)
    setLogs([])
    addLog(`INITIATING SCAN: ${file.name}`)
    addLog(`MODALITY: ${mode.toUpperCase()} | SIZE: ${(file.size/1024/1024).toFixed(2)}MB`)
    
    // Simulate some backend log streaming
    const logInterval = setInterval(() => {
      const messages = [
        "Extracting tensor features...",
        "Running MTCNN face localization...",
        "Computing cross-entropy loss...",
        "Analyzing pixel gradients...",
        "Querying EfficientNetV2-S..."
      ]
      addLog(messages[Math.floor(Math.random() * messages.length)])
    }, 1500)

    try {
      let data
      if (mode === 'video') data = await detectVideo(file, onProgress)
      else if (mode === 'image') data = await detectImage(file, onProgress)
      else if (mode === 'audio') data = await detectAudio(file, onProgress)
      
      clearInterval(logInterval)
      addLog(`SCAN COMPLETE. VERDICT: ${data.verdict}`)
      
      if (data) {
        data.modality = mode
        setResult(data)
      }
    } catch (err) {
      clearInterval(logInterval)
      addLog(`ERROR: ${err.message}`)
      console.error(err)
      alert(err.response?.data?.detail || err.message || 'Detection failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      
      {/* Dashboard Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end border-b border-primary-500/20 pb-4 mb-6">
        <div>
          <h1 className="text-3xl font-mono font-bold text-white tracking-tight uppercase flex items-center gap-3">
            <span className="w-3 h-8 bg-primary-500 animate-pulse" />
            Telemetric Analysis
          </h1>
          <p className="text-sm font-mono text-primary-400/60 mt-1">
            SYS.STATUS: ONLINE | DEEPFAKE.DETECTION.CORE.V2
          </p>
        </div>
        
        {/* Fake Terminal Logs */}
        <div className="mt-4 md:mt-0 glass p-3 w-full md:w-96 h-24 overflow-hidden flex flex-col justify-end">
          {logs.length === 0 ? (
            <div className="text-xs font-mono text-primary-400/30">Awaiting input stream...</div>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="text-[10px] font-mono text-primary-400/80 tracking-wider truncate">
                {log}
              </div>
            ))
          )}
        </div>
      </div>

      <StatsCards stats={stats} loading={!stats && loading} />

      <div className="grid lg:grid-cols-12 gap-6 mt-6">
        <div className="lg:col-span-4">
          <UploadZone onAnalyze={handleAnalyze} loading={loading} />
        </div>
        <div className="lg:col-span-8">
          <ResultPanel result={result} loading={loading} />
        </div>
      </div>
    </div>
  )
}
