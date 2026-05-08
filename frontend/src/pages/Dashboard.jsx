import { useState, useEffect } from 'react'
import UploadZone from '../components/UploadZone'
import ResultPanel from '../components/ResultPanel'
import StatsCards from '../components/StatsCards'
import { detectVideo, detectImage, detectAudio, getHistory } from '../api/deepshield'

export default function Dashboard() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [stats, setStats] = useState(null)

  useEffect(() => {
    // Quick load of recent stats
    getHistory(100).then(data => {
      if (data && data.history) {
        const total_scans = data.total
        const total_fakes = data.history.filter(h => h.verdict === 'FAKE').length
        setStats({ total_scans, total_fakes })
      }
    }).catch(console.error)
  }, [result]) // Refresh stats when a new result comes in

  const handleAnalyze = async (file, mode, onProgress) => {
    setLoading(true)
    setResult(null)
    try {
      let data
      if (mode === 'video') data = await detectVideo(file, onProgress)
      else if (mode === 'image') data = await detectImage(file, onProgress)
      else if (mode === 'audio') data = await detectAudio(file, onProgress)
      
      // Inject modality into result for UI
      if (data) {
        data.modality = mode
        setResult(data)
      }
    } catch (err) {
      console.error(err)
      alert(err.response?.data?.detail || err.message || 'Detection failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Hero Section */}
      <div className="text-center max-w-3xl mx-auto mb-12">
        <h1 className="text-5xl md:text-6xl font-black text-white mb-6 tracking-tight">
          Detect <span className="text-gradient">Deepfakes</span> with Multi-Modal AI
        </h1>
        <p className="text-lg text-white/60 leading-relaxed">
          DeepShield analyzes videos, images, and audio using ensemble AI models (EfficientNetV2, MTCNN, Wav2Vec2) to uncover deepfakes and digital manipulation with high precision.
        </p>
      </div>

      <StatsCards stats={stats} loading={!stats && loading} />

      <div className="grid lg:grid-cols-12 gap-8">
        <div className="lg:col-span-5">
          <UploadZone onAnalyze={handleAnalyze} loading={loading} />
        </div>
        <div className="lg:col-span-7">
          <ResultPanel result={result} loading={loading} />
        </div>
      </div>
    </div>
  )
}
