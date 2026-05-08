import { useState, useEffect } from 'react'
import { getHistory } from '../api/deepshield'
import HistoryTable from '../components/HistoryTable'
import { Clock } from 'lucide-react'

export default function History() {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getHistory(50)
      .then(data => {
        setHistory(data?.history || [])
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      <div className="flex items-center gap-4 border-b border-white/5 pb-6">
        <div className="p-3 bg-white/5 rounded-xl border border-white/10">
          <Clock className="w-6 h-6 text-primary-400" />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-white">Scan History</h1>
          <p className="text-white/50 text-sm mt-1">Review past deepfake detection results stored in the database.</p>
        </div>
      </div>

      <HistoryTable history={history} loading={loading} />
    </div>
  )
}
