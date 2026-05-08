import { formatDistanceToNow } from 'date-fns'
import { FileVideo, FileImage, FileAudio, AlertTriangle, CheckCircle } from 'lucide-react'

export default function HistoryTable({ history, loading }) {
  if (loading) {
    return (
      <div className="glass rounded-2xl overflow-hidden p-8 flex flex-col items-center justify-center min-h-[300px]">
        <div className="w-8 h-8 border-4 border-white/10 border-t-primary-500 rounded-full animate-spin mb-4" />
        <p className="text-white/50 text-sm">Loading history from database...</p>
      </div>
    )
  }

  if (!history || history.length === 0) {
    return (
      <div className="glass rounded-2xl p-8 text-center border-dashed border-2 border-white/5">
        <p className="text-white/40 text-sm">No scans found in history.</p>
      </div>
    )
  }

  const getModalityIcon = (modality) => {
    switch (modality?.toLowerCase()) {
      case 'video': return <FileVideo className="w-4 h-4 text-blue-400" />
      case 'image': return <FileImage className="w-4 h-4 text-purple-400" />
      case 'audio': return <FileAudio className="w-4 h-4 text-amber-400" />
      default: return null
    }
  }

  return (
    <div className="glass rounded-2xl overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-white/5 bg-white/5">
              <th className="p-4 text-xs font-semibold text-white/50 uppercase tracking-wider">File</th>
              <th className="p-4 text-xs font-semibold text-white/50 uppercase tracking-wider">Type</th>
              <th className="p-4 text-xs font-semibold text-white/50 uppercase tracking-wider">Verdict</th>
              <th className="p-4 text-xs font-semibold text-white/50 uppercase tracking-wider">Confidence</th>
              <th className="p-4 text-xs font-semibold text-white/50 uppercase tracking-wider">Time</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {history.map((item, idx) => {
              const isFake = item.verdict?.toUpperCase() === 'FAKE'
              const dateObj = new Date(item.timestamp)
              const timeAgo = isNaN(dateObj) ? 'Unknown' : formatDistanceToNow(dateObj, { addSuffix: true })
              
              return (
                <tr key={item._id || idx} className="hover:bg-white/5 transition-colors">
                  <td className="p-4">
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-white truncate max-w-[200px]" title={item.filename}>
                        {item.filename || 'Unknown'}
                      </span>
                      <span className="text-xs text-white/30 truncate max-w-[200px]" title={item.session_id}>
                        {item.session_id}
                      </span>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      {getModalityIcon(item.modality)}
                      <span className="text-sm text-white/70 capitalize">{item.modality}</span>
                    </div>
                  </td>
                  <td className="p-4">
                    <span className={isFake ? 'badge-fake' : 'badge-real'}>
                      {isFake ? <AlertTriangle className="w-3 h-3" /> : <CheckCircle className="w-3 h-3" />}
                      {item.verdict?.toUpperCase()}
                    </span>
                  </td>
                  <td className="p-4">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full ${isFake ? 'bg-red-500' : 'bg-green-500'}`}
                          style={{ width: `${Math.round((item.confidence || 0) * 100)}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-white">
                        {Math.round((item.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </td>
                  <td className="p-4 text-sm text-white/50">
                    {timeAgo}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
