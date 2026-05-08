import { Shield, Activity, Users, Database } from 'lucide-react'

export default function StatsCards({ stats, loading }) {
  const cards = [
    { label: 'Total Scans', value: stats?.total_scans || '0', icon: Database, color: 'text-primary-400', bg: 'bg-primary-500/10' },
    { label: 'Deepfakes Found', value: stats?.total_fakes || '0', icon: Shield, color: 'text-red-400', bg: 'bg-red-500/10' },
    { label: 'System Accuracy', value: '98.4%', icon: Activity, color: 'text-green-400', bg: 'bg-green-500/10' },
    { label: 'Active Models', value: '3', icon: Users, color: 'text-purple-400', bg: 'bg-purple-500/10' },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {cards.map((card, idx) => {
        const Icon = card.icon
        return (
          <div key={idx} className="stat-card">
            <div className="flex items-start justify-between mb-2">
              <div className={`p-2 rounded-lg ${card.bg}`}>
                <Icon className={`w-5 h-5 ${card.color}`} />
              </div>
            </div>
            {loading ? (
              <div className="h-8 bg-white/5 rounded animate-pulse w-16 mb-1" />
            ) : (
              <h4 className="text-2xl font-bold text-white">{card.value}</h4>
            )}
            <p className="text-xs font-medium text-white/50 uppercase tracking-wider">{card.label}</p>
          </div>
        )
      })}
    </div>
  )
}
