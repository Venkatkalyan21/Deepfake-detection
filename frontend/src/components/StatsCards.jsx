import { Database, ShieldAlert, Cpu, Network } from 'lucide-react'

export default function StatsCards({ stats, loading }) {
  const cards = [
    { label: 'TOTAL_SCANS', value: stats?.total_scans || '0', icon: Database, color: 'text-primary-400', border: 'border-primary-500/30' },
    { label: 'DETECTED_ANOMALIES', value: stats?.total_fakes || '0', icon: ShieldAlert, color: 'text-danger-400', border: 'border-danger-500/30' },
    { label: 'TENSOR_NODES', value: '3', icon: Cpu, color: 'text-white/60', border: 'border-white/10' },
    { label: 'SYS_ACCURACY', value: '98.4%', icon: Network, color: 'text-success-400', border: 'border-success-500/30' },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {cards.map((card, idx) => {
        const Icon = card.icon
        return (
          <div key={idx} className={`stat-card ${card.border}`}>
            <div className="flex justify-between items-start mb-2">
              <Icon className={`w-4 h-4 ${card.color} opacity-70`} />
              <div className="text-[8px] font-mono text-white/20">NODE_{idx+1}</div>
            </div>
            {loading ? (
              <div className="h-6 bg-white/5 animate-pulse w-12 mb-1" />
            ) : (
              <h4 className={`text-xl font-mono font-bold ${card.color} tracking-wider`}>
                {card.value}
              </h4>
            )}
            <p className="text-[9px] font-mono text-white/40 mt-1">{card.label}</p>
          </div>
        )
      })}
    </div>
  )
}
