import { useNavigate } from 'react-router-dom'
import {
  AlertTriangle, Activity, ShieldCheck, FolderOpen,
  CheckCircle, Cpu, Users, ArrowRight
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, BarChart, Bar
} from 'recharts'
import StatCard from '../components/StatCard'
import { PriorityBadge } from '../components/PriorityBadge'
import { PageLoader } from '../components/LoadingSpinner'
import { useApi } from '../hooks/useApi'
import { getDashboardSummary, getRecentAlerts, getAlertTimeseries } from '../services/api'
import { fmtAmount, fmtNumber } from '../utils/format'

export default function Dashboard() {
  const { data: summary, loading: sl } = useApi(getDashboardSummary)
  const { data: recent,  loading: rl } = useApi(getRecentAlerts)
  const { data: ts,      loading: tl } = useApi(getAlertTimeseries)
  const nav = useNavigate()

  if (sl || rl || tl) return <PageLoader />

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Overview</h1>
        <p className="text-sm text-gray-400 mt-1">Real-time AML alert monitoring dashboard</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Transactions" value={summary?.total_transactions ?? 0} icon={Activity} />
        <StatCard label="Total Accounts"     value={summary?.total_accounts ?? 0}     icon={Users} />
        <StatCard label="Total Alerts"       value={summary?.total_alerts ?? 0}       icon={AlertTriangle} color="text-amber-400" />
        <StatCard label="Fraud Labels"       value={summary?.fraud_label_count ?? 0}  icon={ShieldCheck}   color="text-red-400" />
      </div>

      {/* Priority + Case stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card border-red-500/20">
          <p className="text-2xl font-bold text-red-400">{fmtNumber(summary?.high_priority_count ?? 0)}</p>
          <p className="text-sm text-gray-400 mt-1">High Priority</p>
        </div>
        <div className="card border-amber-500/20">
          <p className="text-2xl font-bold text-amber-400">{fmtNumber(summary?.medium_priority_count ?? 0)}</p>
          <p className="text-sm text-gray-400 mt-1">Medium Priority</p>
        </div>
        <div className="card">
          <p className="text-2xl font-bold text-blue-400">{fmtNumber(summary?.open_cases ?? 0)}</p>
          <div className="flex items-center gap-1.5 mt-1">
            <FolderOpen size={13} className="text-gray-500" />
            <p className="text-sm text-gray-400">Open Cases</p>
          </div>
        </div>
        <div className="card">
          <p className="text-2xl font-bold text-green-400">{fmtNumber(summary?.resolved_cases ?? 0)}</p>
          <div className="flex items-center gap-1.5 mt-1">
            <CheckCircle size={13} className="text-gray-500" />
            <p className="text-sm text-gray-400">Resolved</p>
          </div>
        </div>
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Timeseries */}
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Alert Activity by Simulation Step</h3>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={ts ?? []}>
              <defs>
                <linearGradient id="aGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <XAxis dataKey="timestamp" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} />
              <Tooltip
                contentStyle={{ background: '#0f2040', border: '1px solid #162b55', fontSize: 12 }}
                labelFormatter={v => `Day ${v}`}
              />
              <Area type="monotone" dataKey="count" stroke="#3b82f6" fill="url(#aGrad)" strokeWidth={2} name="Alerts" />
              <Area type="monotone" dataKey="fraud_count" stroke="#ef4444" fill="none" strokeWidth={1.5} name="Fraud" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Priority distribution */}
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Priority Distribution</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={[
              { name: 'High',   count: summary?.high_priority_count ?? 0 },
              { name: 'Medium', count: summary?.medium_priority_count ?? 0 },
              { name: 'Low',    count: summary?.low_priority_count ?? 0 },
            ]}>
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#9ca3af' }} />
              <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} />
              <Tooltip contentStyle={{ background: '#0f2040', border: '1px solid #162b55', fontSize: 12 }} />
              <Bar dataKey="count" name="Alerts" radius={[4, 4, 0, 0]}
                fill="#3b82f6"
                label={false}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model status + Recent alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Model status */}
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <Cpu size={15} className="text-accent-400" />
            <h3 className="text-sm font-medium text-gray-300">Model Status</h3>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Status</span>
              <span className={summary?.model_status === 'loaded' ? 'text-green-400' : 'text-red-400'}>
                {summary?.model_status === 'loaded' ? '● Loaded' : '○ Not loaded'}
              </span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Model</span>
              <span className="text-gray-200">{summary?.model_name ?? '—'}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Version</span>
              <span className="text-gray-200">{summary?.model_version ?? '—'}</span>
            </div>
          </div>
        </div>

        {/* Recent high-priority */}
        <div className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-300">Recent High-Priority Alerts</h3>
            <button
              onClick={() => nav('/queue')}
              className="flex items-center gap-1 text-xs text-accent-400 hover:text-accent-300"
            >View all <ArrowRight size={12} /></button>
          </div>
          <div className="space-y-2">
            {(recent ?? []).slice(0, 5).map(a => (
              <div key={a.id}
                className="flex items-center justify-between text-sm py-1.5 border-b border-navy-700 last:border-0 cursor-pointer hover:text-white transition-colors"
                onClick={() => nav(`/queue/${a.id}`)}
              >
                <div className="flex items-center gap-3">
                  <PriorityBadge priority={a.priority} />
                  <span className="text-gray-300">TX #{a.tx_id}</span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-gray-400">{fmtAmount(a.tx_amount)}</span>
                  <span className="text-accent-400 font-mono text-xs">
                    {(a.risk_score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
            {(!recent || recent.length === 0) && (
              <p className="text-gray-500 text-sm text-center py-4">No high-priority alerts</p>
            )}
          </div>
        </div>
      </div>

      <p className="text-xs text-gray-600">
        ⓘ This is an academic prototype using AMLSim simulated data. Risk scores do not constitute legal findings.
      </p>
    </div>
  )
}
