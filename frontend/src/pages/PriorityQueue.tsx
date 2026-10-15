import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Filter, Download, ChevronLeft, ChevronRight, ArrowUpDown } from 'lucide-react'
import { PriorityBadge, StatusBadge } from '../components/PriorityBadge'
import { PageLoader } from '../components/LoadingSpinner'
import { useApi } from '../hooks/useApi'
import { getAlerts, getExportUrl } from '../services/api'
import type { AlertFilters } from '../services/api'
import { fmtAmount, fmtTs } from '../utils/format'

const PRIORITIES = ['', 'HIGH', 'MEDIUM', 'LOW']
const SORT_OPTS  = [
  { value: 'risk_score', label: 'Risk Score' },
  { value: 'timestamp',  label: 'Timestamp'  },
  { value: 'tx_amount',  label: 'Amount'     },
]

export default function PriorityQueue() {
  const nav = useNavigate()
  const [filters, setFilters] = useState<AlertFilters>({
    page: 1, size: 25, sort_by: 'risk_score', sort_dir: 'desc'
  })

  const fetcher = useCallback(() => getAlerts(filters), [JSON.stringify(filters)])
  const { data, loading } = useApi(fetcher, [JSON.stringify(filters)])

  const set = (patch: Partial<AlertFilters>) =>
    setFilters(f => ({ ...f, ...patch, page: 1 }))

  const exportUrl = getExportUrl({
    ...(filters.priority ? { priority: filters.priority } : {}),
    ...(filters.search   ? { search:   filters.search   } : {}),
  })

  return (
    <div className="p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-white">Investigation Priority Queue</h1>
          <p className="text-sm text-gray-400 mt-1">
            {data ? `${data.total.toLocaleString()} alerts` : 'Loading…'}
          </p>
        </div>
        <a href={exportUrl} download className="btn-secondary flex items-center gap-2">
          <Download size={14} /> Export CSV
        </a>
      </div>

      {/* Filters */}
      <div className="card flex flex-wrap gap-3 items-end">
        <div className="flex-1 min-w-48">
          <label className="text-xs text-gray-400 mb-1 block">Search TX/Account</label>
          <div className="relative">
            <Search size={13} className="absolute left-2.5 top-2.5 text-gray-500" />
            <input
              className="w-full bg-navy-700 border border-navy-600 rounded-lg pl-8 pr-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent-500"
              placeholder="Transaction ID…"
              value={filters.search ?? ''}
              onChange={e => set({ search: e.target.value || undefined })}
            />
          </div>
        </div>

        <div>
          <label className="text-xs text-gray-400 mb-1 block">Priority</label>
          <select
            className="bg-navy-700 border border-navy-600 rounded-lg px-3 py-2 text-sm text-white"
            value={filters.priority ?? ''}
            onChange={e => set({ priority: e.target.value || undefined })}
          >
            {PRIORITIES.map(p => <option key={p} value={p}>{p || 'All'}</option>)}
          </select>
        </div>

        <div>
          <label className="text-xs text-gray-400 mb-1 block">Sort by</label>
          <select
            className="bg-navy-700 border border-navy-600 rounded-lg px-3 py-2 text-sm text-white"
            value={filters.sort_by}
            onChange={e => set({ sort_by: e.target.value })}
          >
            {SORT_OPTS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </div>

        <div>
          <label className="text-xs text-gray-400 mb-1 block">Dir</label>
          <select
            className="bg-navy-700 border border-navy-600 rounded-lg px-3 py-2 text-sm text-white"
            value={filters.sort_dir}
            onChange={e => set({ sort_dir: e.target.value })}
          >
            <option value="desc">Desc</option>
            <option value="asc">Asc</option>
          </select>
        </div>

        <button className="btn-secondary flex items-center gap-1.5"
          onClick={() => setFilters({ page: 1, size: 25, sort_by: 'risk_score', sort_dir: 'desc' })}>
          <Filter size={13} /> Reset
        </button>
      </div>

      {/* Table */}
      <div className="card p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-navy-700 text-xs text-gray-400">
                <th className="px-4 py-3 text-left">Alert ID</th>
                <th className="px-4 py-3 text-left">TX ID</th>
                <th className="px-4 py-3 text-left">Sender</th>
                <th className="px-4 py-3 text-left">Receiver</th>
                <th className="px-4 py-3 text-right">
                  <button className="flex items-center gap-1 ml-auto"
                    onClick={() => set({ sort_by: 'tx_amount', sort_dir: filters.sort_dir === 'desc' ? 'asc' : 'desc' })}>
                    Amount <ArrowUpDown size={11} />
                  </button>
                </th>
                <th className="px-4 py-3 text-left">Step</th>
                <th className="px-4 py-3 text-right">
                  <button className="flex items-center gap-1 ml-auto"
                    onClick={() => set({ sort_by: 'risk_score', sort_dir: filters.sort_dir === 'desc' ? 'asc' : 'desc' })}>
                    Risk <ArrowUpDown size={11} />
                  </button>
                </th>
                <th className="px-4 py-3 text-left">Priority</th>
                <th className="px-4 py-3 text-left">Status</th>
                <th className="px-4 py-3 text-left">Type</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={10} className="text-center py-12 text-gray-500">Loading…</td></tr>
              ) : data?.items.length === 0 ? (
                <tr><td colSpan={10} className="text-center py-12 text-gray-500">No alerts match filters</td></tr>
              ) : data?.items.map(a => (
                <tr
                  key={a.id}
                  className="border-b border-navy-700/50 hover:bg-navy-800/50 cursor-pointer transition-colors"
                  onClick={() => nav(`/queue/${a.id}`)}
                >
                  <td className="px-4 py-3 text-gray-300 font-mono text-xs">#{a.id}</td>
                  <td className="px-4 py-3 text-gray-300 font-mono text-xs">{a.tx_id}</td>
                  <td className="px-4 py-3 text-gray-400 font-mono text-xs">{a.sender_account_id}</td>
                  <td className="px-4 py-3 text-gray-400 font-mono text-xs">{a.receiver_account_id}</td>
                  <td className="px-4 py-3 text-right text-gray-200">{fmtAmount(a.tx_amount)}</td>
                  <td className="px-4 py-3 text-gray-400">{fmtTs(a.timestamp)}</td>
                  <td className="px-4 py-3 text-right font-mono">
                    <span className={a.risk_score >= 0.7 ? 'text-red-400' : a.risk_score >= 0.4 ? 'text-amber-400' : 'text-green-400'}>
                      {(a.risk_score * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3"><PriorityBadge priority={a.priority} /></td>
                  <td className="px-4 py-3">
                    {a.case_status ? <StatusBadge status={a.case_status} /> : <span className="text-gray-600 text-xs">—</span>}
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs">{a.alert_type ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {data && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-navy-700 text-xs text-gray-400">
            <span>
              {((filters.page! - 1) * filters.size! + 1).toLocaleString()}–
              {Math.min(filters.page! * filters.size!, data.total).toLocaleString()} of {data.total.toLocaleString()}
            </span>
            <div className="flex gap-2">
              <button
                className="btn-secondary py-1 px-2 disabled:opacity-40"
                disabled={filters.page === 1}
                onClick={() => setFilters(f => ({ ...f, page: f.page! - 1 }))}
              ><ChevronLeft size={14} /></button>
              <button
                className="btn-secondary py-1 px-2 disabled:opacity-40"
                disabled={filters.page! * filters.size! >= data.total}
                onClick={() => setFilters(f => ({ ...f, page: f.page! + 1 }))}
              ><ChevronRight size={14} /></button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
