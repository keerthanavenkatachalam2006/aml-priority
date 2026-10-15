import { useState } from 'react'
import { RefreshCw, CheckCircle, XCircle, Database, Cpu } from 'lucide-react'
import { PageLoader } from '../components/LoadingSpinner'
import { useApi } from '../hooks/useApi'
import { getModelStatus, getDatasetInfo, reloadModel } from '../services/api'
import { fmtNumber, fmtPct } from '../utils/format'

export default function ModelManagement() {
  const { data: status, loading: sl, refetch: refetchStatus } = useApi(getModelStatus)
  const { data: dsInfo, loading: dl } = useApi(getDatasetInfo)
  const [reloading, setReloading] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  const handleReload = async () => {
    setReloading(true)
    try {
      await reloadModel()
      await refetchStatus()
      setMsg('Model reloaded successfully')
    } catch { setMsg('Reload failed') }
    setReloading(false)
    setTimeout(() => setMsg(null), 3000)
  }

  if (sl || dl) return <PageLoader />

  const split = dsInfo?.split_metadata
  const txInfo = dsInfo?.transactions

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-semibold text-white">Model & Data Management</h1>

      {msg && (
        <div className="bg-green-500/10 border border-green-500/30 text-green-400 text-sm px-4 py-2 rounded-lg">
          {msg}
        </div>
      )}

      {/* Model status */}
      <div className="card space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-accent-400" />
            <h2 className="text-sm font-medium text-gray-300">Model Status</h2>
          </div>
          <button className="btn-secondary flex items-center gap-1.5 text-xs" onClick={handleReload} disabled={reloading}>
            <RefreshCw size={12} className={reloading ? 'animate-spin' : ''} /> Reload
          </button>
        </div>

        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-gray-500 text-xs">Status</p>
            <div className="flex items-center gap-1.5 mt-1">
              {status?.loaded
                ? <><CheckCircle size={13} className="text-green-400" /><span className="text-green-400">Loaded</span></>
                : <><XCircle size={13} className="text-red-400" /><span className="text-red-400">Not loaded</span></>
              }
            </div>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Model Name</p>
            <p className="text-gray-200 mt-1">{status?.model_name ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Version</p>
            <p className="text-gray-200 mt-1">{status?.version ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Threshold</p>
            <p className="text-gray-200 mt-1">{status?.threshold != null ? `${(status.threshold*100).toFixed(1)}%` : '—'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Feature Count</p>
            <p className="text-gray-200 mt-1">{status?.feature_count ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Prediction Unit</p>
            <p className="text-gray-200 mt-1">{status?.prediction_unit ?? '—'}</p>
          </div>
          <div>
            <p className="text-gray-500 text-xs">Sample Mode</p>
            <p className={`mt-1 ${status?.sample_mode ? 'text-amber-400' : 'text-gray-200'}`}>
              {status?.sample_mode ? 'Yes (sample)' : 'No (full dataset)'}
            </p>
          </div>
          {status?.error && (
            <div className="col-span-2">
              <p className="text-gray-500 text-xs">Error</p>
              <p className="text-red-400 mt-1 text-xs">{status.error}</p>
            </div>
          )}
        </div>

        {/* Feature schema */}
        {status?.feature_cols && status.feature_cols.length > 0 && (
          <div>
            <p className="text-xs text-gray-400 mb-2">Feature Schema ({status.feature_cols.length} features)</p>
            <div className="flex flex-wrap gap-1.5">
              {status.feature_cols.map(f => (
                <span key={f} className="text-xs bg-navy-700 text-gray-300 px-2 py-0.5 rounded font-mono">{f}</span>
              ))}
            </div>
          </div>
        )}

        {/* Val metrics */}
        {status?.val_metrics && (
          <div>
            <p className="text-xs text-gray-400 mb-2">Validation Metrics</p>
            <div className="grid grid-cols-4 gap-3">
              {['f1','pr_auc','precision','recall'].map(k => (
                <div key={k} className="bg-navy-700/50 rounded p-2 text-center">
                  <p className="text-sm font-semibold text-accent-400">
                    {fmtPct(status.val_metrics![k] ?? 0)}
                  </p>
                  <p className="text-xs text-gray-500 mt-0.5">{k.replace('_',' ').toUpperCase()}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Dataset info */}
      <div className="card space-y-4">
        <div className="flex items-center gap-2">
          <Database size={16} className="text-accent-400" />
          <h2 className="text-sm font-medium text-gray-300">Dataset Information</h2>
        </div>

        {txInfo && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-500 text-xs">Total Transactions</p>
              <p className="text-gray-200 mt-1">{fmtNumber(txInfo.total_rows ?? 0)}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Fraud Rows</p>
              <p className="text-red-400 mt-1">{fmtNumber(txInfo.fraud_rows ?? 0)}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Fraud Rate</p>
              <p className="text-gray-200 mt-1">{txInfo.fraud_rate_pct?.toFixed(3) ?? '—'}%</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs">Timestamp Range</p>
              <p className="text-gray-200 mt-1">Steps {txInfo.timestamp_min ?? '?'}–{txInfo.timestamp_max ?? '?'}</p>
            </div>
          </div>
        )}

        {split && (
          <div>
            <p className="text-xs text-gray-400 mb-2">Chronological Split</p>
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: 'Train', rows: split.train_rows, fraud: split.train_fraud_count, range: `Step 0–${split.train_end_timestamp}` },
                { label: 'Val',   rows: split.val_rows,   fraud: split.val_fraud_count,   range: `Step ${split.train_end_timestamp+1}–${split.val_end_timestamp}` },
                { label: 'Test',  rows: split.test_rows,  fraud: split.test_fraud_count,  range: `Step ${split.val_end_timestamp+1}–199` },
              ].map(s => (
                <div key={s.label} className="bg-navy-700/50 rounded p-3">
                  <p className="text-sm font-medium text-gray-300">{s.label}</p>
                  <p className="text-xs text-gray-500 mt-0.5">{s.range}</p>
                  <p className="text-sm text-gray-200 mt-1">{fmtNumber(s.rows)} rows</p>
                  <p className="text-xs text-red-400">{fmtNumber(s.fraud)} fraud</p>
                </div>
              ))}
            </div>
            <p className="text-xs text-gray-600 mt-2">
              Chronological split prevents future data from leaking into model training.
              Scaler fitted on training data only and applied to val/test.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
