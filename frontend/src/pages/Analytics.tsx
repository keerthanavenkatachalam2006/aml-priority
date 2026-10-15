import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, ScatterChart, Scatter, CartesianGrid, Legend
} from 'recharts'
import { PageLoader } from '../components/LoadingSpinner'
import { useApi } from '../hooks/useApi'
import { getModelMetrics, getAnalyticsSummary } from '../services/api'
import { fmtPct, fmtNumber } from '../utils/format'

function MetricCard({ label, value, color = 'text-accent-400' }: { label: string; value: string; color?: string }) {
  return (
    <div className="card text-center">
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      <p className="text-xs text-gray-400 mt-1">{label}</p>
    </div>
  )
}

export default function Analytics() {
  const { data: metrics, loading: ml } = useApi(getModelMetrics)
  const { data: summary, loading: sl } = useApi(getAnalyticsSummary)

  if (ml || sl) return <PageLoader />
  if (!metrics)  return (
    <div className="p-6">
      <div className="card text-center py-12">
        <p className="text-gray-400">Evaluation report not available.</p>
        <p className="text-gray-500 text-sm mt-2">Run <code className="text-accent-400">python ml/src/evaluate.py</code> first.</p>
      </div>
    </div>
  )

  const cm  = metrics.confusion_matrix
  const fi  = metrics.feature_importance
    ? Object.entries(metrics.feature_importance).sort((a,b)=>b[1]-a[1]).slice(0,12)
    : []

  const sd  = metrics.score_distribution
  const sdData = sd ? sd.bins.map((b, i) => ({
    bin: `${(b*100).toFixed(0)}–${((sd.bins[i+1] ?? 1)*100).toFixed(0)}%`,
    total: sd.counts[i],
    fraud: sd.fraud_counts[i],
  })) : []

  const tt = (metrics.threshold_table ?? []).slice(0, 18)
  interface ModelComp { model: string; val_f1: number; val_pr_auc: number; val_recall?: number; train_time_s?: number; selected?: boolean }
  const modelComp: ModelComp[] = summary?.model_comparison ?? []

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Analytics</h1>
        <p className="text-sm text-gray-400 mt-1">
          Test-set results — {metrics.model_name}
          {metrics.sample_mode && <span className="ml-2 text-amber-400 text-xs">(sample mode)</span>}
        </p>
      </div>

      {/* Core metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Precision"  value={fmtPct(metrics.test_precision)} color="text-blue-400" />
        <MetricCard label="Recall"     value={fmtPct(metrics.test_recall)}    color="text-green-400" />
        <MetricCard label="F1-Score"   value={fmtPct(metrics.test_f1)}        color="text-accent-400" />
        <MetricCard label="PR-AUC"     value={fmtPct(metrics.test_pr_auc)}    color="text-purple-400" />
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="ROC-AUC"          value={fmtPct(metrics.test_roc_auc)}         />
        <MetricCard label="False Positive Rate" value={fmtPct(metrics.false_positive_rate)} color="text-red-400" />
        <MetricCard label="True Positives"   value={fmtNumber(cm.tp)}  color="text-green-400" />
        <MetricCard label="False Positives"  value={fmtNumber(cm.fp)}  color="text-red-400"   />
      </div>

      {/* Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Confusion Matrix</h3>
          <div className="grid grid-cols-2 gap-3 max-w-xs mx-auto text-center">
            {[
              { label: 'True Positive',  val: cm.tp, bg: 'bg-green-500/20 border-green-500/40', col: 'text-green-400' },
              { label: 'False Negative', val: cm.fn, bg: 'bg-red-500/20 border-red-500/40',     col: 'text-red-400'   },
              { label: 'False Positive', val: cm.fp, bg: 'bg-amber-500/20 border-amber-500/40', col: 'text-amber-400' },
              { label: 'True Negative',  val: cm.tn, bg: 'bg-navy-700 border-navy-600',         col: 'text-gray-300'  },
            ].map(({ label, val, bg, col }) => (
              <div key={label} className={`${bg} border rounded-lg p-4`}>
                <p className={`text-xl font-bold ${col}`}>{fmtNumber(val)}</p>
                <p className="text-xs text-gray-400 mt-1">{label}</p>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-500 text-center mt-3">
            Threshold: {(metrics.threshold * 100).toFixed(1)}%
          </p>
        </div>

        {/* Val vs Test comparison */}
        {metrics.val_metrics && (
          <div className="card">
            <h3 className="text-sm font-medium text-gray-300 mb-4">Validation vs Test Metrics</h3>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={[
                { metric: 'Precision', val: (metrics.val_metrics.precision??0)*100, test: metrics.test_precision*100 },
                { metric: 'Recall',    val: (metrics.val_metrics.recall??0)*100,    test: metrics.test_recall*100    },
                { metric: 'F1',        val: (metrics.val_metrics.f1??0)*100,        test: metrics.test_f1*100        },
                { metric: 'PR-AUC',    val: (metrics.val_metrics.pr_auc??0)*100,    test: metrics.test_pr_auc*100    },
              ]}>
                <XAxis dataKey="metric" tick={{ fontSize: 10, fill: '#9ca3af' }} />
                <YAxis domain={[0,100]} tick={{ fontSize: 10, fill: '#6b7280' }} />
                <Tooltip contentStyle={{ background: '#0f2040', border: '1px solid #162b55', fontSize: 12 }}
                  formatter={(v: number) => `${v.toFixed(1)}%`} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="val"  name="Validation" fill="#6366f1" radius={[3,3,0,0]} />
                <Bar dataKey="test" name="Test"        fill="#3b82f6" radius={[3,3,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* PR Curve */}
      {metrics.pr_curve && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Precision-Recall Curve</h3>
          <p className="text-xs text-gray-500 mb-3">
            Higher area = better ranking. The PR-AUC of {fmtPct(metrics.test_pr_auc)} indicates strong discrimination on the simulated AMLSim test set.
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={metrics.pr_curve.recall.map((r, i) => ({
              recall:    Math.round(r * 1000) / 1000,
              precision: Math.round(metrics.pr_curve!.precision[i] * 1000) / 1000,
            }))}>
              <CartesianGrid stroke="#162b55" strokeDasharray="3 3" />
              <XAxis dataKey="recall"    label={{ value: 'Recall',    position: 'insideBottom', dy: 10, fill: '#6b7280', fontSize: 11 }}
                tick={{ fontSize: 10, fill: '#6b7280' }} domain={[0,1]} />
              <YAxis dataKey="precision" label={{ value: 'Precision', angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }}
                tick={{ fontSize: 10, fill: '#6b7280' }} domain={[0,1]} />
              <Tooltip contentStyle={{ background: '#0f2040', border: '1px solid #162b55', fontSize: 11 }}
                formatter={(v: number) => v.toFixed(3)} />
              <Line type="monotone" dataKey="precision" stroke="#3b82f6" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Threshold table */}
      {tt.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-3">Precision/Recall at Thresholds</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-gray-300">
              <thead>
                <tr className="border-b border-navy-700 text-gray-400">
                  <th className="py-2 px-3 text-left">Threshold</th>
                  <th className="py-2 px-3 text-right">Precision</th>
                  <th className="py-2 px-3 text-right">Recall</th>
                  <th className="py-2 px-3 text-right">F1</th>
                  <th className="py-2 px-3 text-right">Alerts</th>
                </tr>
              </thead>
              <tbody>
                {tt.map(row => (
                  <tr key={row.threshold}
                    className={`border-b border-navy-700/50 ${Math.abs(row.threshold - metrics.threshold) < 0.03 ? 'bg-accent-500/10' : ''}`}>
                    <td className="py-1.5 px-3 font-mono">{(row.threshold*100).toFixed(0)}%</td>
                    <td className="py-1.5 px-3 text-right">{(row.precision*100).toFixed(1)}%</td>
                    <td className="py-1.5 px-3 text-right">{(row.recall*100).toFixed(1)}%</td>
                    <td className="py-1.5 px-3 text-right">{(row.f1*100).toFixed(1)}%</td>
                    <td className="py-1.5 px-3 text-right font-mono">{row.alerts.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Feature Importance */}
      {fi.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Feature Importance (top 12)</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={fi.map(([k,v]) => ({ name: k, importance: +(v*100).toFixed(2) }))} layout="vertical">
              <XAxis type="number" tick={{ fontSize: 10, fill: '#6b7280' }} />
              <YAxis type="category" dataKey="name" width={190} tick={{ fontSize: 10, fill: '#9ca3af' }} />
              <Tooltip contentStyle={{ background: '#0f2040', border: '1px solid #162b55', fontSize: 11 }}
                formatter={(v: number) => `${v}%`} />
              <Bar dataKey="importance" fill="#3b82f6" radius={[0,3,3,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Risk score distribution */}
      {sdData.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Risk Score Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={sdData}>
              <XAxis dataKey="bin" tick={{ fontSize: 9, fill: '#6b7280' }} />
              <YAxis tick={{ fontSize: 10, fill: '#6b7280' }} />
              <Tooltip contentStyle={{ background: '#0f2040', border: '1px solid #162b55', fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="total" name="All"   fill="#1e3a6e" radius={[3,3,0,0]} />
              <Bar dataKey="fraud" name="Fraud" fill="#ef4444" radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Model Comparison */}
      {modelComp.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-medium text-gray-300 mb-4">Model Comparison (Validation Set)</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-gray-300">
              <thead>
                <tr className="border-b border-navy-700 text-xs text-gray-400">
                  <th className="py-2 px-3 text-left">Model</th>
                  <th className="py-2 px-3 text-right">Val F1</th>
                  <th className="py-2 px-3 text-right">Val PR-AUC</th>
                  <th className="py-2 px-3 text-right">Val Recall</th>
                  <th className="py-2 px-3 text-right">Train Time</th>
                  <th className="py-2 px-3 text-left">Selected</th>
                </tr>
              </thead>
              <tbody>
                {modelComp.map(m => (
                  <tr key={m.model} className={`border-b border-navy-700/50 ${m.selected ? 'bg-accent-500/5' : ''}`}>
                    <td className="py-2 px-3 font-medium">{m.model}</td>
                    <td className="py-2 px-3 text-right">{(m.val_f1*100).toFixed(2)}%</td>
                    <td className="py-2 px-3 text-right">{(m.val_pr_auc*100).toFixed(2)}%</td>
                    <td className="py-2 px-3 text-right">{((m.val_recall ?? 0)*100).toFixed(2)}%</td>
                    <td className="py-2 px-3 text-right">{m.train_time_s ?? '—'}s</td>
                    <td className="py-2 px-3">{m.selected ? <span className="text-green-400 text-xs">Selected</span> : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <p className="text-xs text-gray-600">
        Metrics reflect performance on AMLSim simulated data. Results do not represent real-world banking effectiveness.
      </p>
    </div>
  )
}
