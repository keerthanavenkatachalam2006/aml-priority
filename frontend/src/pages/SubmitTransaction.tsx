import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Send, AlertTriangle, CheckCircle, ArrowRight } from 'lucide-react'
import api from '../services/api'

interface SubmitResult {
  alert_id:   number
  tx_id:      number
  risk_score: number
  priority:   string
  message:    string
}

const PRIORITY_STYLE: Record<string, string> = {
  HIGH:   'text-red-400 border-red-500/40 bg-red-500/10',
  MEDIUM: 'text-amber-400 border-amber-500/40 bg-amber-500/10',
  LOW:    'text-green-400 border-green-500/40 bg-green-500/10',
}

export default function SubmitTransaction() {
  const nav = useNavigate()
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState<SubmitResult | null>(null)
  const [error, setError]     = useState<string | null>(null)

  // Form state
  const [form, setForm] = useState({
    tx_id:               Math.floor(Math.random() * 90000000) + 10000000,
    sender_account_id:   Math.floor(Math.random() * 100000),
    receiver_account_id: Math.floor(Math.random() * 100000),
    tx_amount:           15.0,
    timestamp:           200,
    sender_tx_count:           10,
    sender_out_volume:         150.0,
    sender_avg_amount:         15.0,
    sender_n_counterparties:   5,
    sender_fraud_account:      0,
    sender_tx_behavior:        1,
    sender_init_balance:       300.0,
    receiver_tx_count:         8,
    receiver_in_volume:        120.0,
    receiver_avg_amount:       15.0,
    receiver_n_counterparties: 4,
    receiver_fraud_account:    0,
    receiver_tx_behavior:      1,
    amount_to_sender_avg_ratio: 1.0,
  })

  function set(key: string, val: number) {
    setForm(f => ({ ...f, [key]: val }))
    // auto-compute ratio
    if (key === 'tx_amount' || key === 'sender_avg_amount') {
      const amt = key === 'tx_amount' ? val : form.tx_amount
      const avg = key === 'sender_avg_amount' ? val : form.sender_avg_amount
      setForm(f => ({ ...f, [key]: val, amount_to_sender_avg_ratio: avg > 0 ? +(amt / avg).toFixed(3) : 0 }))
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await api.post<SubmitResult>('/alerts/submit', form)
      setResult(res.data)
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { detail?: string } } })
        ?.response?.data?.detail ?? 'Submission failed'
      setError(typeof msg === 'string' ? msg : JSON.stringify(msg))
    }
    setLoading(false)
  }

  function reset() {
    setResult(null)
    setError(null)
    setForm(f => ({
      ...f,
      tx_id: Math.floor(Math.random() * 90000000) + 10000000,
    }))
  }

  const Field = ({ label, fkey, step = 1, min = 0, max, hint }: {
    label: string; fkey: keyof typeof form; step?: number; min?: number; max?: number; hint?: string
  }) => (
    <div>
      <label className="block text-xs text-gray-400 mb-1">{label}</label>
      <input
        type="number"
        step={step}
        min={min}
        max={max}
        value={form[fkey]}
        onChange={e => set(fkey, parseFloat(e.target.value) || 0)}
        className="w-full bg-navy-700 border border-navy-600 rounded-lg px-3 py-2 text-sm text-gray-100
                   focus:outline-none focus:border-accent-500 transition-colors"
      />
      {hint && <p className="text-xs text-gray-600 mt-0.5">{hint}</p>}
    </div>
  )

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-xl font-semibold text-white">Submit New Transaction</h1>
        <p className="text-sm text-gray-400 mt-1">
          Enter transaction details → the AI model scores it in real-time → it appears in the Priority Queue.
        </p>
      </div>

      {/* Result banner */}
      {result && (
        <div className={`border rounded-xl p-5 ${PRIORITY_STYLE[result.priority]}`}>
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle size={18} />
            <span className="font-semibold">Transaction scored and added to queue!</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-gray-400 text-xs">Alert ID</p>
              <p className="font-bold text-lg">#{result.alert_id}</p>
            </div>
            <div>
              <p className="text-gray-400 text-xs">Risk Score</p>
              <p className="font-bold text-lg">{(result.risk_score * 100).toFixed(2)}%</p>
            </div>
            <div>
              <p className="text-gray-400 text-xs">Priority</p>
              <p className="font-bold text-lg">{result.priority}</p>
            </div>
            <div>
              <p className="text-gray-400 text-xs">TX ID</p>
              <p className="font-mono text-sm">{result.tx_id}</p>
            </div>
          </div>
          <div className="flex gap-3 mt-4">
            <button
              onClick={() => nav(`/queue/${result.alert_id}`)}
              className="btn-primary flex items-center gap-1.5 text-xs"
            >
              View in Queue <ArrowRight size={12} />
            </button>
            <button onClick={reset} className="btn-secondary text-xs">Submit Another</button>
          </div>
        </div>
      )}

      {error && (
        <div className="border border-red-500/30 bg-red-500/10 rounded-xl p-4 flex items-start gap-2">
          <AlertTriangle size={16} className="text-red-400 mt-0.5 shrink-0" />
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {!result && (
        <form onSubmit={handleSubmit} className="space-y-5">

          {/* Transaction identity */}
          <div className="card space-y-4">
            <h2 className="text-sm font-medium text-gray-300 border-b border-navy-700 pb-2">
              Transaction Identity
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Field label="Transaction ID"      fkey="tx_id"               hint="Unique — auto-generated" />
              <Field label="Sender Account ID"   fkey="sender_account_id"   />
              <Field label="Receiver Account ID" fkey="receiver_account_id" />
              <Field label="Amount ($)"          fkey="tx_amount"  step={0.01} min={0.01} hint="AMLSim: $2.50–$20" />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <Field label="Simulation Step (Timestamp)" fkey="timestamp" min={0} max={999} hint="Day/step number" />
              <Field label="Amount ÷ Sender Avg (auto)"  fkey="amount_to_sender_avg_ratio" step={0.001} hint="Auto-computed" />
            </div>
          </div>

          {/* Sender history */}
          <div className="card space-y-4">
            <h2 className="text-sm font-medium text-gray-300 border-b border-navy-700 pb-2">
              Sender Account History
              <span className="text-xs text-gray-500 font-normal ml-2">(rolling past-only stats)</span>
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Field label="Past Tx Count"       fkey="sender_tx_count"         hint="How many txs sender made before" />
              <Field label="Total Out Volume ($)" fkey="sender_out_volume" step={0.01} />
              <Field label="Avg Amount ($)"      fkey="sender_avg_amount"  step={0.01} hint="Used for ratio calculation" />
              <Field label="# Counterparties"    fkey="sender_n_counterparties" hint="Unique accounts sender used" />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Account Flagged as Fraud?</label>
                <select
                  value={form.sender_fraud_account}
                  onChange={e => set('sender_fraud_account', parseInt(e.target.value))}
                  className="w-full bg-navy-700 border border-navy-600 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-accent-500"
                >
                  <option value={0}>No (0)</option>
                  <option value={1}>Yes (1)</option>
                </select>
              </div>
              <Field label="Behavior Model (1–5)" fkey="sender_tx_behavior" min={1} max={5} hint="Simulation behavior type" />
              <Field label="Initial Balance ($)"  fkey="sender_init_balance" step={0.01} />
            </div>
          </div>

          {/* Receiver history */}
          <div className="card space-y-4">
            <h2 className="text-sm font-medium text-gray-300 border-b border-navy-700 pb-2">
              Receiver Account History
              <span className="text-xs text-gray-500 font-normal ml-2">(rolling past-only stats)</span>
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Field label="Past Tx Count"       fkey="receiver_tx_count"         />
              <Field label="Total In Volume ($)" fkey="receiver_in_volume"  step={0.01} />
              <Field label="Avg Amount ($)"      fkey="receiver_avg_amount" step={0.01} />
              <Field label="# Counterparties"    fkey="receiver_n_counterparties" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs text-gray-400 mb-1">Account Flagged as Fraud?</label>
                <select
                  value={form.receiver_fraud_account}
                  onChange={e => set('receiver_fraud_account', parseInt(e.target.value))}
                  className="w-full bg-navy-700 border border-navy-600 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:border-accent-500"
                >
                  <option value={0}>No (0)</option>
                  <option value={1}>Yes (1)</option>
                </select>
              </div>
              <Field label="Behavior Model (1–5)" fkey="receiver_tx_behavior" min={1} max={5} />
            </div>
          </div>

          {/* Quick presets */}
          <div className="card">
            <h2 className="text-sm font-medium text-gray-300 mb-3">Quick Presets</h2>
            <div className="flex flex-wrap gap-2">
              <button type="button" className="btn-secondary text-xs"
                onClick={() => setForm(f => ({
                  ...f,
                  tx_id: Math.floor(Math.random() * 90000000) + 10000000,
                  tx_amount: 15, sender_fraud_account: 1, receiver_fraud_account: 1,
                  sender_tx_count: 200, receiver_tx_count: 150,
                  sender_out_volume: 3000, receiver_in_volume: 2250,
                  sender_avg_amount: 1.0, receiver_avg_amount: 1.0,
                  sender_n_counterparties: 80, receiver_n_counterparties: 60,
                  amount_to_sender_avg_ratio: 15.0,
                  sender_tx_behavior: 2, receiver_tx_behavior: 2,
                  sender_init_balance: 150,
                }))}>
                🔴 High Risk (Fraud Flags ON)
              </button>
              <button type="button" className="btn-secondary text-xs"
                onClick={() => setForm(f => ({
                  ...f,
                  tx_id: Math.floor(Math.random() * 90000000) + 10000000,
                  tx_amount: 500, sender_fraud_account: 0, receiver_fraud_account: 0,
                  sender_tx_count: 5, receiver_tx_count: 3,
                  sender_out_volume: 2500, receiver_in_volume: 1500,
                  sender_avg_amount: 500, receiver_avg_amount: 500,
                  sender_n_counterparties: 2, receiver_n_counterparties: 1,
                  amount_to_sender_avg_ratio: 1.0,
                  sender_tx_behavior: 1, receiver_tx_behavior: 1,
                  sender_init_balance: 5000,
                }))}>
                🟢 Low Risk (Normal Transaction)
              </button>
              <button type="button" className="btn-secondary text-xs"
                onClick={() => setForm(f => ({
                  ...f,
                  tx_id: Math.floor(Math.random() * 90000000) + 10000000,
                  tx_amount: 12, sender_fraud_account: 0, receiver_fraud_account: 1,
                  sender_tx_count: 200, receiver_tx_count: 150,
                  sender_out_volume: 2400, receiver_in_volume: 2250,
                  sender_avg_amount: 8, receiver_avg_amount: 8,
                  sender_n_counterparties: 100, receiver_n_counterparties: 75,
                  amount_to_sender_avg_ratio: 1.5,
                  sender_tx_behavior: 2, receiver_tx_behavior: 2,
                  sender_init_balance: 150,
                }))}>
                🟡 Medium Risk (Receiver Flagged)
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="btn-primary w-full flex items-center justify-center gap-2 py-3 text-base"
          >
            {loading
              ? <><span className="animate-spin">⟳</span> Scoring with AI model...</>
              : <><Send size={16} /> Score Transaction &amp; Add to Priority Queue</>
            }
          </button>

          <p className="text-xs text-gray-600 text-center">
            The XGBoost model scores this transaction in real-time using all 16 features.
            It will appear immediately at the top of the Priority Queue if risk is high.
          </p>
        </form>
      )}
    </div>
  )
}
