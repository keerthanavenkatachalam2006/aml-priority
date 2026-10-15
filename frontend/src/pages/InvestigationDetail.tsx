import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, Plus, CheckCircle, AlertOctagon, Eye } from 'lucide-react'
import { PriorityBadge, StatusBadge } from '../components/PriorityBadge'
import { PageLoader } from '../components/LoadingSpinner'
import { useApi } from '../hooks/useApi'
import {
  getAlertById, getCaseById, createCaseFromAlert,
  updateCaseStatus, addCaseNote
} from '../services/api'
import { fmtAmount, fmtTs } from '../utils/format'

export default function InvestigationDetail() {
  const { id } = useParams<{ id: string }>()
  const nav = useNavigate()
  const alertId = Number(id)

  const { data: alert, loading: al } = useApi(() => getAlertById(alertId), [alertId])
  const { data: caseData, loading: cl, refetch: refetchCase } = useApi(
    () => alert?.case_id ? getCaseById(alert.case_id) : Promise.resolve(null),
    [alert?.case_id]
  )

  const [noteText, setNoteText] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  const notify = (msg: string) => { setMessage(msg); setTimeout(() => setMessage(null), 3000) }

  const ensureCase = async () => {
    if (alert?.case_id) return alert.case_id
    const c = await createCaseFromAlert(alertId)
    await refetchCase()
    return c.id
  }

  const handleStatus = async (status: string) => {
    setSubmitting(true)
    try {
      const cid = await ensureCase()
      await updateCaseStatus(cid, status)
      await refetchCase()
      notify(`Status updated to ${status}`)
    } catch { notify('Failed to update status') }
    setSubmitting(false)
  }

  const handleNote = async () => {
    if (!noteText.trim()) return
    setSubmitting(true)
    try {
      const cid = await ensureCase()
      await addCaseNote(cid, noteText)
      setNoteText('')
      await refetchCase()
      notify('Note added')
    } catch { notify('Failed to add note') }
    setSubmitting(false)
  }

  if (al || cl) return <PageLoader />
  if (!alert) return <div className="p-6 text-gray-400">Alert not found</div>

  return (
    <div className="p-6 space-y-5 max-w-5xl">
      {/* Header */}
      <div className="flex items-center gap-3">
        <button onClick={() => nav('/queue')} className="text-gray-400 hover:text-white transition-colors">
          <ArrowLeft size={18} />
        </button>
        <div>
          <h1 className="text-xl font-semibold text-white">Alert #{alert.id}</h1>
          <p className="text-sm text-gray-400">Transaction {alert.tx_id}</p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <PriorityBadge priority={alert.priority} />
          {caseData && <StatusBadge status={caseData.status} />}
        </div>
      </div>

      {message && (
        <div className="bg-green-500/10 border border-green-500/30 text-green-400 text-sm px-4 py-2 rounded-lg">
          {message}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Transaction info */}
        <div className="card lg:col-span-2 space-y-4">
          <h2 className="text-sm font-medium text-gray-300">Transaction Details</h2>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
            {[
              ['Transaction ID',  alert.tx_id],
              ['Amount',          fmtAmount(alert.tx_amount)],
              ['Simulation Step', fmtTs(alert.timestamp)],
              ['Alert Type',      alert.alert_type ?? '—'],
              ['Sender Account',  alert.sender_account_id],
              ['Receiver Account',alert.receiver_account_id],
              ['Fraud Label',     alert.is_fraud_label ? '✓ Flagged (simulated)' : 'Not flagged'],
            ].map(([k, v]) => (
              <div key={String(k)}>
                <dt className="text-gray-500 text-xs">{k}</dt>
                <dd className="text-gray-200 mt-0.5">{String(v)}</dd>
              </div>
            ))}
          </dl>
        </div>

        {/* Risk score panel */}
        <div className="card space-y-3">
          <h2 className="text-sm font-medium text-gray-300">Risk Assessment</h2>
          <div className="flex items-end gap-2">
            <span className="text-4xl font-bold text-white">
              {(alert.risk_score * 100).toFixed(1)}
            </span>
            <span className="text-gray-400 text-lg mb-1">%</span>
          </div>
          <div className="w-full bg-navy-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all ${
                alert.risk_score >= 0.7 ? 'bg-red-500' :
                alert.risk_score >= 0.4 ? 'bg-amber-400' : 'bg-green-500'
              }`}
              style={{ width: `${alert.risk_score * 100}%` }}
            />
          </div>
          <p className="text-xs text-gray-500">
            Score from XGBoost model trained on AMLSim data.
            This score reflects pattern similarity to simulated laundering and does not
            constitute a legal finding.
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="card">
        <h2 className="text-sm font-medium text-gray-300 mb-3">Investigation Actions</h2>
        <div className="flex flex-wrap gap-2">
          <button
            className="btn-secondary flex items-center gap-1.5"
            disabled={submitting}
            onClick={() => handleStatus('UNDER_REVIEW')}
          ><Eye size={13} /> Mark Under Review</button>
          <button
            className="flex items-center gap-1.5 bg-purple-600/20 hover:bg-purple-600/30 text-purple-400 border border-purple-500/30 px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            disabled={submitting}
            onClick={() => handleStatus('ESCALATED')}
          ><AlertOctagon size={13} /> Escalate</button>
          <button
            className="flex items-center gap-1.5 bg-green-600/20 hover:bg-green-600/30 text-green-400 border border-green-500/30 px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            disabled={submitting}
            onClick={() => handleStatus('RESOLVED')}
          ><CheckCircle size={13} /> Mark Resolved</button>
        </div>
      </div>

      {/* Notes */}
      <div className="card space-y-3">
        <h2 className="text-sm font-medium text-gray-300">Investigation Notes</h2>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {(caseData?.notes ?? []).length === 0 ? (
            <p className="text-gray-500 text-sm">No notes yet.</p>
          ) : caseData?.notes.map(n => (
            <div key={n.id} className="bg-navy-700/50 rounded-lg p-3 text-sm">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>{n.author}</span>
                <span>{new Date(n.created_at).toLocaleString()}</span>
              </div>
              <p className="text-gray-300">{n.content}</p>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            className="flex-1 bg-navy-700 border border-navy-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent-500"
            placeholder="Add investigation note…"
            value={noteText}
            onChange={e => setNoteText(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleNote()}
          />
          <button
            className="btn-primary flex items-center gap-1.5"
            disabled={submitting || !noteText.trim()}
            onClick={handleNote}
          ><Plus size={13} /> Add</button>
        </div>
      </div>
    </div>
  )
}
