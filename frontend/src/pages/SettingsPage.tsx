import { useState, useEffect } from 'react'
import { Save } from 'lucide-react'
import { PageLoader } from '../components/LoadingSpinner'
import { useApi } from '../hooks/useApi'
import { getSettings, updateSettings } from '../services/api'
import type { AppSettings } from '../types'

export default function SettingsPage() {
  const { data, loading } = useApi(getSettings)
  const [form, setForm] = useState<Partial<AppSettings>>({})
  const [saving, setSaving] = useState(false)
  const [msg, setMsg] = useState<{ text: string; ok: boolean } | null>(null)

  useEffect(() => {
    if (data) setForm({ ...data })
  }, [data])

  const save = async () => {
    setSaving(true)
    try {
      await updateSettings({
        high_threshold:    form.high_threshold,
        medium_threshold:  form.medium_threshold,
        default_page_size: form.default_page_size,
        app_theme:         form.app_theme,
      })
      setMsg({ text: 'Settings saved.', ok: true })
    } catch {
      setMsg({ text: 'Failed to save settings.', ok: false })
    }
    setSaving(false)
    setTimeout(() => setMsg(null), 3000)
  }

  if (loading) return <PageLoader />

  return (
    <div className="p-6 max-w-xl space-y-6">
      <h1 className="text-xl font-semibold text-white">Settings</h1>

      {msg && (
        <div className={`text-sm px-4 py-2 rounded-lg border ${
          msg.ok ? 'bg-green-500/10 border-green-500/30 text-green-400'
                 : 'bg-red-500/10 border-red-500/30 text-red-400'
        }`}>
          {msg.text}
        </div>
      )}

      <div className="card space-y-5">
        <h2 className="text-sm font-medium text-gray-300">Priority Thresholds</h2>
        <p className="text-xs text-gray-500">
          These thresholds map model risk scores to High/Medium/Low priority labels.
          They are configurable operational settings — not validated legal or regulatory standards.
        </p>

        <div className="space-y-4">
          <div>
            <label className="text-xs text-gray-400 block mb-1">
              High Priority Threshold (≥ this score = HIGH)
            </label>
            <div className="flex items-center gap-3">
              <input type="range" min={0} max={1} step={0.01}
                value={form.high_threshold ?? 0.7}
                onChange={e => setForm(f => ({ ...f, high_threshold: parseFloat(e.target.value) }))}
                className="flex-1 accent-red-500"
              />
              <span className="w-12 text-center text-sm text-white font-mono">
                {((form.high_threshold ?? 0.7) * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Medium Priority Threshold (≥ this score = MEDIUM)
            </label>
            <div className="flex items-center gap-3">
              <input type="range" min={0} max={1} step={0.01}
                value={form.medium_threshold ?? 0.4}
                onChange={e => setForm(f => ({ ...f, medium_threshold: parseFloat(e.target.value) }))}
                className="flex-1 accent-amber-500"
              />
              <span className="w-12 text-center text-sm text-white font-mono">
                {((form.medium_threshold ?? 0.4) * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {(form.high_threshold ?? 0.7) <= (form.medium_threshold ?? 0.4) && (
            <p className="text-xs text-red-400">
              High threshold must be greater than medium threshold.
            </p>
          )}
        </div>
      </div>

      <div className="card space-y-4">
        <h2 className="text-sm font-medium text-gray-300">Display Preferences</h2>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Default Page Size</label>
          <select
            className="bg-navy-700 border border-navy-600 rounded-lg px-3 py-2 text-sm text-white"
            value={form.default_page_size ?? 25}
            onChange={e => setForm(f => ({ ...f, default_page_size: parseInt(e.target.value) }))}
          >
            {[10, 25, 50, 100].map(n => <option key={n} value={n}>{n} rows</option>)}
          </select>
        </div>
        <div>
          <label className="text-xs text-gray-400 block mb-1">Theme</label>
          <select
            className="bg-navy-700 border border-navy-600 rounded-lg px-3 py-2 text-sm text-white"
            value={form.app_theme ?? 'dark'}
            onChange={e => setForm(f => ({ ...f, app_theme: e.target.value }))}
          >
            <option value="dark">Dark</option>
            <option value="light">Light</option>
          </select>
        </div>
      </div>

      <button
        className="btn-primary flex items-center gap-2"
        onClick={save}
        disabled={saving || (form.high_threshold ?? 0.7) <= (form.medium_threshold ?? 0.4)}
      >
        <Save size={14} /> {saving ? 'Saving…' : 'Save Settings'}
      </button>
    </div>
  )
}
