import axios from 'axios'
import type {
  Alert, AlertListResponse, Case, DashboardSummary,
  ModelMetrics, ModelStatus, AppSettings, TimeSeriesPoint
} from '../types'

const api = axios.create({ baseURL: '/api', timeout: 30_000 })

// ── Dashboard ────────────────────────────────────────────────────────────────
export const getDashboardSummary  = () => api.get<DashboardSummary>('/dashboard/summary').then(r => r.data)
export const getRecentAlerts      = (limit = 10) => api.get<Alert[]>(`/dashboard/recent-alerts?limit=${limit}`).then(r => r.data)
export const getAlertTimeseries   = () => api.get<TimeSeriesPoint[]>('/dashboard/timeseries').then(r => r.data)

// ── Alerts ───────────────────────────────────────────────────────────────────
export interface AlertFilters {
  page?: number; size?: number; priority?: string; status?: string
  search?: string; min_amount?: number; max_amount?: number
  min_ts?: number; max_ts?: number; sort_by?: string; sort_dir?: string
}
export const getAlerts    = (f: AlertFilters = {}) =>
  api.get<AlertListResponse>('/alerts', { params: f }).then(r => r.data)
export const getAlertById = (id: number) => api.get<Alert>(`/alerts/${id}`).then(r => r.data)
export const getExportUrl = (params: Record<string, string>) => {
  const q = new URLSearchParams(params).toString()
  return `/api/alerts/export?${q}`
}

// ── Cases ────────────────────────────────────────────────────────────────────
export const getCases          = (page = 1, size = 25, status?: string) =>
  api.get('/cases', { params: { page, size, status } }).then(r => r.data)
export const getCaseById       = (id: number) => api.get<Case>(`/cases/${id}`).then(r => r.data)
export const createCaseFromAlert = (alertId: number) =>
  api.post<Case>(`/cases/from-alert/${alertId}`).then(r => r.data)
export const updateCaseStatus  = (id: number, status: string, assigned_to?: string) =>
  api.patch<Case>(`/cases/${id}/status`, { status, assigned_to }).then(r => r.data)
export const addCaseNote       = (id: number, content: string, author = 'Analyst') =>
  api.post(`/cases/${id}/notes`, { content, author }).then(r => r.data)

// ── Analytics ────────────────────────────────────────────────────────────────
export const getAnalyticsSummary = () => api.get('/analytics/summary').then(r => r.data)
export const getModelMetrics     = () => api.get<ModelMetrics>('/analytics/model-metrics').then(r => r.data)

// ── Model ────────────────────────────────────────────────────────────────────
export const getModelStatus  = () => api.get<ModelStatus>('/model/status').then(r => r.data)
export const getDatasetInfo  = () => api.get('/model/dataset-info').then(r => r.data)
export const reloadModel     = () => api.post('/model/reload').then(r => r.data)
export const scoreFeatures   = (features: Record<string, number>) =>
  api.post('/model/predictions/score', { features }).then(r => r.data)

// ── Settings ─────────────────────────────────────────────────────────────────
export const getSettings    = () => api.get<AppSettings>('/settings').then(r => r.data)
export const updateSettings = (data: Partial<AppSettings>) =>
  api.patch<AppSettings>('/settings', data).then(r => r.data)

// ── Health ───────────────────────────────────────────────────────────────────
export const getHealth = () => api.get('/health').then(r => r.data)

export default api
