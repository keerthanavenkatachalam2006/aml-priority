export interface Alert {
  id: number
  tx_id: number
  sender_account_id: number
  receiver_account_id: number
  tx_amount: number
  timestamp: number
  risk_score: number
  priority: 'HIGH' | 'MEDIUM' | 'LOW'
  is_fraud_label: boolean
  alert_type: string | null
  created_at: string
  case_id: number | null
  case_status: string | null
}

export interface AlertListResponse {
  total: number
  page: number
  size: number
  items: Alert[]
}

export interface CaseNote {
  id: number
  author: string
  content: string
  created_at: string
}

export interface Case {
  id: number
  alert_id: number
  status: string
  assigned_to: string | null
  created_at: string
  updated_at: string
  resolved_at: string | null
  notes: CaseNote[]
  alert: Alert | null
}

export interface DashboardSummary {
  total_transactions: number
  total_accounts: number
  total_alerts: number
  high_priority_count: number
  medium_priority_count: number
  low_priority_count: number
  open_cases: number
  resolved_cases: number
  fraud_label_count: number
  model_status: string
  model_name: string | null
  model_version: string | null
}

export interface ConfusionMatrix { tn: number; fp: number; fn: number; tp: number }

export interface ModelMetrics {
  model_name: string
  threshold: number
  test_precision: number
  test_recall: number
  test_f1: number
  test_pr_auc: number
  test_roc_auc: number
  confusion_matrix: ConfusionMatrix
  false_positive_rate: number
  pr_curve: { precision: number[]; recall: number[] } | null
  threshold_table: { threshold: number; precision: number; recall: number; f1: number; alerts: number }[] | null
  feature_importance: Record<string, number> | null
  score_distribution: { bins: number[]; counts: number[]; fraud_counts: number[] } | null
  val_metrics: Record<string, number> | null
  sample_mode: boolean
}

export interface ModelStatus {
  loaded: boolean
  model_name?: string
  version?: string
  threshold?: number
  feature_count?: number
  feature_cols?: string[]
  val_metrics?: Record<string, number>
  sample_mode?: boolean
  prediction_unit?: string
  error?: string
}

export interface AppSettings {
  id: number
  high_threshold: number
  medium_threshold: number
  default_page_size: number
  app_theme: string
}

export interface TimeSeriesPoint {
  timestamp: number
  count: number
  fraud_count: number
}
