import { priorityClass, statusClass } from '../utils/format'

export function PriorityBadge({ priority }: { priority: string }) {
  return <span className={priorityClass(priority)}>{priority}</span>
}

export function StatusBadge({ status }: { status: string }) {
  const label = status?.replace('_', ' ') ?? '—'
  return <span className={statusClass(status)}>{label}</span>
}
