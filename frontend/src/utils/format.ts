export const fmtAmount = (n: number) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(n)

export const fmtNumber = (n: number) =>
  new Intl.NumberFormat('en-US').format(n)

export const fmtPct = (n: number, decimals = 2) =>
  `${(n * 100).toFixed(decimals)}%`

export const fmtScore = (n: number) => (n * 100).toFixed(1)

export const priorityClass = (p: string) => {
  switch (p?.toUpperCase()) {
    case 'HIGH':   return 'badge-high'
    case 'MEDIUM': return 'badge-medium'
    default:       return 'badge-low'
  }
}

export const statusClass = (s: string) => {
  switch (s?.toUpperCase()) {
    case 'OPEN':         return 'badge-open'
    case 'UNDER_REVIEW': return 'badge-open'
    case 'ESCALATED':    return 'badge-escalated'
    case 'RESOLVED':     return 'badge-resolved'
    default:             return 'badge-open'
  }
}

export const fmtTs = (ts: number) => `Day ${ts}`
