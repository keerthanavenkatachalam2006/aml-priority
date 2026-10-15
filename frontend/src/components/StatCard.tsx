import { motion } from 'framer-motion'
import { LucideIcon } from 'lucide-react'
import { fmtNumber } from '../utils/format'

interface Props {
  label: string
  value: number | string
  icon: LucideIcon
  color?: string
  sub?: string
}

export default function StatCard({ label, value, icon: Icon, color = 'text-accent-400', sub }: Props) {
  return (
    <motion.div
      className="card flex items-start gap-4"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <div className={`p-2.5 rounded-lg bg-navy-700 ${color}`}>
        <Icon size={18} />
      </div>
      <div>
        <p className="text-2xl font-semibold text-white">
          {typeof value === 'number' ? fmtNumber(value) : value}
        </p>
        <p className="text-sm text-gray-400 mt-0.5">{label}</p>
        {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
      </div>
    </motion.div>
  )
}
