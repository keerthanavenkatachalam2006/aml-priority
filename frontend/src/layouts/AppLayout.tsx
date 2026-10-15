import { NavLink, Outlet } from 'react-router-dom'
import {
  LayoutDashboard, ListFilter, BarChart3,
  Settings, Database, ShieldAlert, PlusCircle
} from 'lucide-react'

const nav = [
  { to: '/',          label: 'Overview',           icon: LayoutDashboard },
  { to: '/queue',     label: 'Priority Queue',      icon: ListFilter },
  { to: '/submit',    label: 'Submit Transaction',  icon: PlusCircle },
  { to: '/analytics', label: 'Analytics',           icon: BarChart3 },
  { to: '/model',     label: 'Model & Data',        icon: Database },
  { to: '/settings',  label: 'Settings',            icon: Settings },
]

export default function AppLayout() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-60 bg-navy-900 border-r border-navy-700 flex flex-col flex-shrink-0">
        <div className="px-5 py-5 border-b border-navy-700">
          <div className="flex items-center gap-2.5">
            <ShieldAlert className="text-accent-400" size={22} />
            <div>
              <p className="text-sm font-semibold text-white leading-tight">AML Priority</p>
              <p className="text-xs text-gray-500">Investigation System</p>
            </div>
          </div>
        </div>
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-accent-500/15 text-accent-400'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-navy-800'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="p-4 border-t border-navy-700">
          <p className="text-xs text-gray-600">Academic prototype</p>
          <p className="text-xs text-gray-700">Not for legal use</p>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto bg-navy-950">
        <Outlet />
      </main>
    </div>
  )
}
