import { Routes, Route } from 'react-router-dom'
import AppLayout from './layouts/AppLayout'
import Dashboard from './pages/Dashboard'
import PriorityQueue from './pages/PriorityQueue'
import InvestigationDetail from './pages/InvestigationDetail'
import Analytics from './pages/Analytics'
import ModelManagement from './pages/ModelManagement'
import SettingsPage from './pages/SettingsPage'
import SubmitTransaction from './pages/SubmitTransaction'

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index              element={<Dashboard />} />
        <Route path="queue"       element={<PriorityQueue />} />
        <Route path="queue/:id"   element={<InvestigationDetail />} />
        <Route path="submit"      element={<SubmitTransaction />} />
        <Route path="analytics"   element={<Analytics />} />
        <Route path="model"       element={<ModelManagement />} />
        <Route path="settings"    element={<SettingsPage />} />
      </Route>
    </Routes>
  )
}
