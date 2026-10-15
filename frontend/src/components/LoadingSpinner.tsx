export default function LoadingSpinner({ size = 24 }: { size?: number }) {
  return (
    <div
      style={{ width: size, height: size }}
      className="border-2 border-navy-600 border-t-accent-400 rounded-full animate-spin"
    />
  )
}

export function PageLoader() {
  return (
    <div className="flex items-center justify-center h-64">
      <LoadingSpinner size={32} />
    </div>
  )
}
