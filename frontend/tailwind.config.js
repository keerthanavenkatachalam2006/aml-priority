/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#050d1a',
          900: '#0a1628',
          800: '#0f2040',
          700: '#162b55',
          600: '#1e3a6e',
        },
        accent: {
          500: '#3b82f6',
          400: '#60a5fa',
          300: '#93c5fd',
        },
        risk: {
          high:   '#ef4444',
          medium: '#f59e0b',
          low:    '#22c55e',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    }
  },
  plugins: []
}
