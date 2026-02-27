import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // WebSocket endpoints
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true, changeOrigin: true },
      // REST endpoints
      '/enroll_status': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/reset_session': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/users': { target: 'http://127.0.0.1:8000', changeOrigin: true },

    },
  },
})
