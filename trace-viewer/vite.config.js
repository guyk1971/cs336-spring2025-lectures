import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: 'https://stanford-cs336.github.io/spring2025-lectures/',
  plugins: [react()],
})
