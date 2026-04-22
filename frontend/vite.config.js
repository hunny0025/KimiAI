import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Optional PWA support — only loaded if package is installed
let VitePWA;
try {
  VitePWA = (await import('vite-plugin-pwa')).VitePWA;
} catch (_) {
  VitePWA = null;
}

const pwaPlugin = VitePWA ? VitePWA({
  registerType: 'autoUpdate',
  workbox: {
    runtimeCaching: [
      {
        urlPattern: ({ url }) => url.pathname.startsWith('/api/'),
        handler: 'StaleWhileRevalidate',
        options: {
          cacheName: 'api-cache',
          expiration: { maxEntries: 50, maxAgeSeconds: 86400 }
        }
      }
    ],
    globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}']
  },
  manifest: {
    name: 'Skill Genome',
    short_name: 'Skill Genome',
    description: 'Skill Genome — Multi-Agent Workforce Intelligence',
    theme_color: '#00ff88',
    background_color: '#0d1117',
    display: 'standalone',
    icons: [
      { src: '/icons/icon-192.png', sizes: '192x192', type: 'image/png' },
      { src: '/icons/icon-512.png', sizes: '512x512', type: 'image/png' }
    ]
  }
}) : null;

export default defineConfig({
  plugins: [
    react(),
    ...(pwaPlugin ? [pwaPlugin] : [])
  ],
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['recharts'],
          motion: ['framer-motion'],
        }
      }
    }
  },
  server: {
    // Local dev proxy — /api/* → Flask on :5000
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true
      }
    }
  }
})
