import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// Fix 7 – PWA support (install: npm install vite-plugin-pwa)
// Comment out VitePWA if the package is not installed yet
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
    name: 'SkillGenome X',
    short_name: 'SkillGenome',
    description: 'National Talent Intelligence System',
    theme_color: '#0F6E56',
    background_color: '#0A0F1E',
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
  server: {
    // Fix 10 – dev proxy so /api/* calls reach Flask on :5000 without CORS issues
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true
      }
    }
  }
})
