import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import tailwindcss from '@tailwindcss/postcss'
import autoprefixer from 'autoprefixer'
import { readFileSync } from 'node:fs'

const pkg = JSON.parse(readFileSync('./package.json', 'utf-8'));

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const grammarTarget = env.VITE_GRAMMAR_ENGINE_URL || 'http://localhost:8300';

  return {
    define: {
      __APP_VERSION__: JSON.stringify(pkg.version),
    },
    plugins: [
      react(),
      VitePWA({
        // 2026-05-27: switched from 'prompt' to 'autoUpdate' so bug fixes
        // and content updates reach users without requiring them to tap a
        // dialog. The previous 'prompt' mode left phones stuck on stale
        // bundles for weeks, including a tremor-tolerance fix that users
        // were reporting as still broken even after deploy.
        registerType: 'autoUpdate',
        injectRegister: false, // We register manually via workbox-window in main.jsx
        workbox: {
          globPatterns: [
            '**/*.{js,css,html,svg,wasm,task,json,onnx}',
          ],
          maximumFileSizeToCacheInBytes: 30 * 1024 * 1024, // 30 MB (hand_landmarker.task is ~7.5 MB)
        },
        manifest: {
          name: 'MLAF — Talk With Your Hands',
          short_name: 'MLAF',
          description: 'Turn hand gestures into spoken English sentences. Free, browser-based, camera-only — built for people with motor impairment and the families who refuse to wait.',
          theme_color: '#FF5252',
          background_color: '#FFF6E6',
          display: 'standalone',
          orientation: 'portrait',
          lang: 'en-IN',
          categories: ['education', 'accessibility', 'medical'],
          icons: [
            { src: 'pwa-192x192.svg',     sizes: '192x192', type: 'image/svg+xml', purpose: 'any' },
            { src: 'pwa-512x512.svg',     sizes: '512x512', type: 'image/svg+xml', purpose: 'any' },
            { src: 'pwa-maskable.svg',    sizes: '512x512', type: 'image/svg+xml', purpose: 'maskable' },
          ],
          shortcuts: [
            { name: 'Start session',    short_name: 'Start',  url: '/' },
            { name: 'Read the guide',   short_name: 'Guide',  url: '/guide.pdf' },
          ],
        },
      }),
    ],
    css: {
      postcss: {
        plugins: [
          tailwindcss,
          autoprefixer,
        ],
      },
    },
    server: {
      proxy: {
        '/grammar': {
          target: grammarTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/grammar/, ''),
        },
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            'onnx': ['onnxruntime-web'],
            'graph': ['graphology', 'graphology-shortest-path', 'graphology-traversal'],
            'parser': ['nearley'],
          },
        },
      },
    },
    test: {
      environment: 'node',
      include: ['src/__tests__/**/*.test.js'],
    },
  };
})
