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
        registerType: 'prompt',
        injectRegister: false, // We register manually via workbox-window in main.jsx
        workbox: {
          globPatterns: [
            '**/*.{js,css,html,svg,wasm,task,json,onnx}',
          ],
          maximumFileSizeToCacheInBytes: 30 * 1024 * 1024, // 30 MB (hand_landmarker.task is ~7.5 MB)
        },
        manifest: {
          name: 'MLAF — Multimodal Language Acquisition Framework',
          short_name: 'MLAF',
          description: 'Grammar acquisition through gesture-based multimodal learning',
          theme_color: '#0f172a',
          background_color: '#0f0f1a',
          display: 'standalone',
          icons: [
            { src: 'pwa-192x192.svg', sizes: '192x192', type: 'image/svg+xml' },
            { src: 'pwa-512x512.svg', sizes: '512x512', type: 'image/svg+xml' },
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
