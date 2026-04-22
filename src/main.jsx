import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(<App />)

// ── PWA Service Worker Registration ─────────────────────────────────────────
// Uses workbox-window to detect when a new version is cached and waiting.
// The 'waiting' SW is communicated to <UpdatePrompt> via a custom event.
if ('serviceWorker' in navigator) {
  import('workbox-window').then(({ Workbox }) => {
    const wb = new Workbox('/sw.js');

    wb.addEventListener('waiting', () => {
      // Dispatch a custom event so UpdatePrompt can show the toast
      window.dispatchEvent(new CustomEvent('swUpdate', { detail: { wb } }));
    });

    // Also handle the case where a waiting SW exists on first load
    wb.addEventListener('externalwaiting', () => {
      window.dispatchEvent(new CustomEvent('swUpdate', { detail: { wb } }));
    });

    wb.register();
  });
}
