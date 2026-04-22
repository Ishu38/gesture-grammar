/**
 * UpdatePrompt.jsx — PWA Update Toast
 *
 * Listens for the 'swUpdate' custom event (dispatched by main.jsx when a new
 * service worker is waiting) and shows a non-intrusive toast prompting the user
 * to reload. Also displays the current app version.
 */

import { useState, useEffect, useCallback } from 'react';

const APP_VERSION = typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : 'dev';

export default function UpdatePrompt() {
  const [waitingWb, setWaitingWb] = useState(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    function onSwUpdate(e) {
      setWaitingWb(e.detail.wb);
      setDismissed(false);
    }
    window.addEventListener('swUpdate', onSwUpdate);
    return () => window.removeEventListener('swUpdate', onSwUpdate);
  }, []);

  const handleUpdate = useCallback(() => {
    if (!waitingWb) return;
    // Tell the waiting SW to activate, then reload the page
    waitingWb.messageSkipWaiting();
    // Reload once the new SW takes control
    waitingWb.addEventListener('controlling', () => {
      window.location.reload();
    });
  }, [waitingWb]);

  const handleDismiss = useCallback(() => {
    setDismissed(true);
  }, []);

  if (!waitingWb || dismissed) return null;

  return (
    <div
      role="alert"
      aria-live="polite"
      style={{
        position: 'fixed',
        bottom: 24,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '12px 20px',
        background: 'rgba(15, 15, 26, 0.95)',
        border: '1px solid rgba(74, 222, 128, 0.3)',
        borderRadius: 12,
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(74, 222, 128, 0.1)',
        backdropFilter: 'blur(12px)',
        maxWidth: 420,
        width: 'calc(100% - 48px)',
        animation: 'slideUp 0.3s ease-out',
      }}
    >
      {/* Pulse indicator */}
      <div style={{
        width: 10, height: 10, borderRadius: '50%',
        background: '#4ade80',
        boxShadow: '0 0 8px rgba(74, 222, 128, 0.6)',
        flexShrink: 0,
      }} />

      {/* Message */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontSize: 13, fontWeight: 700, color: '#e2e8f0',
          marginBottom: 2,
        }}>
          Update Available
        </div>
        <div style={{
          fontSize: 11, color: '#94a3b8',
        }}>
          A new version of MLAF is ready. Current: v{APP_VERSION}
        </div>
      </div>

      {/* Actions */}
      <button
        onClick={handleUpdate}
        style={{
          padding: '6px 16px',
          fontSize: 12,
          fontWeight: 700,
          color: '#0f0f1a',
          background: '#4ade80',
          border: 'none',
          borderRadius: 6,
          cursor: 'pointer',
          whiteSpace: 'nowrap',
          flexShrink: 0,
        }}
      >
        Update
      </button>
      <button
        onClick={handleDismiss}
        aria-label="Dismiss update notification"
        style={{
          padding: 4,
          background: 'none',
          border: 'none',
          color: '#64748b',
          cursor: 'pointer',
          fontSize: 16,
          lineHeight: 1,
          flexShrink: 0,
        }}
      >
        &times;
      </button>
    </div>
  );
}

/**
 * Returns the current app version string for display elsewhere (e.g., footer).
 */
export function getAppVersion() {
  return APP_VERSION;
}
