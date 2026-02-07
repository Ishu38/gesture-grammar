/**
 * PWAUpdatePrompt.jsx
 * Shows a prompt when a new version of the app is available
 */

import { useEffect, useState } from 'react';
import { useRegisterSW } from 'virtual:pwa-register/react';

function PWAUpdatePrompt() {
  const [showPrompt, setShowPrompt] = useState(false);

  const {
    needRefresh: [needRefresh, setNeedRefresh],
    updateServiceWorker,
  } = useRegisterSW({
    onRegistered(registration) {
      console.log('Service Worker registered:', registration);

      // Check for updates every hour
      if (registration) {
        setInterval(() => {
          registration.update();
        }, 60 * 60 * 1000);
      }
    },
    onRegisterError(error) {
      console.error('Service Worker registration error:', error);
    },
  });

  useEffect(() => {
    setShowPrompt(needRefresh);
  }, [needRefresh]);

  const handleUpdate = () => {
    updateServiceWorker(true);
  };

  const handleDismiss = () => {
    setNeedRefresh(false);
    setShowPrompt(false);
  };

  if (!showPrompt) return null;

  return (
    <div className="pwa-update-prompt">
      <div className="pwa-update-content">
        <div className="pwa-update-icon">🔄</div>
        <div className="pwa-update-text">
          <strong>New version available!</strong>
          <p>Reload to get the latest features and improvements.</p>
        </div>
        <div className="pwa-update-actions">
          <button className="pwa-btn pwa-btn-primary" onClick={handleUpdate}>
            Reload Now
          </button>
          <button className="pwa-btn pwa-btn-secondary" onClick={handleDismiss}>
            Later
          </button>
        </div>
      </div>
    </div>
  );
}

export default PWAUpdatePrompt;
