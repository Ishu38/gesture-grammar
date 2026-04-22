/**
 * AchievementToast.jsx — Pop-up notification for newly unlocked achievements.
 * Auto-dismisses after 4 seconds.
 */

import { useEffect, useState, useRef } from 'react';

export default function AchievementToast({ achievement, onDismiss, lowStimulus = false }) {
  const [visible, setVisible] = useState(true);
  const onDismissRef = useRef(onDismiss);
  onDismissRef.current = onDismiss;

  useEffect(() => {
    if (!achievement) return;
    setVisible(true);
    const timer = setTimeout(() => {
      if (onDismissRef.current) {
        onDismissRef.current();
      } else {
        setVisible(false); // Self-dismiss if no onDismiss provided
      }
    }, 4000);
    return () => clearTimeout(timer);
  }, [achievement]);

  if (!achievement || !visible) return null;

  return (
    <div style={{
      position: 'fixed', top: 20, right: 20, zIndex: 10000,
      background: 'linear-gradient(135deg, #1a1a2e 0%, #2a1a3e 100%)',
      border: '2px solid #fbbf24',
      borderRadius: 12, padding: '14px 20px',
      boxShadow: '0 8px 32px rgba(251, 191, 36, 0.3)',
      animation: lowStimulus ? 'none' : 'toast-slide-in 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
      cursor: 'pointer', maxWidth: 320,
    }} onClick={onDismiss}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{ fontSize: 28 }}>{achievement.icon}</span>
        <div>
          <div style={{ color: '#fbbf24', fontWeight: 700, fontSize: 14 }}>
            Achievement Unlocked!
          </div>
          <div style={{ color: '#e2e8f0', fontWeight: 600, fontSize: 13 }}>
            {achievement.title}
          </div>
          <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 2 }}>
            {achievement.description}
          </div>
        </div>
      </div>
    </div>
  );
}
