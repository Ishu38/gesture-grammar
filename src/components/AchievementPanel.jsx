/**
 * AchievementPanel.jsx — Achievement & Streak Display
 * Shows unlocked/locked achievements, streak counter, and progress bars.
 */

import { achievementColor, streakColor } from '../core/AchievementSystem';

export default function AchievementPanel({ achievementReport }) {
  if (!achievementReport) return null;

  const { unlocked, locked, closeToUnlock, streak, unlockedCount, totalAchievements } = achievementReport;

  return (
    <div className="achievement-panel" style={{
      background: '#1a1a2e', borderRadius: 8, padding: '10px 14px',
      fontSize: 12, color: '#e2e8f0', marginTop: 8,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontWeight: 700, color: '#fbbf24' }}>
          Achievements ({unlockedCount}/{totalAchievements})
        </span>
        <span style={{
          color: streakColor(streak.current),
          fontWeight: 700,
          fontSize: 13,
        }}>
          {streak.current > 0 ? `${streak.current}d streak` : 'No streak'}
          {streak.longest > 0 && streak.longest > streak.current
            ? ` (best: ${streak.longest}d)` : ''}
        </span>
      </div>

      {/* Unlocked achievements */}
      {unlocked.length > 0 && (
        <div style={{ marginBottom: 6 }}>
          {unlocked.map(a => (
            <div key={a.id} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '3px 0', color: '#fbbf24',
            }}>
              <span style={{ fontSize: 14 }}>{a.icon}</span>
              <span style={{ fontWeight: 600 }}>{a.title}</span>
              <span style={{ color: '#94a3b8', fontSize: 11, marginLeft: 'auto' }}>
                {a.description}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Close to unlocking */}
      {closeToUnlock.length > 0 && (
        <div>
          <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 4 }}>Almost there:</div>
          {closeToUnlock.slice(0, 3).map(a => (
            <div key={a.id} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '2px 0', color: '#64748b',
            }}>
              <span style={{ fontSize: 14, opacity: 0.5 }}>{a.icon}</span>
              <span>{a.title}</span>
              <div style={{
                marginLeft: 'auto', width: 60, height: 6,
                background: '#2a2a3e', borderRadius: 3, overflow: 'hidden',
              }}>
                <div style={{
                  width: `${a.progress.percent}%`, height: '100%',
                  background: '#fbbf24', borderRadius: 3,
                }} />
              </div>
              <span style={{ fontSize: 10, color: '#94a3b8', minWidth: 30, textAlign: 'right' }}>
                {a.progress.current}/{a.progress.threshold}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {unlocked.length === 0 && closeToUnlock.length === 0 && (
        <div style={{ color: '#64748b', textAlign: 'center', padding: 8 }}>
          Keep practicing to earn achievements!
        </div>
      )}
    </div>
  );
}
