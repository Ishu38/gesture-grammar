/**
 * WelcomeBack.jsx — returning-user hook for the landing screen.
 *
 * The retention gap this closes: MLAF already persists per-user progress
 * (streak, mastery, spaced-repetition schedule) to localStorage, but nothing
 * surfaced it on return — a coming-back user saw the same cold start as a
 * first-timer, so there was no reward for returning. This card greets them
 * with their progress and a concrete reason to jump back in (gestures due for
 * review). Renders NOTHING for first-time visitors, so onboarding is untouched.
 *
 * All numbers are read live from the same engines that write them during play:
 *   AchievementSystem ('mlaf_achievements_v1') · GestureMasteryGate
 *   ('mlaf_mastery_v1') · SpacedRepetitionScheduler ('mlaf_srs_v1') ·
 *   SessionDataLogger history. Read-only — these constructors only load.
 */
import { useMemo } from 'react';
import { AchievementSystem } from '../core/AchievementSystem';
import { GestureMasteryGate } from '../core/GestureMasteryGate';
import { SpacedRepetitionScheduler } from '../core/SpacedRepetitionScheduler';
import { SessionDataLogger } from '../core/SessionDataLogger';

const C = {
  cream: '#FFF6E6', paper: '#FFFFFF', ink: '#0F172A', inkSoft: '#475569',
  coral: '#FF5252', violet: '#A855F7', teal: '#14B8A6', sun: '#FACC15',
};
const DISPLAY = "'Bungee', 'Impact', system-ui, sans-serif";
const BODY = "'Space Grotesk', 'Inter', -apple-system, system-ui, sans-serif";

function Chip({ icon, value, label, color }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: '0.5rem',
      background: C.paper, border: `2px solid ${C.ink}`, borderRadius: 8,
      padding: '8px 12px', boxShadow: `2px 2px 0 0 ${C.ink}`,
    }}>
      <span style={{ fontSize: '1.3rem', lineHeight: 1 }} aria-hidden="true">{icon}</span>
      <span style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
        <strong style={{ fontFamily: DISPLAY, fontSize: '1.1rem', color: color || C.ink }}>{value}</strong>
        <span style={{ fontFamily: BODY, fontSize: '0.68rem', color: C.inkSoft, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</span>
      </span>
    </div>
  );
}

export default function WelcomeBack({ onResume }) {
  const stats = useMemo(() => {
    try {
      const streak = new AchievementSystem().getStreak();
      const mastery = new GestureMasteryGate().getMasteryReport();
      const dueCount = new SpacedRepetitionScheduler().getDueCount();
      const history = new SessionDataLogger({ profileType: 'default' }).getSessionHistory() || [];
      const sessions = history.length;
      const totalSentences = history.reduce((s, h) => s + (h.sentences_completed || 0), 0);
      const returning =
        !!streak.lastDate || (mastery.totalMastered || 0) > 0 || sessions > 0 || dueCount > 0;
      return {
        returning,
        streakDays: streak.current || 0,
        totalMastered: mastery.totalMastered || 0,
        dueCount,
        sessions,
        totalSentences,
      };
    } catch {
      return { returning: false };
    }
  }, []);

  if (!stats.returning) return null; // first-time visitor — show nothing

  return (
    <section style={{
      maxWidth: 980, margin: '16px auto 0', padding: '0 22px',
      position: 'relative', zIndex: 2,
    }}>
      <div style={{
        background: C.cream, border: `3px solid ${C.ink}`, borderRadius: 14,
        boxShadow: `5px 5px 0 0 ${C.ink}`, padding: '18px 20px',
        display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '16px',
      }}>
        <div style={{ flex: '1 1 220px', minWidth: 200 }}>
          <h2 style={{
            fontFamily: DISPLAY, fontSize: '1.4rem', color: C.ink,
            margin: '0 0 0.3rem', letterSpacing: '0.01em',
          }}>
            Welcome back <span aria-hidden="true">👋</span>
          </h2>
          <p style={{ fontFamily: BODY, fontSize: '0.9rem', color: C.inkSoft, margin: 0 }}>
            {stats.dueCount > 0
              ? <><strong style={{ color: C.coral }}>{stats.dueCount} gesture{stats.dueCount > 1 ? 's' : ''}</strong> ready for review — a quick win to keep your streak alive.</>
              : <>Pick up right where you left off.</>}
          </p>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', flex: '2 1 360px' }}>
          {/* Streak intentionally NOT shown here — it's earned through practice
              and lives in the in-app achievements, not flashed on arrival. */}
          {stats.totalMastered > 0 && (
            <Chip icon="✅" value={stats.totalMastered} label="mastered" color={C.teal} />
          )}
          {stats.totalSentences > 0 && (
            <Chip icon="💬" value={stats.totalSentences} label="sentences" color={C.violet} />
          )}
          {stats.dueCount > 0 && (
            <Chip icon="📚" value={stats.dueCount} label="to review" color={C.coral} />
          )}
        </div>

        <button
          onClick={onResume}
          style={{
            fontFamily: DISPLAY, fontSize: '0.95rem', color: C.cream,
            background: stats.dueCount > 0 ? C.coral : C.ink,
            border: `3px solid ${C.ink}`, borderRadius: 10,
            padding: '12px 20px', cursor: 'pointer',
            boxShadow: `3px 3px 0 0 ${C.ink}`, whiteSpace: 'nowrap',
            flex: '0 0 auto',
          }}
        >
          {stats.dueCount > 0 ? 'Review now →' : 'Continue →'}
        </button>
      </div>
    </section>
  );
}
