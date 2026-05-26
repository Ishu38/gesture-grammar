/**
 * WelcomeScreen.jsx — Professional landing screen for MLAF
 * First impression for educators, funders, and institutional partners.
 * Clean, modern, accessible — sells the vision in 10 seconds.
 */

const BRAND = {
  bg:        'linear-gradient(160deg, #06080d 0%, #0f172a 40%, #1a1a2e 70%, #16213e 100%)',
  primary:   '#4ade80',
  secondary: '#22d3ee',
  accent:    '#8b5cf6',
  text:      '#e2e8f0',
  muted:     '#94a3b8',
  dim:       '#475569',
  border:    'rgba(255,255,255,0.06)',
};

const STATS = [
  { value: '10', label: 'Patent claims' },
  { value: '99.5%', label: 'CNN accuracy' },
  { value: '19', label: 'Gesture vocabulary' },
  { value: '5', label: 'Curriculum stages' },
];

const FEATURES = [
  {
    icon: 'M12 6v6l4 2', // Play button stylized
    title: 'Gesture-Based Grammar',
    desc: 'Build English sentences with hand movements — no keyboard, no voice required.',
    color: '#4ade80',
    symbol: '✋',
  },
  {
    icon: 'M9 12l2 2 4-4',
    title: 'ISL Transfer Detection',
    desc: 'Identifies Indian Sign Language word-order patterns and guides toward English SVO.',
    color: '#8b5cf6',
    symbol: '🔤',
  },
  {
    icon: 'M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z',
    title: 'Motor-Adaptive AI',
    desc: 'Calibrates to hand tremor in real time. Cognitive load monitoring prevents fatigue.',
    color: '#3b82f6',
    symbol: '🎯',
  },
  {
    icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
    title: 'Diagnostic Reports',
    desc: 'Bayesian Knowledge Tracing estimates mastery per concept. Data-driven instruction.',
    color: '#f59e0b',
    symbol: '📊',
  },
];

const STEP_CARDS = [
  { step: 1, title: 'Select Profile', desc: 'Motor / Deaf / Blind / Dyslexia — the system adapts.' },
  { step: 2, title: 'Make a Gesture', desc: 'Perform the hand sign in front of the camera.' },
  { step: 3, title: 'Build a Sentence', desc: 'Words lock in place. Grammar validated in real time.' },
  { step: 4, title: 'Speak & Learn', desc: 'TTS reads the sentence aloud. Reports show progress.' },
];

export default function WelcomeScreen({ onStart, onHandout }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      minHeight: '100vh',
      background: BRAND.bg,
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
      color: BRAND.text,
      overflowX: 'hidden',
    }}>
      {/* ── HERO ── */}
      <div style={{
        textAlign: 'center',
        padding: '3.5rem 2rem 2.5rem',
        maxWidth: '680px',
      }}>
        {/* Badge */}
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '0.4rem',
          padding: '4px 14px', borderRadius: 20,
          background: 'rgba(74,222,128,0.08)', border: '1px solid rgba(74,222,128,0.2)',
          marginBottom: '1.5rem',
        }}>
          <span style={{
            width: 7, height: 7, borderRadius: '50%', background: '#4ade80',
            boxShadow: '0 0 6px #4ade80',
          }} />
          <span style={{ fontSize: '0.7rem', color: '#4ade80', fontWeight: 600, letterSpacing: '0.05em' }}>
            PATENT PENDING · LIVE DEMO
          </span>
        </div>

        {/* Title */}
        <h1 style={{
          fontSize: 'clamp(2.5rem, 6vw, 4rem)',
          fontWeight: 900,
          color: '#ffffff',
          letterSpacing: '0.12em',
          margin: 0,
          lineHeight: 1,
        }}>
          M L A F
        </h1>
        <p style={{
          fontSize: 'clamp(0.85rem, 1.5vw, 1.05rem)',
          color: BRAND.muted,
          marginTop: '0.6rem',
          letterSpacing: '0.08em',
          fontWeight: 400,
        }}>
          Multimodal Language Acquisition Framework
        </p>

        {/* Hero description */}
        <p style={{
          fontSize: '1.05rem',
          color: '#cbd5e1',
          lineHeight: 1.7,
          marginTop: '1.75rem',
          fontWeight: 400,
        }}>
          Teaching English grammar to <strong style={{ color: '#4ade80' }}>motor-impaired learners</strong> and 
          <strong style={{ color: '#8b5cf6' }}> deaf/hard-of-hearing children</strong> through 
          hand gestures, adaptive AI, and real-time grammar analysis.
          No keyboard. No voice. Just your hands.
        </p>

        {/* CTA buttons */}
        <div style={{
          display: 'flex', gap: '0.75rem', justifyContent: 'center',
          marginTop: '2rem', flexWrap: 'wrap',
        }}>
          <button onClick={onStart} style={{
            padding: '0.9rem 2.75rem',
            fontSize: '1rem', fontWeight: 700,
            color: '#0f172a',
            background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
            border: 'none', borderRadius: '12px',
            cursor: 'pointer', letterSpacing: '0.04em',
            boxShadow: '0 4px 28px rgba(74,222,128,0.3)',
            transition: 'transform 0.15s, box-shadow 0.15s',
          }}
            onMouseEnter={e => { e.target.style.transform = 'translateY(-2px)'; e.target.style.boxShadow = '0 8px 36px rgba(74,222,128,0.45)'; }}
            onMouseLeave={e => { e.target.style.transform = 'translateY(0)'; e.target.style.boxShadow = '0 4px 28px rgba(74,222,128,0.3)'; }}
          >
            Begin Session
          </button>

          {onHandout && (
            <button onClick={onHandout} style={{
              padding: '0.9rem 2rem',
              fontSize: '0.95rem', fontWeight: 600,
              color: BRAND.muted,
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: '12px', cursor: 'pointer',
              letterSpacing: '0.03em',
              transition: 'all 0.15s',
            }}
              onMouseEnter={e => { e.target.style.background = 'rgba(255,255,255,0.08)'; e.target.style.borderColor = 'rgba(255,255,255,0.2)'; }}
              onMouseLeave={e => { e.target.style.background = 'rgba(255,255,255,0.04)'; e.target.style.borderColor = 'rgba(255,255,255,0.12)'; }}
            >
              Demo Handout
            </button>
          )}
        </div>
      </div>

      {/* ── STATS STRIP ── */}
      <div style={{
        display: 'flex', gap: '2rem', justifyContent: 'center',
        padding: '1.25rem 2rem', flexWrap: 'wrap',
        borderTop: `1px solid ${BRAND.border}`, borderBottom: `1px solid ${BRAND.border}`,
        width: '100%', maxWidth: '680px',
        background: 'rgba(255,255,255,0.01)',
      }}>
        {STATS.map(s => (
          <div key={s.label} style={{ textAlign: 'center' }}>
            <div style={{
              fontSize: '1.5rem', fontWeight: 800,
              background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
            }}>
              {s.value}
            </div>
            <div style={{ fontSize: '0.65rem', color: BRAND.dim, marginTop: '0.15rem', letterSpacing: '0.04em' }}>
              {s.label}
            </div>
          </div>
        ))}
      </div>

      {/* ── FEATURE CARDS ── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
        gap: '1rem',
        maxWidth: '820px',
        width: '100%',
        padding: '2rem',
        boxSizing: 'border-box',
      }}>
        {FEATURES.map(f => (
          <div key={f.title} style={{
            background: 'rgba(255,255,255,0.02)',
            border: `1px solid ${BRAND.border}`,
            borderRadius: '14px',
            padding: '1.5rem',
            transition: 'border-color 0.2s, background 0.2s',
          }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = `${f.color}40`; e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = BRAND.border; e.currentTarget.style.background = 'rgba(255,255,255,0.02)'; }}
          >
            <div style={{
              width: 44, height: 44, borderRadius: 12,
              background: `${f.color}15`, color: f.color,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '1.3rem', marginBottom: '1rem',
            }}>
              {f.symbol}
            </div>
            <div style={{
              fontWeight: 700, fontSize: '0.95rem',
              color: '#e2e8f0', marginBottom: '0.4rem',
            }}>
              {f.title}
            </div>
            <div style={{
              fontSize: '0.8rem', color: BRAND.muted,
              lineHeight: 1.55,
            }}>
              {f.desc}
            </div>
          </div>
        ))}
      </div>

      {/* ── HOW IT WORKS ── */}
      <div style={{
        maxWidth: '820px', width: '100%',
        padding: '1.5rem 2rem 2rem', boxSizing: 'border-box',
      }}>
        <h3 style={{
          textAlign: 'center', fontSize: '0.9rem', fontWeight: 700,
          color: BRAND.muted, letterSpacing: '0.08em', textTransform: 'uppercase',
          marginBottom: '1.5rem',
        }}>
          How It Works
        </h3>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.75rem',
        }}>
          {STEP_CARDS.map(s => (
            <div key={s.step} style={{
              textAlign: 'center',
              padding: '1rem 0.5rem',
              borderRadius: 10,
              background: 'rgba(255,255,255,0.015)',
              border: `1px solid ${BRAND.border}`,
            }}>
              <div style={{
                width: 32, height: 32, borderRadius: '50%', margin: '0 auto 0.6rem',
                background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
                color: '#0f172a', display: 'flex', alignItems: 'center',
                justifyContent: 'center', fontSize: '0.85rem', fontWeight: 800,
              }}>
                {s.step}
              </div>
              <div style={{ fontWeight: 700, fontSize: '0.78rem', color: '#e2e8f0', marginBottom: '0.3rem' }}>
                {s.title}
              </div>
              <div style={{ fontSize: '0.68rem', color: BRAND.muted, lineHeight: 1.4 }}>
                {s.desc}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── SECONDARY CTA ── */}
      <div style={{
        textAlign: 'center', padding: '0 2rem 2.5rem',
      }}>
        <p style={{
          fontSize: '0.8rem', color: BRAND.dim, marginBottom: '1rem',
          fontStyle: 'italic',
        }}>
          Under the hood: 1D CNN (99.5% accuracy) via ONNX Runtime, Bayesian trimodal fusion,<br />
          Earley parser with Chomskyan X-bar grammar, and adaptive cognitive tutoring — all in-browser.
        </p>
      </div>

      {/* ── FOOTER ── */}
      <div style={{
        padding: '1.25rem 2rem', borderTop: `1px solid ${BRAND.border}`,
        width: '100%', textAlign: 'center',
        fontSize: '0.7rem', color: BRAND.dim,
        letterSpacing: '0.03em', boxSizing: 'border-box',
      }}>
        Designed & Created by Neil Shankar Ray · NLP & Speech AI Engineer · Applied Linguist (MA, 14 yrs) · IIT Patna AI/ML
      </div>
    </div>
  );
}
