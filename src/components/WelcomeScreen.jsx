/**
 * WelcomeScreen.jsx — MVP Landing Screen
 * First thing the user sees. Sets the context: what MLAF is,
 * who it's for, and funnels into Guided Practice as the primary path.
 */

const FEATURES = [
  {
    title: 'Gesture-Based Learning',
    desc: 'Build English sentences using hand gestures detected by your camera.',
    icon: 'G',
    color: '#4ade80',
  },
  {
    title: 'ISL Transfer Correction',
    desc: 'Detects ISL word order habits and guides you toward English SVO structure.',
    icon: 'I',
    color: '#60a5fa',
  },
  {
    title: 'Motor-Adaptive',
    desc: 'Calibrates to your hand tremor. More forgiving when you need it.',
    icon: 'M',
    color: '#c084fc',
  },
  {
    title: 'Neuro-Symbolic AI',
    desc: 'Neural perception meets Chomskyan grammar. Bayesian fusion of vision, audio, and formal syntax.',
    icon: 'N',
    color: '#fbbf24',
  },
];

export default function WelcomeScreen({ onStart }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      padding: '2rem',
      background: 'linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%)',
      fontFamily: 'system-ui, -apple-system, sans-serif',
    }}>
      {/* Logo / Title */}
      <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
        <h1 style={{
          fontSize: '3.5rem',
          fontWeight: 800,
          color: '#ffffff',
          letterSpacing: '0.15em',
          margin: 0,
        }}>
          MLAF
        </h1>
        <p style={{
          fontSize: '1rem',
          color: '#94a3b8',
          marginTop: '0.5rem',
          letterSpacing: '0.05em',
        }}>
          Multimodal Language Acquisition Framework
        </p>
      </div>

      {/* One-liner */}
      <p style={{
        fontSize: '1.15rem',
        color: '#cbd5e1',
        textAlign: 'center',
        maxWidth: '540px',
        lineHeight: 1.6,
        marginBottom: '2.5rem',
      }}>
        Teaching English to motor-impaired learners through hand gestures,
        adaptive AI, and real-time grammar analysis.
      </p>

      {/* Feature cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(2, 1fr)',
        gap: '1rem',
        maxWidth: '560px',
        width: '100%',
        marginBottom: '2.5rem',
      }}>
        {FEATURES.map(f => (
          <div key={f.title} style={{
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '12px',
            padding: '1.25rem',
          }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: `${f.color}20`,
              color: f.color,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontWeight: 800, fontSize: 14,
              marginBottom: '0.75rem',
            }}>
              {f.icon}
            </div>
            <div style={{ fontWeight: 700, fontSize: '0.85rem', color: '#e2e8f0', marginBottom: '0.35rem' }}>
              {f.title}
            </div>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', lineHeight: 1.5 }}>
              {f.desc}
            </div>
          </div>
        ))}
      </div>

      {/* Under the hood */}
      <div style={{
        maxWidth: '560px',
        width: '100%',
        marginBottom: '2rem',
        padding: '1.25rem 1.5rem',
        background: 'rgba(255,255,255,0.02)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: '12px',
      }}>
        <p style={{
          fontSize: '0.8rem',
          color: '#64748b',
          margin: 0,
          lineHeight: 1.7,
          textAlign: 'center',
        }}>
          Under the hood: a 1D CNN gesture classifier (99.5% accuracy) running via ONNX Runtime,
          Bayesian late fusion across vision, acoustics, and eye-gaze modalities,
          an Earley parser enforcing Chomskyan X-bar grammar in real time,
          spaced repetition, cognitive load adaptation, and a formal gesture lifecycle DFA
          &mdash; 3 months of research compressed into one click.
        </p>
      </div>

      {/* CTA */}
      <button
        onClick={onStart}
        style={{
          padding: '1rem 3rem',
          fontSize: '1.1rem',
          fontWeight: 700,
          color: '#0f0f1a',
          background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
          border: 'none',
          borderRadius: '12px',
          cursor: 'pointer',
          letterSpacing: '0.05em',
          transition: 'transform 0.15s, box-shadow 0.15s',
          boxShadow: '0 4px 24px rgba(74, 222, 128, 0.25)',
        }}
        onMouseEnter={e => { e.target.style.transform = 'translateY(-2px)'; e.target.style.boxShadow = '0 8px 32px rgba(74, 222, 128, 0.35)'; }}
        onMouseLeave={e => { e.target.style.transform = 'translateY(0)'; e.target.style.boxShadow = '0 4px 24px rgba(74, 222, 128, 0.25)'; }}
      >
        Begin Session
      </button>

      {/* Footer attribution */}
      <p style={{
        marginTop: '2.5rem',
        fontSize: '0.7rem',
        color: '#475569',
        letterSpacing: '0.03em',
      }}>
        Designed & Created by Neil Shankar Ray
      </p>
    </div>
  );
}
