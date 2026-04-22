/**
 * ModeSelect.jsx — Choose between Guided Practice (primary) and Sandbox (advanced).
 * Guided Practice is visually emphasized as the recommended path.
 */

export default function ModeSelect({ onSelectGuided, onSelectSandbox, onSelectRecorder, profileType }) {
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
      <h2 style={{
        fontSize: '1.5rem',
        fontWeight: 700,
        color: '#e2e8f0',
        marginBottom: '0.5rem',
      }}>
        Choose Your Mode
      </h2>
      <p style={{
        fontSize: '0.85rem',
        color: '#94a3b8',
        marginBottom: '2.5rem',
        textAlign: 'center',
      }}>
        Profile: <span style={{ color: '#60a5fa' }}>{profileType || 'default'}</span>
      </p>

      <div style={{
        display: 'flex',
        gap: '1.5rem',
        maxWidth: '640px',
        width: '100%',
      }}>
        {/* Guided Practice — Primary */}
        <button
          onClick={onSelectGuided}
          style={{
            flex: 1,
            background: 'rgba(74, 222, 128, 0.06)',
            border: '2px solid #4ade80',
            borderRadius: '16px',
            padding: '2rem 1.5rem',
            cursor: 'pointer',
            textAlign: 'left',
            transition: 'transform 0.15s, box-shadow 0.15s',
          }}
          aria-label="Guided Practice — Step-by-step lessons teaching English sentence structure through gestures"
          onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = '0 8px 32px rgba(74, 222, 128, 0.2)'; }}
          onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = 'none'; }}
        >
          <div style={{
            display: 'inline-block',
            background: '#4ade80',
            color: '#0f0f1a',
            padding: '4px 10px',
            borderRadius: '6px',
            fontSize: '0.65rem',
            fontWeight: 800,
            letterSpacing: '0.1em',
            marginBottom: '1rem',
          }}>
            RECOMMENDED
          </div>
          <h3 style={{ color: '#e2e8f0', fontSize: '1.25rem', fontWeight: 700, margin: '0 0 0.75rem' }}>
            Guided Practice
          </h3>
          <p style={{ color: '#94a3b8', fontSize: '0.8rem', lineHeight: 1.6, margin: 0 }}>
            Step-by-step lessons that teach English sentence structure through gestures.
            Starts with calibration, then progresses through SVO word order, pronouns,
            verb agreement, and complete sentences.
          </p>
          <div style={{ marginTop: '1.25rem' }}>
            {['Hand tremor calibration', 'Structured lessons (5 levels)', 'Real-time grammar feedback', 'ISL transfer correction', 'Progress tracking'].map(item => (
              <div key={item} style={{
                fontSize: '0.75rem',
                color: '#4ade80',
                padding: '2px 0',
              }}>
                + {item}
              </div>
            ))}
          </div>
        </button>

        {/* Sandbox — Advanced */}
        <button
          onClick={onSelectSandbox}
          aria-label="Open Sandbox — Full access to all gestures, debug panels, and diagnostic tools"
          style={{
            flex: 1,
            background: 'rgba(255, 255, 255, 0.02)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '16px',
            padding: '2rem 1.5rem',
            cursor: 'pointer',
            textAlign: 'left',
            transition: 'transform 0.15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; }}
          onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; }}
        >
          <div style={{
            display: 'inline-block',
            background: 'rgba(255,255,255,0.08)',
            color: '#94a3b8',
            padding: '4px 10px',
            borderRadius: '6px',
            fontSize: '0.65rem',
            fontWeight: 700,
            letterSpacing: '0.1em',
            marginBottom: '1rem',
          }}>
            ADVANCED
          </div>
          <h3 style={{ color: '#e2e8f0', fontSize: '1.25rem', fontWeight: 700, margin: '0 0 0.75rem' }}>
            Open Sandbox
          </h3>
          <p style={{ color: '#94a3b8', fontSize: '0.8rem', lineHeight: 1.6, margin: 0 }}>
            Full access to all gestures, debug panels, UMCE fusion visualization,
            Prolog parse trees, and every diagnostic tool. For researchers and developers.
          </p>
          <div style={{ marginTop: '1.25rem' }}>
            {['All 19 gestures unlocked', 'UMCE Bayesian fusion panel', 'AGGME pipeline debug', 'Prolog X-bar parse trees', 'Cognitive load monitor'].map(item => (
              <div key={item} style={{
                fontSize: '0.75rem',
                color: '#64748b',
                padding: '2px 0',
              }}>
                + {item}
              </div>
            ))}
          </div>
        </button>
      </div>

      <div style={{ display: 'flex', gap: '1.5rem', marginTop: '2rem', alignItems: 'center' }}>
        <button
          onClick={onSelectSandbox}
          style={{
            background: 'none',
            border: 'none',
            color: '#475569',
            fontSize: '0.75rem',
            cursor: 'pointer',
            textDecoration: 'underline',
          }}
        >
          Skip to sandbox mode
        </button>
        {onSelectRecorder && (
          <button
            onClick={onSelectRecorder}
            aria-label="Record Gestures — Capture gesture training data"
            style={{
              background: 'rgba(251, 191, 36, 0.08)',
              border: '1px solid rgba(251, 191, 36, 0.3)',
              borderRadius: '8px',
              padding: '0.5rem 1.25rem',
              color: '#fbbf24',
              fontSize: '0.75rem',
              fontWeight: 700,
              cursor: 'pointer',
              transition: 'transform 0.15s',
            }}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; }}
          >
            Record Gestures
          </button>
        )}
      </div>
    </div>
  );
}
