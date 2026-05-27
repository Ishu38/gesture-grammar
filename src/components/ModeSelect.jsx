/**
 * ModeSelect.jsx — graffiti-vibe mode picker (Guided / Sandbox / Recorder).
 * Matches the WelcomeScreen / ContactUs / AboutUs design system.
 * Honest copy: 10 gestures (per src/data/GestureLexicon.json), not 19.
 */

const C = {
  cream:    '#FFF6E6',
  paper:    '#FFFFFF',
  ink:      '#0F172A',
  inkSoft:  '#475569',
  coral:    '#FF5252',
  violet:   '#A855F7',
  teal:     '#14B8A6',
  sun:      '#FACC15',
};
const DISPLAY = "'Bungee', 'Impact', system-ui, sans-serif";
const BODY    = "'Space Grotesk', 'Inter', -apple-system, system-ui, sans-serif";

function Splat({ color = C.coral, style = {} }) {
  return (
    <svg viewBox="0 0 120 120" style={style} aria-hidden="true">
      <path d="M62 8 C 80 6, 95 22, 92 38 C 105 42, 116 56, 108 72 C 116 88, 100 105, 82 100
               C 78 116, 56 118, 48 104 C 30 110, 14 96, 20 78 C 6 70, 4 50, 22 44
               C 22 24, 44 12, 62 8 Z" fill={color}/>
    </svg>
  );
}

function ModeCard({ tagText, tagColor, shadowColor, accentColor, title, desc, bullets, onClick, ariaLabel, isPrimary }) {
  return (
    <button
      onClick={onClick}
      aria-label={ariaLabel}
      style={{
        flex: 1,
        background: C.paper,
        border: `2px solid ${C.ink}`,
        borderRadius: 18,
        padding: '1.6rem 1.4rem',
        cursor: 'pointer',
        textAlign: 'left',
        boxShadow: `6px 6px 0 0 ${shadowColor}`,
        transition: 'transform 0.15s ease, box-shadow 0.15s ease',
        fontFamily: BODY,
      }}
      onMouseEnter={e => { e.currentTarget.style.transform = 'translate(-3px, -3px)'; e.currentTarget.style.boxShadow = `9px 9px 0 0 ${shadowColor}`; }}
      onMouseLeave={e => { e.currentTarget.style.transform = 'translate(0, 0)'; e.currentTarget.style.boxShadow = `6px 6px 0 0 ${shadowColor}`; }}
    >
      <div style={{
        display: 'inline-block',
        background: tagColor, color: C.ink,
        fontFamily: BODY, fontSize: '0.66rem', fontWeight: 800,
        letterSpacing: '0.12em', textTransform: 'uppercase',
        padding: '5px 10px', border: `2px solid ${C.ink}`, borderRadius: 4,
        marginBottom: '1rem',
        boxShadow: `2px 2px 0 0 ${C.ink}`,
      }}>
        {tagText}
      </div>
      <h3 style={{
        fontFamily: DISPLAY, fontSize: '1.4rem', color: C.ink,
        margin: '0 0 0.6rem', letterSpacing: '-0.005em',
      }}>
        {title}
      </h3>
      <p style={{
        fontFamily: BODY, fontSize: '0.88rem', color: C.inkSoft,
        lineHeight: 1.6, margin: '0 0 1rem',
      }}>
        {desc}
      </p>
      <div>
        {bullets.map(item => (
          <div key={item} style={{
            fontFamily: BODY, fontSize: '0.82rem',
            color: isPrimary ? C.ink : C.inkSoft,
            padding: '3px 0',
            display: 'flex', alignItems: 'baseline', gap: '0.4rem',
          }}>
            <span style={{ color: accentColor, fontWeight: 900 }}>+</span>
            <span>{item}</span>
          </div>
        ))}
      </div>
    </button>
  );
}

export default function ModeSelect({ onSelectGuided, onSelectSandbox, onSelectRecorder, profileType }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      minHeight: '100vh', padding: '70px 1.5rem 3rem',
      background: C.cream, fontFamily: BODY,
      position: 'relative', overflow: 'hidden',
    }}>
      <Splat color={C.violet} style={{ position: 'absolute', top: 60, right: -30, width: 180, height: 180, opacity: 0.16, transform: 'rotate(18deg)', pointerEvents: 'none' }} />
      <Splat color={C.teal}   style={{ position: 'absolute', bottom: 80, left: -40, width: 170, height: 170, opacity: 0.14, transform: 'rotate(-22deg)', pointerEvents: 'none' }} />
      <Splat color={C.sun}    style={{ position: 'absolute', top: 480, right: -20, width: 130, height: 130, opacity: 0.18, transform: 'rotate(35deg)', pointerEvents: 'none' }} />

      <div style={{ maxWidth: 720, width: '100%', position: 'relative', zIndex: 1, textAlign: 'center' }}>
        <div style={{
          display: 'inline-block', background: C.sun, color: C.ink,
          fontFamily: BODY, fontSize: '0.72rem', fontWeight: 700,
          letterSpacing: '0.1em', textTransform: 'uppercase',
          padding: '6px 12px', border: `2px solid ${C.ink}`, borderRadius: 4,
          marginBottom: '1.2rem',
          boxShadow: `3px 3px 0 0 ${C.ink}`,
        }}>
          Step 2 of 3
        </div>

        <h2 style={{
          fontFamily: DISPLAY, fontSize: 'clamp(1.8rem, 4.8vw, 2.8rem)',
          color: C.ink, margin: '0 0 0.6rem',
          letterSpacing: '-0.005em', lineHeight: 1.1,
        }}>
          Pick your <span style={{ color: C.coral }}>path.</span>
        </h2>

        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: '0.4rem',
          background: C.paper, border: `2px solid ${C.ink}`, borderRadius: 999,
          padding: '4px 12px 4px 8px', marginBottom: '2rem',
          boxShadow: `3px 3px 0 0 ${C.violet}`,
        }}>
          <span style={{
            width: 18, height: 18, borderRadius: '50%',
            background: C.violet, border: `2px solid ${C.ink}`,
          }}/>
          <span style={{ fontFamily: BODY, fontSize: '0.82rem', color: C.inkSoft }}>
            Profile:
          </span>
          <span style={{ fontFamily: BODY, fontSize: '0.85rem', fontWeight: 700, color: C.ink }}>
            {profileType || 'default'}
          </span>
        </div>
      </div>

      <div style={{
        display: 'grid', gap: '1.4rem',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        maxWidth: 760, width: '100%', position: 'relative', zIndex: 1,
      }}>
        <ModeCard
          tagText="Recommended" tagColor={C.sun}
          shadowColor={C.coral} accentColor={C.coral}
          title="Guided Practice"
          isPrimary
          desc="Step-by-step lessons that teach English sentence structure through gestures. Starts with calibration, then progresses through SVO word order, pronouns, verb agreement, and complete sentences."
          bullets={[
            'Hand-tremor calibration',
            'Structured lessons (5 levels)',
            'Real-time grammar feedback',
            'ISL transfer correction',
            'Progress tracking',
          ]}
          onClick={onSelectGuided}
          ariaLabel="Guided Practice — Step-by-step lessons teaching English sentence structure through gestures"
        />
        <ModeCard
          tagText="Advanced" tagColor={C.violet}
          shadowColor={C.violet} accentColor={C.violet}
          title="Open Sandbox"
          desc="Full access to all gestures, debug panels, UMCE fusion visualization, Prolog parse trees, and every diagnostic tool. For researchers and developers."
          bullets={[
            'All 10 gestures unlocked',
            'UMCE Bayesian fusion panel',
            'AGGME pipeline debug',
            'Prolog X-bar parse trees',
            'Cognitive-load monitor',
          ]}
          onClick={onSelectSandbox}
          ariaLabel="Open Sandbox — Full access to all gestures, debug panels, and diagnostic tools"
        />
      </div>

      <div style={{
        display: 'flex', gap: '1.2rem', marginTop: '2.4rem',
        alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center',
        position: 'relative', zIndex: 1,
      }}>
        <button
          onClick={onSelectSandbox}
          style={{
            background: 'transparent', border: 'none',
            color: C.ink, fontFamily: BODY, fontSize: '0.85rem',
            fontWeight: 600, cursor: 'pointer',
            textDecoration: 'underline', textDecorationThickness: 2,
            textDecorationColor: C.violet, textUnderlineOffset: 4,
            padding: '6px 10px',
          }}
        >
          Skip to sandbox mode
        </button>
        {onSelectRecorder && (
          <button
            onClick={onSelectRecorder}
            aria-label="Record Gestures — Capture gesture training data"
            style={{
              fontFamily: BODY, fontSize: '0.85rem', fontWeight: 700,
              color: C.ink, background: C.sun,
              border: `2px solid ${C.ink}`, borderRadius: 10,
              padding: '9px 18px', cursor: 'pointer',
              boxShadow: `3px 3px 0 0 ${C.ink}`,
              transition: 'transform 0.12s ease, box-shadow 0.12s ease',
            }}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translate(-1px, -1px)'; e.currentTarget.style.boxShadow = `4px 4px 0 0 ${C.ink}`; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translate(0, 0)'; e.currentTarget.style.boxShadow = `3px 3px 0 0 ${C.ink}`; }}
          >
            Record Gestures
          </button>
        )}
      </div>
    </div>
  );
}
