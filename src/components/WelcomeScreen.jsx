/**
 * WelcomeScreen.jsx — Vibrant, graffiti-inflected landing for MLAF.
 *
 * Design intent: warm cream background, spray-paint accents, bold display
 * type. Should feel like a community zine, not a hospital intake form.
 * Honest copy only — every number/claim matches what's actually shipped
 * (see src/data/GestureLexicon.json for the 10 gestures, AccessibilityProfile.js
 * for the 8 profiles).
 */
import WelcomeBack from './WelcomeBack';

const C = {
  cream:    '#FFF6E6',
  paper:    '#FFFFFF',
  dust:     '#F5EDD8',
  ink:      '#0F172A',
  inkSoft:  '#475569',
  coral:    '#FF5252',
  violet:   '#A855F7',
  teal:     '#14B8A6',
  sun:      '#FACC15',
  line:     '#1F2937',
};

const DISPLAY = "'Bungee', 'Impact', system-ui, sans-serif";
const BODY    = "'Space Grotesk', 'Inter', -apple-system, system-ui, sans-serif";

/* ─────────────────────────  decorative SVGs  ───────────────────────── */

function Splat({ color = C.coral, style = {} }) {
  return (
    <svg viewBox="0 0 120 120" style={style} aria-hidden="true">
      <path
        d="M62 8 C 80 6, 95 22, 92 38 C 105 42, 116 56, 108 72 C 116 88, 100 105, 82 100
           C 78 116, 56 118, 48 104 C 30 110, 14 96, 20 78 C 6 70, 4 50, 22 44
           C 22 24, 44 12, 62 8 Z"
        fill={color}
      />
    </svg>
  );
}

function HandMark({ color = C.ink, size = 56 }) {
  // Stylised open-hand silhouette, single colour, recognizable at any size.
  return (
    <svg viewBox="0 0 64 64" width={size} height={size} aria-hidden="true">
      <g fill={color}>
        <rect x="14" y="34" width="36" height="22" rx="10"/>
        <rect x="14" y="14" width="7"  height="26" rx="3.5"/>
        <rect x="24" y="6"  width="7"  height="34" rx="3.5"/>
        <rect x="34" y="10" width="7"  height="30" rx="3.5"/>
        <rect x="44" y="18" width="7"  height="22" rx="3.5"/>
      </g>
    </svg>
  );
}

function ArrowDoodle({ color = C.ink, style = {} }) {
  return (
    <svg viewBox="0 0 80 24" style={style} aria-hidden="true">
      <path d="M4 12 Q 22 0, 40 12 T 70 12" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round"/>
      <path d="M62 6 L 72 12 L 62 18" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function Drips({ color = C.coral, style = {} }) {
  return (
    <svg viewBox="0 0 240 18" style={style} aria-hidden="true">
      <path d="M0 2 H240 V8 L228 8 V14 L222 14 V10 L200 10 V16 L194 16 V10
               L160 10 V14 L154 14 V10 L120 10 V18 L114 18 V10
               L80 10 V14 L74 14 V10 L40 10 V16 L34 16 V10 L0 10 Z"
            fill={color}/>
    </svg>
  );
}

/* ─────────────────────────  building blocks  ───────────────────────── */

function NavLink({ children, onClick, active }) {
  return (
    <button onClick={onClick} style={{
      fontFamily: BODY, fontSize: '0.85rem', fontWeight: 600,
      color: active ? C.coral : C.ink, background: 'transparent',
      border: 'none', cursor: 'pointer', padding: '6px 10px', borderRadius: 6,
    }}>
      {children}
    </button>
  );
}

function Pillar({ icon, title, body, accent }) {
  return (
    <div style={{
      background: C.paper,
      border: `2px solid ${C.ink}`,
      borderRadius: 18,
      padding: '1.4rem 1.2rem',
      boxShadow: `5px 5px 0 0 ${accent}`,
      transition: 'transform 0.15s ease, box-shadow 0.15s ease',
    }}
      onMouseEnter={e => { e.currentTarget.style.transform = 'translate(-2px, -2px)'; e.currentTarget.style.boxShadow = `8px 8px 0 0 ${accent}`; }}
      onMouseLeave={e => { e.currentTarget.style.transform = 'translate(0, 0)'; e.currentTarget.style.boxShadow = `5px 5px 0 0 ${accent}`; }}
    >
      <div style={{ width: 48, height: 48, borderRadius: 12, background: accent,
        display: 'grid', placeItems: 'center', marginBottom: '0.9rem' }}>
        {icon}
      </div>
      <h3 style={{ fontFamily: DISPLAY, fontSize: '1.05rem', color: C.ink,
        margin: '0 0 0.4rem', letterSpacing: '0.02em' }}>{title}</h3>
      <p style={{ fontFamily: BODY, fontSize: '0.88rem', color: C.inkSoft,
        margin: 0, lineHeight: 1.55 }}>{body}</p>
    </div>
  );
}

function Step({ n, title, body, color }) {
  return (
    <div style={{ display: 'flex', gap: '0.9rem', alignItems: 'flex-start' }}>
      <div style={{
        flex: '0 0 44px', width: 44, height: 44, borderRadius: '50%',
        background: color, color: C.ink,
        fontFamily: DISPLAY, fontSize: '1rem',
        display: 'grid', placeItems: 'center',
        border: `2px solid ${C.ink}`,
      }}>{n}</div>
      <div>
        <div style={{ fontFamily: DISPLAY, fontSize: '0.95rem', color: C.ink, marginBottom: 2 }}>{title}</div>
        <div style={{ fontFamily: BODY, fontSize: '0.85rem', color: C.inkSoft, lineHeight: 1.5 }}>{body}</div>
      </div>
    </div>
  );
}

/* ─────────────────────────  main screen  ───────────────────────── */

export default function WelcomeScreen({ onStart, onNavigate }) {
  return (
    <div style={{
      minHeight: '100vh', background: C.cream, color: C.ink,
      fontFamily: BODY, overflowX: 'hidden', position: 'relative',
    }}>
      {/* Decorative splats positioned in the background */}
      <Splat color={C.violet} style={{ position: 'absolute', top: 80, right: -30, width: 180, height: 180, opacity: 0.18, transform: 'rotate(20deg)' }} />
      <Splat color={C.teal}   style={{ position: 'absolute', top: 420, left: -40, width: 160, height: 160, opacity: 0.14, transform: 'rotate(-25deg)' }} />
      <Splat color={C.sun}    style={{ position: 'absolute', top: 880, right: -20, width: 140, height: 140, opacity: 0.20, transform: 'rotate(45deg)' }} />

      {/* ─────────────────  TOP NAV  ───────────────── */}
      <nav style={{
        position: 'sticky', top: 0, zIndex: 20,
        background: 'rgba(255, 246, 230, 0.92)',
        backdropFilter: 'blur(8px)',
        borderBottom: `2px solid ${C.ink}`,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '12px 20px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <HandMark color={C.coral} size={28} />
          <span style={{ fontFamily: DISPLAY, fontSize: '1.25rem', color: C.ink, letterSpacing: '0.04em' }}>
            MLAF
          </span>
        </div>
        <div style={{ display: 'flex', gap: '0.1rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <NavLink onClick={() => onNavigate && onNavigate('HOME')}>Home</NavLink>
          <NavLink onClick={() => onNavigate && onNavigate('ABOUT')}>About</NavLink>
          <NavLink onClick={() => onNavigate && onNavigate('HANDOUT')}>Guide</NavLink>
          <NavLink onClick={() => onNavigate && onNavigate('BENGALI')}>বাংলা</NavLink>
          <NavLink onClick={() => onNavigate && onNavigate('CONTACT')}>Contact</NavLink>
        </div>
      </nav>

      {/* ─────────────────  WELCOME-BACK (returning users only)  ───────────────── */}
      <WelcomeBack onResume={onStart} />

      {/* ─────────────────  HERO  ───────────────── */}
      <section style={{
        maxWidth: 980, margin: '0 auto',
        padding: '60px 22px 24px',
        position: 'relative', zIndex: 1,
      }}>
        <div style={{
          display: 'inline-block',
          background: C.sun, color: C.ink,
          fontFamily: BODY, fontSize: '0.72rem', fontWeight: 700,
          letterSpacing: '0.1em', textTransform: 'uppercase',
          padding: '6px 12px', border: `2px solid ${C.ink}`, borderRadius: 4,
          marginBottom: '1.4rem',
          boxShadow: `3px 3px 0 0 ${C.ink}`,
        }}>
          Free · Browser · Camera-only
        </div>

        <h1 style={{
          fontFamily: DISPLAY,
          fontSize: 'clamp(2.4rem, 7vw, 4.6rem)',
          color: C.ink, lineHeight: 1.05, margin: '0 0 0.6rem',
          letterSpacing: '-0.005em',
        }}>
          Talk with your<br />
          <span style={{ position: 'relative', display: 'inline-block' }}>
            <span style={{ color: C.coral }}>hands.</span>
            <Drips color={C.coral} style={{
              position: 'absolute', left: 0, bottom: '-12px',
              width: '100%', maxWidth: 280, height: 18,
            }} />
          </span>
        </h1>

        <p style={{
          fontFamily: BODY, fontSize: '1.08rem', color: C.inkSoft,
          maxWidth: 580, lineHeight: 1.6, margin: '1.4rem 0 1.8rem',
        }}>
          A camera watches your hand. A wave becomes a word — a word becomes
          a sentence — and a sentence, at last, is spoken out loud. For every
          child whose voice has been waiting. For every parent who has waited
          beside them.
        </p>

        <div style={{ display: 'flex', gap: '0.8rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <button onClick={onStart} style={{
            fontFamily: DISPLAY, fontSize: '1.05rem',
            color: C.cream, background: C.ink,
            border: `2px solid ${C.ink}`, borderRadius: 12,
            padding: '14px 28px', cursor: 'pointer',
            letterSpacing: '0.03em',
            boxShadow: `5px 5px 0 0 ${C.coral}`,
            transition: 'transform 0.12s ease, box-shadow 0.12s ease',
          }}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translate(-2px, -2px)'; e.currentTarget.style.boxShadow = `7px 7px 0 0 ${C.coral}`; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translate(0, 0)'; e.currentTarget.style.boxShadow = `5px 5px 0 0 ${C.coral}`; }}
          >
            Start now →
          </button>
          <button onClick={() => onNavigate && onNavigate('HANDOUT')} style={{
            fontFamily: BODY, fontSize: '0.92rem', fontWeight: 700,
            color: C.ink, background: 'transparent',
            border: `2px solid ${C.ink}`, borderRadius: 12,
            padding: '12px 22px', cursor: 'pointer',
          }}>
            Read the guide
          </button>
        </div>
      </section>

      {/* ─────────────────  PILLARS (3, honest)  ───────────────── */}
      <section style={{
        maxWidth: 980, margin: '40px auto', padding: '0 22px',
        position: 'relative', zIndex: 1,
      }}>
        <div style={{
          display: 'grid', gap: '1.2rem',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        }}>
          <Pillar
            accent={C.coral}
            title="No hardware to buy"
            body="Works on any laptop, phone, or tablet with a camera. No special device. No tablet to ship. Open the browser, you're in."
            icon={<HandMark color={C.ink} size={28} />}
          />
          <Pillar
            accent={C.violet}
            title="Built for motor profiles"
            body="Eight profiles — including dedicated calibration for spastic, athetoid, ataxic, and mixed cerebral palsy, plus eye-gaze fallback for users who can't move their hands."
            icon={
              <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">
                <circle cx="16" cy="16" r="13" fill="none" stroke={C.ink} strokeWidth="2.5"/>
                <circle cx="16" cy="16" r="4"  fill={C.ink}/>
              </svg>
            }
          />
          <Pillar
            accent={C.teal}
            title="Speaks for them"
            body="Every grammatical sentence the system builds is spoken aloud through your device. Their words. Out loud. For the first time, for many."
            icon={
              <svg viewBox="0 0 32 32" width="28" height="28" aria-hidden="true">
                <path d="M6 12 H12 L20 6 V26 L12 20 H6 Z" fill={C.ink}/>
                <path d="M24 10 Q28 16 24 22"  fill="none" stroke={C.ink} strokeWidth="2.5" strokeLinecap="round"/>
              </svg>
            }
          />
        </div>
      </section>

      {/* ─────────────────  HOW IT WORKS  ───────────────── */}
      <section style={{
        maxWidth: 880, margin: '70px auto 40px', padding: '0 22px',
        position: 'relative', zIndex: 1,
      }}>
        <div style={{ marginBottom: '1.8rem' }}>
          <div style={{ fontFamily: BODY, fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.18em', textTransform: 'uppercase', color: C.coral, marginBottom: 6 }}>
            Four steps
          </div>
          <h2 style={{ fontFamily: DISPLAY, fontSize: 'clamp(1.6rem, 4vw, 2.4rem)', color: C.ink, margin: 0, letterSpacing: '-0.005em' }}>
            How it works
          </h2>
        </div>

        <div style={{ display: 'grid', gap: '1.4rem',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))' }}>
          <Step n="1" color={C.coral}  title="Pick your profile"
                body="Tell MLAF whether you're a learner, a CP family, an SLP, or whoever you are. The system calibrates to you." />
          <Step n="2" color={C.violet} title="Make a gesture"
                body="Form a hand-shape in front of the camera. Hold it (or just pass through, for fast profiles). The lock bar fills." />
          <Step n="3" color={C.teal}   title="Build the sentence"
                body="Subject, verb, object. The grammar engine checks every word as it lands. Bad combos won't lock." />
          <Step n="4" color={C.sun}    title="Hear it spoken"
                body="When the sentence is whole, it speaks. Loud. Clear. Their idea, in the room." />
        </div>
      </section>

      {/* ─────────────────  INVITE  ───────────────── */}
      <section style={{
        maxWidth: 760, margin: '50px auto 0', padding: '0 22px',
        position: 'relative', zIndex: 1,
      }}>
        <div style={{
          background: C.ink, color: C.cream, borderRadius: 22,
          padding: '2rem 1.6rem', textAlign: 'center',
          border: `2px solid ${C.ink}`,
          boxShadow: `6px 6px 0 0 ${C.violet}`,
          position: 'relative', overflow: 'hidden',
        }}>
          <Splat color={C.coral} style={{ position: 'absolute', top: -30, right: -30, width: 120, height: 120, opacity: 0.5 }} />
          <Splat color={C.teal}  style={{ position: 'absolute', bottom: -40, left: -30, width: 110, height: 110, opacity: 0.4 }} />

          <h3 style={{ fontFamily: DISPLAY, fontSize: 'clamp(1.4rem, 3.5vw, 2rem)',
            color: C.cream, margin: '0 0 0.6rem', letterSpacing: '-0.005em', position: 'relative' }}>
            Try it. Right now. No signup.
          </h3>
          <p style={{ fontFamily: BODY, fontSize: '0.95rem', color: 'rgba(255,246,230,0.85)',
            maxWidth: 460, margin: '0 auto 1.4rem', lineHeight: 1.55, position: 'relative' }}>
            Open the camera. Wave hello. Form a fist. Watch the word "I" land
            on the screen. You'll know in 30 seconds whether MLAF works for you.
          </p>
          <button onClick={onStart} style={{
            fontFamily: DISPLAY, fontSize: '1rem',
            color: C.ink, background: C.sun,
            border: `2px solid ${C.cream}`, borderRadius: 12,
            padding: '12px 26px', cursor: 'pointer', letterSpacing: '0.03em',
            position: 'relative',
            boxShadow: `4px 4px 0 0 ${C.coral}`,
          }}>
            Open the camera →
          </button>
        </div>
      </section>

      {/* ─────────────────  FOOTER  ───────────────── */}
      <footer style={{
        maxWidth: 980, margin: '60px auto 0', padding: '24px 22px',
        borderTop: `2px solid ${C.ink}`,
        display: 'flex', flexWrap: 'wrap', gap: '0.8rem 1.4rem',
        justifyContent: 'space-between', alignItems: 'center',
        fontFamily: BODY, fontSize: '0.78rem', color: C.inkSoft,
      }}>
        <div>
          <strong style={{ color: C.ink }}>Neil Shankar Ray</strong> · Applied Linguist (MA, 15 yrs) · PhD candidate (Comp. Ling.) + AI/ML cert, IIT&nbsp;Patna<br/>
          <span style={{ color: C.inkSoft }}>MLAF · BSL 1.1 · Indian patent app. 202631020540 (10 claims)</span>
        </div>
        <div style={{ display: 'flex', gap: '0.9rem', flexWrap: 'wrap' }}>
          <a href="https://github.com/Ishu38/gesture-grammar" style={{ color: C.ink, textDecoration: 'underline' }}>GitHub</a>
          <a href="/guide.pdf" style={{ color: C.ink, textDecoration: 'underline' }}>Guide (EN+BN)</a>
          <a href="mailto:neilshankarray@vaaani.in" style={{ color: C.ink, textDecoration: 'underline' }}>neilshankarray@vaaani.in</a>
        </div>
      </footer>
    </div>
  );
}
