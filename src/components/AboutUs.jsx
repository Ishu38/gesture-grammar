/**
 * AboutUs.jsx — graffiti-vibe About page, matching WelcomeScreen / ContactUs.
 * Preserves the creator's bio facts; only the visual layer changed.
 * Patent number updated TEMP/E-1/22951/2026-KOL → 202631020540.
 * "Current" credential set to AI/ML cert at IIT Patna per latest user statement.
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

function HandMark({ color = C.coral, size = 28 }) {
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

function CredRow({ label, value, accent }) {
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: 'minmax(110px, 140px) 1fr',
      gap: '0.6rem 1.2rem', alignItems: 'baseline', padding: '0.55rem 0',
      borderBottom: `1px dashed ${C.ink}22`,
    }}>
      <div style={{
        fontFamily: BODY, fontSize: '0.7rem', fontWeight: 700,
        letterSpacing: '0.1em', textTransform: 'uppercase',
        color: accent || C.inkSoft,
      }}>
        {label}
      </div>
      <div style={{ fontFamily: BODY, fontSize: '0.92rem', color: C.ink, lineHeight: 1.5 }}>
        {value}
      </div>
    </div>
  );
}

export default function AboutUs({ onNavigate }) {
  return (
    <div style={{
      minHeight: '100vh', background: C.cream, color: C.ink,
      fontFamily: BODY, overflowX: 'hidden', position: 'relative',
    }}>
      <Splat color={C.violet} style={{ position: 'absolute', top: 100, right: -40, width: 180, height: 180, opacity: 0.18, transform: 'rotate(18deg)' }} />
      <Splat color={C.teal}   style={{ position: 'absolute', top: 540, left: -50, width: 160, height: 160, opacity: 0.14, transform: 'rotate(-22deg)' }} />
      <Splat color={C.sun}    style={{ position: 'absolute', top: 1080, right: -20, width: 150, height: 150, opacity: 0.22, transform: 'rotate(38deg)' }} />

      <nav style={{
        position: 'sticky', top: 0, zIndex: 20,
        background: 'rgba(255, 246, 230, 0.92)', backdropFilter: 'blur(8px)',
        borderBottom: `2px solid ${C.ink}`,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '12px 20px',
      }}>
        <button onClick={() => onNavigate('HOME')} style={{
          display: 'flex', alignItems: 'center', gap: '0.6rem',
          background: 'transparent', border: 'none', cursor: 'pointer',
        }}>
          <HandMark color={C.coral} size={28}/>
          <span style={{ fontFamily: DISPLAY, fontSize: '1.25rem', color: C.ink, letterSpacing: '0.04em' }}>MLAF</span>
        </button>
        <div style={{ display: 'flex', gap: '0.1rem' }}>
          <NavLink onClick={() => onNavigate('HOME')}>Home</NavLink>
          <NavLink active onClick={() => onNavigate('ABOUT')}>About</NavLink>
          <NavLink onClick={() => onNavigate('CONTACT')}>Contact</NavLink>
        </div>
      </nav>

      {/* ── About MLAF ── */}
      <section style={{
        maxWidth: 840, margin: '0 auto', padding: '50px 22px 20px',
        position: 'relative', zIndex: 1,
      }}>
        <div style={{
          display: 'inline-block', background: C.sun, color: C.ink,
          fontFamily: BODY, fontSize: '0.72rem', fontWeight: 700,
          letterSpacing: '0.1em', textTransform: 'uppercase',
          padding: '6px 12px', border: `2px solid ${C.ink}`, borderRadius: 4,
          marginBottom: '1.4rem', boxShadow: `3px 3px 0 0 ${C.ink}`,
        }}>
          About MLAF
        </div>
        <h1 style={{
          fontFamily: DISPLAY, fontSize: 'clamp(2rem, 5.5vw, 3.2rem)',
          color: C.ink, lineHeight: 1.1, margin: '0 0 1rem', letterSpacing: '-0.005em',
        }}>
          Multimodal Language<br/>
          <span style={{ color: C.coral }}>Acquisition Framework.</span>
        </h1>

        <p style={{
          fontFamily: BODY, fontSize: '1rem', color: C.inkSoft, lineHeight: 1.65, margin: '0 0 1rem',
        }}>
          MLAF is a patented, browser-based platform that teaches English grammar
          to children with motor impairment, cerebral palsy, non-verbal autism,
          and deaf/hard-of-hearing learners — using <strong style={{ color: C.ink }}>hand
          gestures</strong> detected by a standard phone or laptop camera.
        </p>
        <p style={{ fontFamily: BODY, fontSize: '1rem', color: C.inkSoft, lineHeight: 1.65, margin: '0 0 1rem' }}>
          Built on a neuro-symbolic architecture combining MediaPipe neural
          perception, ONNX Runtime classification, Bayesian trimodal fusion
          (vision + acoustics + gaze), and a formal X-bar grammar engine via
          SWI-Prolog. The system adapts to each learner's motor abilities in
          real time, detects Indian Sign Language syntactic transfer, and
          generates diagnostic-prescriptive reports showing exactly what each
          learner knows.
        </p>
        <p style={{ fontFamily: BODY, fontSize: '0.92rem', color: C.inkSoft, lineHeight: 1.6, margin: 0 }}>
          <strong style={{ color: C.ink }}>Patent Pending</strong> — Indian
          application no. <strong style={{ color: C.ink }}>202631020540</strong>,
          10 claims covering the multimodal fusion architecture, abductive
          feedback loop, and gesture-to-grammar mapping system. PCT planned by
          21 February 2027.
        </p>
      </section>

      {/* ── Creator ── */}
      <section style={{
        maxWidth: 840, margin: '40px auto 0', padding: '0 22px',
        position: 'relative', zIndex: 1,
      }}>
        <div style={{
          fontFamily: BODY, fontSize: '0.72rem', fontWeight: 700,
          letterSpacing: '0.18em', textTransform: 'uppercase', color: C.coral, marginBottom: 6,
        }}>
          The creator
        </div>
        <h2 style={{ fontFamily: DISPLAY, fontSize: 'clamp(1.6rem, 4vw, 2.4rem)', color: C.ink, margin: '0 0 1.2rem', letterSpacing: '-0.005em' }}>
          Neil Shankar Ray
        </h2>

        <div style={{
          background: C.paper, border: `2px solid ${C.ink}`, borderRadius: 18,
          padding: '1.4rem 1.4rem 0.6rem', boxShadow: `6px 6px 0 0 ${C.violet}`,
          marginBottom: '1.4rem',
        }}>
          <CredRow label="Current"       accent={C.coral}  value="Pursuing PhD in Computational Linguistics, IIT Patna"/>
          <CredRow label="Current"       accent={C.coral}  value="AI / ML certification, IIT Patna"/>
          <CredRow label="Degree"        accent={C.violet} value="MA in English, Central University"/>
          <CredRow label="Teaching"      accent={C.teal}   value="B.Ed (Bachelor of Education)"/>
          <CredRow label="Certification" accent={C.coral}  value="CELTA — University of Cambridge"/>
          <CredRow label="Certification" accent={C.violet} value="TESOL — Teaching English to Speakers of Other Languages"/>
          <CredRow label="Experience"    accent={C.teal}   value="15 years of classroom teaching as Senior ESL Facilitator"/>
          <CredRow label="Research"      accent={C.coral}  value="NLP & Speech AI Engineering · Neuro-Symbolic AI · Applied Linguistics"/>
          <CredRow label="Patent"        accent={C.violet} value="MLAF — Multimodal Language Acquisition Framework (Indian app. 202631020540)"/>
        </div>

        <p style={{ fontFamily: BODY, fontSize: '0.98rem', color: C.inkSoft, lineHeight: 1.65, margin: 0 }}>
          Neil Shankar Ray is an Applied Linguist and Speech-AI engineer with
          15 years of classroom teaching experience as a Senior ESL Facilitator.
          He holds an MA in English from Central University, a B.Ed, the CELTA
          from the University of Cambridge, and TESOL certification. He is
          currently pursuing a PhD in Computational Linguistics at IIT Patna,
          alongside an AI/ML certification at the same institute. His research
          focuses on neuro-symbolic AI for accessible language acquisition —
          specifically, multimodal systems that adapt to learners with motor,
          sensory, and cognitive disabilities.
        </p>
      </section>

      {/* ── Footer ── */}
      <footer style={{
        maxWidth: 840, margin: '60px auto 0', padding: '24px 22px',
        borderTop: `2px solid ${C.ink}`, display: 'flex', flexWrap: 'wrap',
        gap: '0.8rem 1.4rem', justifyContent: 'space-between', alignItems: 'center',
        fontFamily: BODY, fontSize: '0.78rem', color: C.inkSoft,
      }}>
        <div>
          <strong style={{ color: C.ink }}>MLAF</strong> · BSL 1.1 · Indian patent app. 202631020540
        </div>
        <div style={{ display: 'flex', gap: '0.9rem', flexWrap: 'wrap' }}>
          <a href="mailto:neilshankarray@vaaani.in" style={{ color: C.ink, textDecoration: 'underline' }}>neilshankarray@vaaani.in</a>
          <a href="https://github.com/Ishu38/gesture-grammar" style={{ color: C.ink, textDecoration: 'underline' }}>GitHub</a>
          <button onClick={() => onNavigate('HOME')} style={{
            fontFamily: BODY, fontSize: '0.8rem', fontWeight: 700, color: C.ink,
            background: 'transparent', border: `2px solid ${C.ink}`, borderRadius: 8,
            padding: '6px 14px', cursor: 'pointer',
          }}>← Back home</button>
        </div>
      </footer>
    </div>
  );
}
