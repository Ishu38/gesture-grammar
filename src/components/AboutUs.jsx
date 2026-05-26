/**
 * AboutUs.jsx — About MLAF and the creator
 */

const STYLE = {
  page: {
    minHeight: '100vh',
    background: 'linear-gradient(160deg, #05070d 0%, #0c1220 30%, #111827 60%, #0f172a 100%)',
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    color: '#e2e8f0',
    padding: '80px 2rem 4rem',
  },
  nav: {
    position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000,
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '14px 24px',
    background: 'rgba(5,7,13,0.85)', backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)', borderBottom: '1px solid rgba(255,255,255,0.05)',
  },
  navBrand: { fontSize: '0.85rem', fontWeight: 800, color: '#4ade80', letterSpacing: '0.15em', cursor: 'pointer', background: 'none', border: 'none' },
  navLinks: { display: 'flex', gap: '0.5rem', alignItems: 'center' },
  navLink: {
    fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600,
    letterSpacing: '0.04em', cursor: 'pointer', background: 'none', border: 'none',
    padding: '6px 14px', borderRadius: 6, transition: 'all 0.15s',
  },
  section: {
    maxWidth: '720px', margin: '0 auto 3rem',
  },
  heading: {
    fontSize: '1.6rem', fontWeight: 800, color: '#f1f5f9', margin: '0 0 0.4rem',
  },
  subheading: {
    fontSize: '0.75rem', color: '#4ade80', fontWeight: 700, letterSpacing: '0.1em',
    textTransform: 'uppercase', marginBottom: '1.5rem',
  },
  body: {
    fontSize: '0.88rem', color: '#94a3b8', lineHeight: 1.75,
  },
  card: {
    background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: 14, padding: '1.5rem', marginBottom: '1rem',
  },
};

function NavButton({ children, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      ...STYLE.navLink,
      color: active ? '#4ade80' : STYLE.navLink.color,
      background: active ? 'rgba(74,222,128,0.08)' : 'transparent',
    }}>
      {children}
    </button>
  );
}

export default function AboutUs({ onNavigate }) {
  return (
    <div style={STYLE.page}>
      {/* Nav */}
      <div style={STYLE.nav}>
        <button onClick={() => onNavigate('HOME')} style={STYLE.navBrand}>MLAF</button>
        <div style={STYLE.navLinks}>
          <NavButton onClick={() => onNavigate('HOME')}>Home</NavButton>
          <NavButton active onClick={() => onNavigate('ABOUT')}>About Us</NavButton>
          <NavButton onClick={() => onNavigate('CONTACT')}>Contact Us</NavButton>
        </div>
      </div>

      <div style={{ ...STYLE.section, paddingTop: '2rem' }}>
        <div style={STYLE.subheading}>About MLAF</div>
        <h2 style={STYLE.heading}>
          Multimodal Language Acquisition Framework
        </h2>
        <p style={STYLE.body}>
          MLAF is a patented, browser-based platform that teaches English grammar
          to children with motor impairment, cerebral palsy, non-verbal autism,
          and deaf/hard-of-hearing learners — using <strong style={{ color: '#e2e8f0' }}>hand gestures</strong> detected by
          a standard phone or laptop camera.
        </p>
        <p style={STYLE.body}>
          Built on a neuro-symbolic architecture combining MediaPipe neural perception,
          ONNX Runtime classification, Bayesian trimodal fusion (vision + acoustics + gaze),
          and a formal X-bar grammar engine via SWI-Prolog. The system adapts to each
          learner's motor abilities in real time, detects Indian Sign Language syntactic
          transfer, and generates diagnostic-prescriptive reports showing exactly what
          each learner knows.
        </p>
        <p style={STYLE.body}>
          Patent Pending — Provisional application TEMP/E-1/22951/2026-KOL filed with
          the Indian Patent Office (2026). 10 claims covering the multimodal fusion
          architecture, abductive feedback loop, and gesture-to-grammar mapping system.
        </p>
      </div>

      <div style={STYLE.section}>
        <div style={STYLE.subheading}>The Creator</div>
        <h2 style={STYLE.heading}>Neil Shankar Ray</h2>

        <div style={STYLE.card}>
          <div style={{
            display: 'grid', gridTemplateColumns: '140px 1fr', gap: '0.6rem 1rem',
            alignItems: 'baseline',
          }}>
            {[
              ['Current', 'Pursuing PhD in Computational Linguistics, IIT Patna'],
              ['Degree', 'MA in English, Central University'],
              ['Teaching', 'B.Ed (Bachelor of Education)'],
              ['Certification', 'CELTA — University of Cambridge'],
              ['Certification', 'TESOL (Teaching English to Speakers of Other Languages)'],
              ['Experience', '15 years of classroom teaching as Senior ESL Facilitator'],
              ['Research', 'NLP & Speech AI Engineering · Neuro-Symbolic AI · Applied Linguistics'],
              ['Patent', 'MLAF — Multimodal Language Acquisition Framework (TEMP/E-1/22951/2026-KOL)'],
            ].map(([label, value]) => (
              <>
                <div key={label + 'l'} style={{
                  fontSize: '0.7rem', color: '#64748b', fontWeight: 600,
                  letterSpacing: '0.04em', textTransform: 'uppercase',
                }}>
                  {label}
                </div>
                <div key={label + 'v'} style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.5 }}>
                  {value}
                </div>
              </>
            ))}
          </div>
        </div>

        <p style={STYLE.body}>
          Neil Shankar Ray is an Applied Linguist and Speech AI Engineer with
          15 years of classroom teaching experience as a Senior ESL Facilitator.
          He holds an MA in English from Central University, a B.Ed, the CELTA
          from the University of Cambridge, and TESOL certification. Currently
          pursuing a PhD in Computational Linguistics at IIT Patna, his research
          focuses on neuro-symbolic AI for accessible language acquisition —
          specifically, multimodal systems that adapt to learners with motor,
          sensory, and cognitive disabilities.
        </p>
      </div>
    </div>
  );
}
