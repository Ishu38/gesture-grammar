/**
 * WelcomeScreen.jsx — Professional landing page for MLAF
 * Designed for educators, funders, and institutional decision-makers.
 * First impression: credibility, clarity, confidence.
 */

const BRAND = {
  green:   '#4ade80',
  cyan:    '#22d3ee',
  purple:  '#8b5cf6',
  blue:    '#3b82f6',
  amber:   '#f59e0b',
  rose:    '#f43f5e',
  surface: 'rgba(255,255,255,0.025)',
  border:  'rgba(255,255,255,0.07)',
  muted:   '#94a3b8',
  dim:     '#64748b',
};

const style = {
  screen: {
    minHeight: '100vh',
    background: 'linear-gradient(160deg, #05070d 0%, #0c1220 30%, #111827 60%, #0f172a 100%)',
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    color: '#e2e8f0',
    overflowX: 'hidden',
    position: 'relative',
  },
  nav: {
    position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000,
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '12px 24px',
    background: 'rgba(5,7,13,0.8)', backdropFilter: 'blur(12px)',
    WebkitBackdropFilter: 'blur(12px)', borderBottom: '1px solid rgba(255,255,255,0.04)',
  },
  navBrand: { fontSize: '0.8rem', fontWeight: 800, color: '#4ade80', letterSpacing: '0.15em' },
  navRight: { display: 'flex', gap: '1rem', alignItems: 'center' },
  navLink: {
    fontSize: '0.7rem', color: '#94a3b8', fontWeight: 600,
    letterSpacing: '0.05em', textDecoration: 'none',
    cursor: 'pointer', background: 'none', border: 'none',
  },
  badge: {
    padding: '3px 10px', borderRadius: 20, fontSize: '0.6rem', fontWeight: 700,
    letterSpacing: '0.08em', background: 'rgba(74,222,128,0.08)',
    border: '1px solid rgba(74,222,128,0.2)', color: '#4ade80',
  },
};

function NavButton({ children, onClick }) {
  return (
    <button onClick={onClick} style={{
      fontSize: '0.7rem', color: BRAND.muted, fontWeight: 600,
      letterSpacing: '0.04em', background: 'none', border: 'none',
      cursor: 'pointer', padding: '6px 12px', borderRadius: 6,
      transition: 'all 0.15s',
    }}
      onMouseEnter={e => { e.target.style.color = '#e2e8f0'; e.target.style.background = 'rgba(255,255,255,0.05)'; }}
      onMouseLeave={e => { e.target.style.color = BRAND.muted; e.target.style.background = 'none'; }}
    >
      {children}
    </button>
  );
}

function StatPill({ value, label }) {
  return (
    <div style={{ textAlign: 'center', padding: '0 1rem', borderLeft: `1px solid ${BRAND.border}` }}>
      <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#fff', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </div>
      <div style={{ fontSize: '0.6rem', color: BRAND.dim, marginTop: 2, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
        {label}
      </div>
    </div>
  );
}

function FeatureCard({ number, title, desc, highlights, color }) {
  return (
    <div style={{
      background: BRAND.surface,
      border: `1px solid ${BRAND.border}`,
      borderRadius: 16, padding: '1.5rem',
      transition: 'border-color 0.25s, background 0.25s, transform 0.25s',
      cursor: 'default',
    }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = `${color}40`;
        e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
        e.currentTarget.style.transform = 'translateY(-2px)';
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = BRAND.border;
        e.currentTarget.style.background = BRAND.surface;
        e.currentTarget.style.transform = 'none';
      }}
    >
      <div style={{
        width: 36, height: 36, borderRadius: 10,
        background: `${color}12`, color,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '0.85rem', fontWeight: 800, marginBottom: '1rem',
        border: `1px solid ${color}20`,
      }}>
        {String(number).padStart(2, '0')}
      </div>
      <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f1f5f9', margin: '0 0 0.5rem' }}>
        {title}
      </h3>
      <p style={{ fontSize: '0.78rem', color: BRAND.muted, lineHeight: 1.6, margin: '0 0 0.75rem' }}>
        {desc}
      </p>
      {highlights && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
          {highlights.map((h, i) => (
            <span key={i} style={{
              fontSize: '0.62rem', padding: '2px 8px', borderRadius: 4,
              background: `${color}08`, color, fontWeight: 600,
              letterSpacing: '0.03em',
            }}>
              {h}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function WelcomeScreen({ onStart, onHandout, onNavigate }) {
  return (
    <div style={style.screen}>
      {/* ── TOP NAV ── */}
      <div style={style.nav}>
        <span style={style.navBrand}>MLAF</span>
        <div style={style.navRight}>
          <NavButton onClick={() => onNavigate && onNavigate('HOME')}>Home</NavButton>
          <NavButton onClick={() => onNavigate && onNavigate('ABOUT')}>About Us</NavButton>
          <NavButton onClick={() => onNavigate && onNavigate('CONTACT')}>Contact Us</NavButton>
          <span style={{ marginLeft: '0.75rem', ...style.badge, fontSize: '0.58rem' }}>PATENT PENDING</span>
          {onHandout && <NavButton onClick={onHandout}>Handout</NavButton>}
        </div>
      </div>

      {/* ── HERO ── */}
      <div style={{
        padding: '120px 2rem 60px', maxWidth: '800px', margin: '0 auto',
        textAlign: 'center',
      }}>
        <h1 style={{
          fontSize: 'clamp(2.2rem, 5vw, 3.5rem)', fontWeight: 900,
          color: '#ffffff', lineHeight: 1.1, margin: 0,
          letterSpacing: '-0.02em',
        }}>
          English grammar for children<br />
          <span style={{ color: '#4ade80' }}>who cannot write, type, or speak.</span>
        </h1>

        <p style={{
          fontSize: '1.05rem', color: BRAND.muted, marginTop: '1.25rem',
          lineHeight: 1.65, maxWidth: '560px', marginLeft: 'auto', marginRight: 'auto',
        }}>
          MLAF teaches sentence structure through hand gestures detected by
          any phone camera. Adaptive AI calibrates to motor impairment,
          detects Indian Sign Language transfer, and speaks completed
          sentences aloud — becoming the child's voice.
        </p>

        {/* CTA */}
        <div style={{ marginTop: '2rem', display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <button onClick={onStart} style={{
            padding: '0.85rem 2.5rem', fontSize: '0.95rem', fontWeight: 700,
            color: '#05070d', background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
            border: 'none', borderRadius: 12, cursor: 'pointer', letterSpacing: '0.03em',
            boxShadow: '0 4px 24px rgba(74,222,128,0.25)',
            transition: 'transform 0.15s, box-shadow 0.15s',
          }}
            onMouseEnter={e => { e.target.style.transform = 'translateY(-1px)'; e.target.style.boxShadow = '0 8px 32px rgba(74,222,128,0.35)'; }}
            onMouseLeave={e => { e.target.style.transform = 'none'; e.target.style.boxShadow = '0 4px 24px rgba(74,222,128,0.25)'; }}
          >
            Try the Demo →
          </button>
          {onHandout && (
            <button onClick={onHandout} style={{
              padding: '0.85rem 2rem', fontSize: '0.9rem', fontWeight: 600,
              color: BRAND.muted, background: BRAND.surface,
              border: `1px solid ${BRAND.border}`, borderRadius: 12,
              cursor: 'pointer', letterSpacing: '0.03em', transition: 'all 0.15s',
            }}
              onMouseEnter={e => { e.target.style.borderColor = 'rgba(255,255,255,0.2)'; e.target.style.color = '#e2e8f0'; }}
              onMouseLeave={e => { e.target.style.borderColor = BRAND.border; e.target.style.color = BRAND.muted; }}
            >
              One-Page Handout
            </button>
          )}
        </div>
      </div>

      {/* ── STATS ── */}
      <div style={{
        display: 'flex', justifyContent: 'center', gap: '0',
        width: 'fit-content', margin: '0 auto', padding: '1.5rem 2rem',
        background: BRAND.surface, borderRadius: 14,
        border: `1px solid ${BRAND.border}`,
        marginBottom: '4rem',
      }}>
        <StatPill value="10" label="Patent claims" />
        <StatPill value="99.5%" label="CNN accuracy" />
        <StatPill value="19" label="Gestures" />
        <StatPill value="In-browser" label="Zero cloud data" />
        <StatPill value="PWA" label="Works offline" />
      </div>

      {/* ── FEATURES GRID ── */}
      <div style={{ maxWidth: '900px', margin: '0 auto', padding: '0 2rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <div style={{ fontSize: '0.7rem', color: BRAND.dim, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            What makes MLAF different
          </div>
          <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#f1f5f9', margin: 0, lineHeight: 1.3 }}>
            Adaptive. Explainable. Built for real classrooms.
          </h2>
        </div>

        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: '1rem', marginBottom: '3rem',
        }}>
          <FeatureCard
            number={1} title="Motor-Adaptive Recognition"
            color={BRAND.green}
            desc="Camera detects hand tremor via positional jitter analysis. Tolerance bands auto-widen as motor instability increases — the system gets MORE forgiving, not less."
            highlights={['Real-time jitter', '6-frame hysteresis', 'Fatigue detection']}
          />
          <FeatureCard
            number={2} title="ISL Transfer Correction"
            color={BRAND.purple}
            desc="Detects when deaf learners default to Indian Sign Language SOV word order and guides toward English SVO. Explicit contrastive feedback with linguistic rationale."
            highlights={['SOV→SVO reorder', 'Pro-drop detection', 'Topic fronting fix']}
          />
          <FeatureCard
            number={3} title="Bayesian Knowledge Tracing"
            color={BRAND.blue}
            desc="Estimates P(known) for every grammar concept after every gesture. Predicts retention decay, identifies transfer latency. No more guesswork — data-driven pedagogy."
            highlights={['Per-concept P(know)', 'Decay forecasting', 'Transfer tracking']}
          />
          <FeatureCard
            number={4} title="Explainable AI"
            color={BRAND.amber}
            desc="Every error generates a human-readable explanation — not a black box. Educators and therapists see WHY the system made each decision, traced to specific linguistic rules."
            highlights={['Root cause analysis', 'Remediation path', 'Session narratives']}
          />
          <FeatureCard
            number={5} title="Neuro-Symbolic Architecture"
            color={BRAND.rose}
            desc="MediaPipe neural perception + ONNX RF classifier + Chomskyan X-bar grammar via SWI-Prolog + Bayesian trimodal fusion. AI that reasons, not just pattern-matches."
            highlights={['Bayesian fusion', 'Earley parser', 'Graph RAG reasoning']}
          />
          <FeatureCard
            number={6} title="Accessibility-First"
            color={BRAND.green}
            desc="5 learner profiles: Motor Impairment, Deaf/HoH, Blind/Low Vision, Dyslexia, Autism (Low-Stimulus). Each adjusts feedback modality, timing, and visual complexity."
            highlights={['5 profiles', 'TTS output', 'Haptic feedback', 'Gaze-dwell input']}
          />
        </div>
      </div>

      {/* ── HOW IT WORKS ── */}
      <div style={{ maxWidth: '900px', margin: '0 auto 4rem', padding: '0 2rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ fontSize: '0.7rem', color: BRAND.dim, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
            How it works
          </div>
        </div>

        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.5rem',
        }}>
          {[
            { num: '01', title: 'Select Profile', desc: 'Choose the accessibility profile that matches the learner.' },
            { num: '02', title: 'Make a Gesture', desc: 'Perform the hand sign in front of any device camera.' },
            { num: '03', title: 'Build Sentence', desc: 'Words lock into Subject-Verb-Object position. Grammar validated in real time.' },
            { num: '04', title: 'Speaks Aloud', desc: 'TTS reads the complete sentence. Session report shows learning progress.' },
          ].map(step => (
            <div key={step.num} style={{
              position: 'relative', padding: '1.5rem 1rem', textAlign: 'center',
              background: BRAND.surface, borderRadius: 12,
              border: `1px solid ${BRAND.border}`,
            }}>
              <div style={{
                fontSize: '1.5rem', fontWeight: 900,
                background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                marginBottom: '0.5rem',
              }}>
                {step.num}
              </div>
              <div style={{ fontWeight: 700, fontSize: '0.8rem', color: '#e2e8f0', marginBottom: '0.35rem' }}>
                {step.title}
              </div>
              <div style={{ fontSize: '0.7rem', color: BRAND.muted, lineHeight: 1.5 }}>
                {step.desc}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── PILOT CTA ── */}
      <div style={{
        maxWidth: '700px', margin: '0 auto 3rem', padding: '2rem',
        background: 'linear-gradient(135deg, rgba(139,92,246,0.06), rgba(59,130,246,0.06))',
        border: `1px solid ${BRAND.border}`, borderRadius: 16, textAlign: 'center',
      }}>
        <div style={{
          fontSize: '0.65rem', color: '#8b5cf6', fontWeight: 700,
          letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.5rem',
        }}>
          Pilot Study — Seeking Partners
        </div>
        <p style={{
          fontSize: '0.85rem', color: BRAND.muted, lineHeight: 1.6,
          maxWidth: '500px', margin: '0 auto 1rem',
        }}>
          We are looking for 10–15 students (ages 6–14) with cerebral palsy, autism, or
          hearing impairment for an 8-week study. We provide tablets, training, and
          diagnostic reports. <strong style={{ color: '#e2e8f0' }}>No cost to your institution.</strong>
        </p>
        <button onClick={onStart} style={{
          padding: '0.6rem 2rem', fontSize: '0.85rem', fontWeight: 700,
          color: '#fff', background: 'rgba(139,92,246,0.2)',
          border: '1px solid rgba(139,92,246,0.4)', borderRadius: 10,
          cursor: 'pointer',
        }}>
          Try the Demo
        </button>
      </div>

      {/* ── FOOTER ── */}
      <div style={{
        borderTop: `1px solid ${BRAND.border}`, padding: '2rem',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        flexWrap: 'wrap', gap: '1rem', maxWidth: '900px', margin: '0 auto',
        fontSize: '0.68rem', color: BRAND.dim,
      }}>
        <div>
          Designed & Created by <strong style={{ color: '#e2e8f0' }}>Neil Shankar Ray</strong>
          {' · '}NLP & Speech AI Engineer · Applied Linguist (MA, 14 yrs) · IIT Patna AI/ML
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <span>Patent Pending TEMP/E-1/22951/2026-KOL</span>
          <span style={{ color: BRAND.muted }}>roychinu45@gmail.com</span>
        </div>
      </div>
    </div>
  );
}
