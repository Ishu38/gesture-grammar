/**
 * WelcomeScreen.jsx — Professional landing page for MLAF
 * Designed for educators, funders, and institutional decision-makers.
 * Responsive across mobile, tablet, and desktop.
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

function StatPill({ value, label }) {
  return (
    <div style={{
      textAlign: 'center', padding: '0.4rem 0.8rem', minWidth: 70,
    }}>
      <div style={{ fontSize: '1.3rem', fontWeight: 800, color: '#fff', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </div>
      <div style={{ fontSize: '0.55rem', color: BRAND.dim, marginTop: 2, letterSpacing: '0.06em', textTransform: 'uppercase', lineHeight: 1.2 }}>
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
      borderRadius: 16, padding: '1.25rem',
    }}>
      <div style={{
        width: 36, height: 36, borderRadius: 10,
        background: `${color}12`, color,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: '0.85rem', fontWeight: 800, marginBottom: '0.75rem',
        border: `1px solid ${color}20`,
      }}>
        {String(number).padStart(2, '0')}
      </div>
      <h3 style={{ fontSize: '0.9rem', fontWeight: 700, color: '#f1f5f9', margin: '0 0 0.4rem' }}>
        {title}
      </h3>
      <p style={{ fontSize: '0.76rem', color: BRAND.muted, lineHeight: 1.6, margin: '0 0 0.6rem' }}>
        {desc}
      </p>
      {highlights && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
          {highlights.map((h, i) => (
            <span key={i} style={{
              fontSize: '0.6rem', padding: '2px 7px', borderRadius: 4,
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

function NavButton({ children, onClick }) {
  return (
    <button onClick={onClick} style={{
      fontSize: '0.65rem', color: BRAND.muted, fontWeight: 600,
      letterSpacing: '0.04em', background: 'none', border: 'none',
      cursor: 'pointer', padding: '5px 8px', borderRadius: 6,
      whiteSpace: 'nowrap',
    }}>
      {children}
    </button>
  );
}

function StepCard({ num, title, desc }) {
  return (
    <div style={{
      position: 'relative', padding: '1rem 0.5rem', textAlign: 'center',
      background: BRAND.surface, borderRadius: 12,
      border: `1px solid ${BRAND.border}`,
    }}>
      <div style={{
        fontSize: '1.3rem', fontWeight: 900,
        background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
        WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
        marginBottom: '0.4rem', lineHeight: 1,
      }}>
        {num}
      </div>
      <div style={{ fontWeight: 700, fontSize: '0.72rem', color: '#e2e8f0', marginBottom: '0.25rem' }}>
        {title}
      </div>
      <div style={{ fontSize: '0.65rem', color: BRAND.muted, lineHeight: 1.4 }}>
        {desc}
      </div>
    </div>
  );
}

export default function WelcomeScreen({ onStart, onNavigate }) {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#05070d',
      fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
      color: '#e2e8f0',
      overflowX: 'hidden',
    }}>
      {/* ── TOP NAV ── */}
      <div style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '10px 16px',
        background: '#05070d', borderBottom: '1px solid rgba(255,255,255,0.04)',
      }}>
        <span style={{ fontSize: '0.8rem', fontWeight: 800, color: '#4ade80', letterSpacing: '0.15em' }}>MLAF</span>
        <div style={{ display: 'flex', gap: '0.25rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <NavButton onClick={() => onNavigate && onNavigate('HOME')}>Home</NavButton>
          <NavButton onClick={() => onNavigate && onNavigate('ABOUT')}>About</NavButton>
          <NavButton onClick={() => onNavigate && onNavigate('CONTACT')}>Contact</NavButton>
          <NavButton onClick={() => onNavigate && onNavigate('HANDOUT')}>Handout</NavButton>
          <NavButton onClick={() => onNavigate && onNavigate('BENGALI')}>বাংলা</NavButton>
        </div>
      </div>

      {/* ── HERO ── */}
      <div style={{
        padding: '100px 1.5rem 40px', maxWidth: '760px', margin: '0 auto',
        textAlign: 'center',
      }}>
        <h1 style={{
          fontSize: 'clamp(1.6rem, 5vw, 3.2rem)', fontWeight: 900,
          color: '#ffffff', lineHeight: 1.15, margin: 0,
          letterSpacing: '-0.01em',
        }}>
          English grammar for children<br />
          <span style={{ color: '#4ade80' }}>who cannot write, type, or speak.</span>
        </h1>

        <p style={{
          fontSize: 'clamp(0.85rem, 1.5vw, 1rem)', color: BRAND.muted, marginTop: '1rem',
          lineHeight: 1.6, maxWidth: '520px', marginLeft: 'auto', marginRight: 'auto',
        }}>
          MLAF teaches sentence structure through hand gestures detected by
          any phone camera. Adaptive AI calibrates to motor impairment,
          detects Indian Sign Language transfer, and speaks completed
          sentences aloud.
        </p>

        <div style={{ marginTop: '1.5rem' }}>
          <button onClick={onStart} style={{
            padding: '0.8rem 2.2rem', fontSize: '0.9rem', fontWeight: 700,
            color: '#05070d', background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
            border: 'none', borderRadius: 12, cursor: 'pointer', letterSpacing: '0.03em',
            boxShadow: '0 4px 20px rgba(74,222,128,0.2)',
          }}>
            Try the Demo →
          </button>
        </div>
      </div>

      {/* ── STATS ── */}
      <div style={{
        display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '0.25rem',
        width: 'fit-content', margin: '0 auto', padding: '1rem 1.5rem',
        background: BRAND.surface, borderRadius: 14,
        border: `1px solid ${BRAND.border}`,
        marginBottom: '3rem',
      }}>
        <StatPill value="10" label="Patent claims" />
        <StatPill value="99.5%" label="Accuracy" />
        <StatPill value="19" label="Gestures" />
        <StatPill value="Offline" label="Zero cloud" />
        <StatPill value="PWA" label="No install" />
      </div>

      {/* ── FEATURES GRID ── */}
      <div style={{ maxWidth: '880px', margin: '0 auto', padding: '0 1.5rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ fontSize: '0.65rem', color: BRAND.dim, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: '0.4rem' }}>
            What makes MLAF different
          </div>
          <h2 style={{ fontSize: 'clamp(1.1rem, 2.5vw, 1.5rem)', fontWeight: 800, color: '#f1f5f9', margin: 0 }}>
            Adaptive. Explainable. Built for real classrooms.
          </h2>
        </div>

        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
          gap: '0.75rem', marginBottom: '2.5rem',
        }}>
          <FeatureCard number={1} title="Motor-Adaptive Recognition" color={BRAND.green}
            desc="Camera detects hand tremor via positional jitter analysis. Tolerance bands auto-widen — the system gets MORE forgiving."
            highlights={['Real-time jitter', 'Fatigue detection']}
          />
          <FeatureCard number={2} title="ISL Transfer Correction" color={BRAND.purple}
            desc="Detects when deaf learners default to ISL SOV word order and guides toward English SVO."
            highlights={['SOV→SVO reorder', 'Pro-drop detection']}
          />
          <FeatureCard number={3} title="Bayesian Knowledge Tracing" color={BRAND.blue}
            desc="Estimates P(known) for every grammar concept. Predicts retention decay. Data-driven pedagogy."
            highlights={['Per-concept P(know)', 'Decay forecasting']}
          />
          <FeatureCard number={4} title="Explainable AI" color={BRAND.amber}
            desc="Every error generates a human-readable explanation. Educators see WHY the system made decisions."
            highlights={['Root cause analysis', 'Remediation']}
          />
          <FeatureCard number={5} title="Neuro-Symbolic Architecture" color={BRAND.rose}
            desc="MediaPipe neural perception + ONNX RF + Chomskyan X-bar grammar + Bayesian trimodal fusion."
            highlights={['Bayesian fusion', 'Graph RAG']}
          />
          <FeatureCard number={6} title="Accessibility-First" color={BRAND.green}
            desc="5 learner profiles. Each adjusts feedback modality, timing, and visual complexity."
            highlights={['Motor · Deaf · Blind', 'Dyslexia · Autism']}
          />
        </div>
      </div>

      {/* ── HOW IT WORKS ── */}
      <div style={{ maxWidth: '880px', margin: '0 auto 3rem', padding: '0 1.5rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <div style={{ fontSize: '0.65rem', color: BRAND.dim, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            How it works
          </div>
        </div>
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
          gap: '0.5rem',
        }}>
          <StepCard num="01" title="Select Profile" desc="Choose the accessibility profile." />
          <StepCard num="02" title="Make Gesture" desc="Hand sign in front of camera." />
          <StepCard num="03" title="Build Sentence" desc="Grammar validated in real time." />
          <StepCard num="04" title="Speaks Aloud" desc="TTS becomes their voice." />
        </div>
      </div>

      {/* ── PILOT CTA ── */}
      <div style={{
        maxWidth: '660px', margin: '0 auto 2.5rem', padding: '1.5rem',
        background: 'rgba(139,92,246,0.04)',
        border: `1px solid ${BRAND.border}`, borderRadius: 16, textAlign: 'center',
        marginLeft: '1.5rem', marginRight: '1.5rem',
      }}>
        <div style={{
          fontSize: '0.6rem', color: '#8b5cf6', fontWeight: 700,
          letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '0.4rem',
        }}>
          Pilot Study — Seeking Partners
        </div>
        <p style={{
          fontSize: '0.8rem', color: BRAND.muted, lineHeight: 1.6,
          maxWidth: '480px', margin: '0 auto 0.8rem',
        }}>
          10–15 students (ages 6–14) with cerebral palsy, autism, or hearing impairment.
          8-week study. We provide tablets, training, and reports.
          <strong style={{ color: '#e2e8f0' }}> No cost to your institution.</strong>
        </p>
        <button onClick={onStart} style={{
          padding: '0.5rem 1.5rem', fontSize: '0.8rem', fontWeight: 700,
          color: '#fff', background: 'rgba(139,92,246,0.2)',
          border: '1px solid rgba(139,92,246,0.4)', borderRadius: 10,
          cursor: 'pointer',
        }}>
          Try the Demo
        </button>
      </div>

      {/* ── FOOTER ── */}
      <div style={{
        borderTop: `1px solid ${BRAND.border}`, padding: '1.25rem 1.5rem',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        flexWrap: 'wrap', gap: '0.5rem', maxWidth: '880px', margin: '0 auto',
        fontSize: '0.62rem', color: BRAND.dim,
      }}>
        <div>
          <strong style={{ color: '#e2e8f0' }}>Neil Shankar Ray</strong>
          {' · '}IIT Patna · Applied Linguist (MA, 14 yrs)
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <span>Patent TEMP/E-1/22951/2026-KOL</span>
          <span style={{ color: BRAND.muted }}>neilshankarray@vaaani.in</span>
        </div>
      </div>
    </div>
  );
}
