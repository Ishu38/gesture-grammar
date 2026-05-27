/**
 * ContactUs.jsx — graffiti-vibe contact page, matching WelcomeScreen.
 * Preserves the real contact facts; only the visual layer changed.
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

function FieldRow({ label, children, accent }) {
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: 'minmax(110px, 130px) 1fr',
      gap: '0.6rem 1.2rem', alignItems: 'start', padding: '0.6rem 0',
      borderBottom: `1px dashed ${C.ink}22`,
    }}>
      <div style={{
        fontFamily: BODY, fontSize: '0.72rem', fontWeight: 700,
        letterSpacing: '0.1em', textTransform: 'uppercase',
        color: accent || C.inkSoft, paddingTop: 4,
      }}>
        {label}
      </div>
      <div style={{ fontFamily: BODY, fontSize: '0.95rem', color: C.ink, lineHeight: 1.55 }}>
        {children}
      </div>
    </div>
  );
}

export default function ContactUs({ onNavigate }) {
  return (
    <div style={{
      minHeight: '100vh', background: C.cream, color: C.ink,
      fontFamily: BODY, overflowX: 'hidden', position: 'relative',
    }}>
      {/* Background splats */}
      <Splat color={C.violet} style={{ position: 'absolute', top: 120, right: -40, width: 170, height: 170, opacity: 0.16, transform: 'rotate(15deg)' }} />
      <Splat color={C.teal}   style={{ position: 'absolute', top: 600, left: -50, width: 150, height: 150, opacity: 0.14, transform: 'rotate(-20deg)' }} />
      <Splat color={C.sun}    style={{ position: 'absolute', top: 1100, right: -20, width: 130, height: 130, opacity: 0.22, transform: 'rotate(35deg)' }} />

      {/* Nav */}
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
          <NavLink onClick={() => onNavigate('ABOUT')}>About</NavLink>
          <NavLink active onClick={() => onNavigate('CONTACT')}>Contact</NavLink>
        </div>
      </nav>

      {/* Hero */}
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
          Get in touch
        </div>
        <h1 style={{
          fontFamily: DISPLAY, fontSize: 'clamp(2.2rem, 6vw, 3.6rem)',
          color: C.ink, lineHeight: 1.05, margin: '0 0 1rem', letterSpacing: '-0.005em',
        }}>
          Drop us a <span style={{ color: C.coral }}>line.</span>
        </h1>
        <p style={{
          fontFamily: BODY, fontSize: '1rem', color: C.inkSoft, lineHeight: 1.55,
          maxWidth: 540, margin: 0,
        }}>
          Pilots, partnerships, licensing, press, or just a hello — every email
          is read by Neil. Replies usually within 24-48 hours.
        </p>
      </section>

      {/* Contact card */}
      <section style={{
        maxWidth: 840, margin: '32px auto 0', padding: '0 22px',
        position: 'relative', zIndex: 1,
      }}>
        <div style={{
          background: C.paper, border: `2px solid ${C.ink}`, borderRadius: 18,
          padding: '1.6rem 1.4rem', boxShadow: `6px 6px 0 0 ${C.coral}`,
        }}>
          <FieldRow label="Name"   accent={C.coral}>
            <strong style={{ fontWeight: 700 }}>Neil Shankar Ray</strong>
          </FieldRow>
          <FieldRow label="C/O"    accent={C.violet}>Mrs Chinu Ray</FieldRow>
          <FieldRow label="Address" accent={C.teal}>
            55/1, Jubilee Park<br/>
            Tollygunge<br/>
            Kolkata — 700033<br/>
            West Bengal, India
          </FieldRow>
          <FieldRow label="Email" accent={C.coral}>
            <a href="mailto:neilshankarray@vaaani.in" style={{
              color: C.ink, textDecoration: 'underline', textDecorationThickness: 2,
              textDecorationColor: C.sun, textUnderlineOffset: 3,
            }}>neilshankarray@vaaani.in</a>
          </FieldRow>
          <FieldRow label="Website" accent={C.violet}>
            <a href="https://vaaani.in" target="_blank" rel="noopener noreferrer" style={{
              color: C.ink, textDecoration: 'underline', textDecorationThickness: 2,
              textDecorationColor: C.sun, textUnderlineOffset: 3,
            }}>vaaani.in</a>
          </FieldRow>
          <FieldRow label="LinkedIn" accent={C.teal}>
            <a href="https://linkedin.com/in/neilsray" target="_blank" rel="noopener noreferrer" style={{
              color: C.ink, textDecoration: 'underline', textDecorationThickness: 2,
              textDecorationColor: C.sun, textUnderlineOffset: 3,
            }}>linkedin.com/in/neilsray</a>
          </FieldRow>
        </div>
      </section>

      {/* Map */}
      <section style={{
        maxWidth: 840, margin: '40px auto 0', padding: '0 22px',
        position: 'relative', zIndex: 1,
      }}>
        <div style={{
          fontFamily: BODY, fontSize: '0.72rem', fontWeight: 700,
          letterSpacing: '0.18em', textTransform: 'uppercase', color: C.coral, marginBottom: 6,
        }}>
          Find us
        </div>
        <h2 style={{ fontFamily: DISPLAY, fontSize: 'clamp(1.4rem, 3vw, 2rem)', color: C.ink, margin: '0 0 1rem', letterSpacing: '-0.005em' }}>
          On the map
        </h2>
        <div style={{
          border: `2px solid ${C.ink}`, borderRadius: 18, overflow: 'hidden',
          boxShadow: `6px 6px 0 0 ${C.violet}`,
        }}>
          <iframe
            title="MLAF Office Location"
            width="100%" height="400"
            style={{ border: 0, display: 'block' }}
            loading="lazy" allowFullScreen
            referrerPolicy="no-referrer-when-downgrade"
            src="https://www.google.com/maps/embed/v1/place?key=AIzaSyBFw0Qbyq9zTFTd-tUY6dZWTgaQzuU17R8&q=55%2F1+Jubilee+Park+Tollygunge+Kolkata+700033"
          />
        </div>
        <p style={{
          fontFamily: BODY, fontSize: '0.78rem', color: C.inkSoft,
          marginTop: '0.6rem', textAlign: 'center',
        }}>
          Jubilee Park, Tollygunge, Kolkata — 700033
        </p>
      </section>

      {/* Footer */}
      <footer style={{
        maxWidth: 840, margin: '60px auto 0', padding: '24px 22px',
        borderTop: `2px solid ${C.ink}`, display: 'flex', flexWrap: 'wrap',
        gap: '0.8rem 1.4rem', justifyContent: 'space-between', alignItems: 'center',
        fontFamily: BODY, fontSize: '0.78rem', color: C.inkSoft,
      }}>
        <div>
          <strong style={{ color: C.ink }}>MLAF</strong> · BSL 1.1 · Indian patent app. 202631020540
        </div>
        <button onClick={() => onNavigate('HOME')} style={{
          fontFamily: BODY, fontSize: '0.8rem', fontWeight: 700, color: C.ink,
          background: 'transparent', border: `2px solid ${C.ink}`, borderRadius: 8,
          padding: '6px 14px', cursor: 'pointer',
        }}>← Back home</button>
      </footer>
    </div>
  );
}
