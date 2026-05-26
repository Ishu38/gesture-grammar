/**
 * ContactUs.jsx — Contact information + Google Maps embed
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
  section: { maxWidth: '720px', margin: '0 auto 3rem' },
  heading: { fontSize: '1.6rem', fontWeight: 800, color: '#f1f5f9', margin: '0 0 0.4rem' },
  subheading: {
    fontSize: '0.75rem', color: '#4ade80', fontWeight: 700, letterSpacing: '0.1em',
    textTransform: 'uppercase', marginBottom: '1.5rem',
  },
  card: {
    background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
    borderRadius: 14, padding: '1.5rem', marginBottom: '1rem',
  },
  body: { fontSize: '0.88rem', color: '#94a3b8', lineHeight: 1.75 },
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

export default function ContactUs({ onNavigate }) {
  return (
    <div style={STYLE.page}>
      {/* Nav */}
      <div style={STYLE.nav}>
        <button onClick={() => onNavigate('HOME')} style={STYLE.navBrand}>MLAF</button>
        <div style={STYLE.navLinks}>
          <NavButton onClick={() => onNavigate('HOME')}>Home</NavButton>
          <NavButton onClick={() => onNavigate('ABOUT')}>About Us</NavButton>
          <NavButton active onClick={() => onNavigate('CONTACT')}>Contact Us</NavButton>
        </div>
      </div>

      <div style={{ ...STYLE.section, paddingTop: '2rem' }}>
        <div style={STYLE.subheading}>Get in Touch</div>
        <h2 style={STYLE.heading}>Contact Us</h2>

        <div style={STYLE.card}>
          <div style={{
            display: 'grid', gridTemplateColumns: '100px 1fr', gap: '0.75rem 1.5rem',
            alignItems: 'start',
          }}>
            <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase', paddingTop: 2 }}>
              Name
            </div>
            <div style={{ fontSize: '1rem', color: '#e2e8f0', fontWeight: 700 }}>
              Neil Shankar Ray
            </div>

            <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase', paddingTop: 2 }}>
              C/O
            </div>
            <div style={{ fontSize: '0.9rem', color: '#cbd5e1' }}>
              Mrs Chinu Ray
            </div>

            <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase', paddingTop: 2 }}>
              Address
            </div>
            <div>
              <div style={{ fontSize: '0.9rem', color: '#cbd5e1', lineHeight: 1.6 }}>
                55/1, Jubilee Park<br />
                Tollygunge<br />
                Kolkata — 700033<br />
                West Bengal, India
              </div>
            </div>

            <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase', paddingTop: 2 }}>
              Email
            </div>
            <div>
              <a href="mailto:neilshankarray@vaaani.in" style={{
                fontSize: '0.9rem', color: '#60a5fa', textDecoration: 'none',
              }}>
                neilshankarray@vaaani.in
              </a>
            </div>

            <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase', paddingTop: 2 }}>
              Website
            </div>
            <div>
              <a href="https://vaaani.in" target="_blank" rel="noopener noreferrer" style={{
                fontSize: '0.9rem', color: '#60a5fa', textDecoration: 'none',
              }}>
                https://vaaani.in
              </a>
            </div>

            <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase', paddingTop: 2 }}>
              LinkedIn
            </div>
            <div>
              <a href="https://linkedin.com/in/neilsray" target="_blank" rel="noopener noreferrer" style={{
                fontSize: '0.9rem', color: '#60a5fa', textDecoration: 'none',
              }}>
                linkedin.com/in/neilsray
              </a>
            </div>
          </div>
        </div>
      </div>

      <div style={STYLE.section}>
        <div style={STYLE.subheading}>Find Us</div>
        <div style={{
          ...STYLE.card, padding: 0, overflow: 'hidden',
          borderRadius: 14, border: '1px solid rgba(255,255,255,0.08)',
        }}>
          <iframe
            title="MLAF Office Location"
            width="100%"
            height="400"
            style={{ border: 0, display: 'block' }}
            loading="lazy"
            allowFullScreen
            referrerPolicy="no-referrer-when-downgrade"
            src="https://www.google.com/maps/embed/v1/place?key=AIzaSyBFw0Qbyq9zTFTd-tUY6dZWTgaQzuU17R8&q=55%2F1+Jubilee+Park+Tollygunge+Kolkata+700033"
          />
        </div>
        <p style={{
          fontSize: '0.68rem', color: '#475569', marginTop: '0.5rem', textAlign: 'center',
        }}>
          Jubilee Park, Tollygunge, Kolkata — 700033
        </p>
      </div>
    </div>
  );
}
