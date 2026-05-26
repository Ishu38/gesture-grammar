/**
 * DemoHandout.jsx — Professional 1-page printable demo handout
 * Print-optimized. Give to educators, NGO directors, and school administrators
 * during MLAF demonstrations across Kolkata.
 */
import { useCallback } from 'react';

const BRAND = { primary: '#0f172a', accent: '#4ade80', muted: '#475569' };

function SectionTitle({ children, color = '#4ade80' }) {
  return (
    <h3 style={{
      fontSize: '0.95rem', fontWeight: 800, color, margin: '0 0 0.75rem',
      letterSpacing: '0.05em', textTransform: 'uppercase',
      borderBottom: `2px solid ${color}20`, paddingBottom: '0.4rem',
    }}>
      {children}
    </h3>
  );
}

function FeatureRow({ icon, title, desc, color = '#4ade80' }) {
  return (
    <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start', marginBottom: '0.6rem' }}>
      <span style={{
        flexShrink: 0, width: 32, height: 32, borderRadius: 8,
        background: `${color}15`, color, display: 'flex',
        alignItems: 'center', justifyContent: 'center',
        fontSize: '0.9rem', fontWeight: 700,
      }}>
        {icon}
      </span>
      <div>
        <div style={{ fontWeight: 700, fontSize: '0.78rem', color: '#1e293b', marginBottom: 1 }}>{title}</div>
        <div style={{ fontSize: '0.7rem', color: '#64748b', lineHeight: 1.4 }}>{desc}</div>
      </div>
    </div>
  );
}

export default function DemoHandout({ onBack }) {
  const handlePrint = useCallback(() => {
    window.print();
  }, []);

  return (
    <>
      {/* Print-only styles */}
      <style>{`
        @media print {
          body { background: #fff !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
          .no-print, header, footer, [role="contentinfo"], .app-footer, .app-header { display: none !important; }
          .handout-page { margin: 0; padding: 0; box-shadow: none; border: none; max-width: 100%; }
          @page { size: A4; margin: 12mm; }
          * { color-adjust: exact; -webkit-print-color-adjust: exact; }
        }
      `}</style>

      {/* Print button (hidden on print) */}
      <div className="no-print" style={{
        display: 'flex', justifyContent: 'center', gap: '1rem',
        padding: '1rem', background: '#0f172a',
      }}>
        <button onClick={handlePrint} style={{
          padding: '0.8rem 2.5rem', fontSize: '1rem', fontWeight: 700,
          color: '#0f172a', background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
          border: 'none', borderRadius: '10px', cursor: 'pointer',
          boxShadow: '0 4px 20px rgba(74,222,128,0.3)',
        }}>
          Print Handout
        </button>
        {onBack && (
          <button onClick={onBack} style={{
            padding: '0.8rem 2rem', fontSize: '0.95rem', fontWeight: 600,
            color: '#94a3b8', background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)', borderRadius: '10px',
            cursor: 'pointer',
          }}>
            Back
          </button>
        )}
      </div>

      {/* Handout Content */}
      <div className="handout-page" style={{
        maxWidth: '740px', margin: '0 auto', padding: '2rem',
        background: '#ffffff', color: '#1e293b',
        fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
        borderRadius: '8px', boxShadow: '0 4px 30px rgba(0,0,0,0.15)',
      }}>
        {/* ─── HEADER ─── */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          marginBottom: '1.5rem', paddingBottom: '1rem',
          borderBottom: '3px solid #4ade80',
        }}>
          <div>
            <h1 style={{ fontSize: '2rem', fontWeight: 900, color: '#0f172a', margin: 0, letterSpacing: '0.08em' }}>
              MLAF
            </h1>
            <p style={{ fontSize: '0.75rem', color: '#475569', margin: '0.25rem 0 0', letterSpacing: '0.04em' }}>
              Multimodal Language Acquisition Framework
            </p>
            <p style={{ fontSize: '0.62rem', color: '#94a3b8', margin: '0.15rem 0 0' }}>
              Patent Pending — TEMP/E-1/22951/2026-KOL · Indian Patent Office (2026)
            </p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{
              padding: '4px 10px', background: '#4ade8015', borderRadius: 6,
              fontSize: '0.62rem', color: '#16a34a', fontWeight: 700, letterSpacing: '0.05em',
            }}>
              PATENT PENDING
            </div>
          </div>
        </div>

        {/* ─── SECTION 1: WHAT IT IS ─── */}
        <div style={{ marginBottom: '1.25rem' }}>
          <SectionTitle>What It Is</SectionTitle>
          <p style={{
            fontSize: '0.82rem', color: '#475569', lineHeight: 1.6, marginBottom: '1rem',
          }}>
            MLAF is a <strong>browser-based application</strong> that teaches English grammar through hand gestures 
            detected by a standard phone or laptop camera. Designed for children with <strong>cerebral palsy, 
            non-verbal autism, motor impairment, and deaf/hard-of-hearing</strong> — learners who cannot access 
            traditional tools that assume writing, typing, or speaking ability.
          </p>

          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.3rem 1rem',
          }}>
            <FeatureRow icon="A" title="Adaptive AI" color="#4ade80"
              desc="Auto-adjusts to motor tremor & cognitive load. No two learners treated the same."
            />
            <FeatureRow icon="I" title="ISL Transfer Detection" color="#8b5cf6"
              desc="Detects Indian Sign Language word-order patterns & guides toward English SVO."
            />
            <FeatureRow icon="D" title="Diagnostic Reports" color="#3b82f6"
              desc="Per-concept knowledge estimates. Shows exactly what each learner knows."
            />
            <FeatureRow icon="O" title="Works Offline" color="#f59e0b"
              desc="PWA — install on any phone. No internet, no specialized hardware needed."
            />
            <FeatureRow icon="T" title="Text-to-Speech" color="#ec4899"
              desc="Speaks completed sentences aloud. Becomes the voice for non-verbal learners."
            />
            <FeatureRow icon="N" title="Neuro-Symbolic AI" color="#14b8a6"
              desc="Bayesian fusion of vision + acoustics + gaze. Formal grammar engine validates sentences."
            />
          </div>
        </div>

        {/* ─── SECTION 2: HOW IT WORKS ─── */}
        <div style={{ marginBottom: '1.25rem' }}>
          <SectionTitle color="#3b82f6">How It Works</SectionTitle>
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.5rem',
            textAlign: 'center',
          }}>
            {[
              { num: 1, title: 'Select Profile', desc: 'Motor / Deaf / Blind / Dyslexia' },
              { num: 2, title: 'Make Gesture', desc: 'Hand gesture in front of camera' },
              { num: 3, title: 'Word Appears', desc: 'AI classifies the gesture shape' },
              { num: 4, title: 'Build Sentence', desc: 'Subject-Verb-Object order' },
              { num: 5, title: 'Speaks Aloud', desc: 'TTS becomes their voice' },
            ].map(step => (
              <div key={step.num} style={{
                background: '#f8fafc', borderRadius: 8, padding: '0.6rem 0.4rem',
                border: '1px solid #e2e8f0',
              }}>
                <div style={{
                  width: 28, height: 28, borderRadius: '50%', margin: '0 auto 0.4rem',
                  background: '#3b82f6', color: '#fff', display: 'flex',
                  alignItems: 'center', justifyContent: 'center',
                  fontSize: '0.8rem', fontWeight: 800,
                }}>
                  {step.num}
                </div>
                <div style={{ fontWeight: 700, fontSize: '0.68rem', color: '#1e293b', marginBottom: 2 }}>
                  {step.title}
                </div>
                <div style={{ fontSize: '0.6rem', color: '#94a3b8', lineHeight: 1.3 }}>
                  {step.desc}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ─── SECTION 3: PILOT INVITATION ─── */}
        <div style={{ marginBottom: '1.25rem' }}>
          <SectionTitle color="#f59e0b">Pilot Study Invitation</SectionTitle>
          <div style={{
            display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem',
            alignItems: 'start',
          }}>
            <div>
              <p style={{
                fontSize: '0.78rem', color: '#475569', lineHeight: 1.6, margin: '0 0 0.75rem',
              }}>
                We are seeking <strong>10–15 students</strong> (ages 6–14) with cerebral palsy, non-verbal autism, 
                or hearing impairment for an <strong>8-week pilot study</strong> (3 sessions/week, 20 min each). 
                We provide tablets, training, and comprehensive diagnostic reports. <strong>No cost to your institution.</strong>
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.4rem', fontSize: '0.7rem' }}>
                <div style={{ background: '#f8fafc', padding: '0.5rem 0.75rem', borderRadius: 6 }}>
                  <strong style={{ color: '#1e293b' }}>Pre/Post Testing</strong><br />
                  <span style={{ color: '#64748b' }}>20-item sentence construction test</span>
                </div>
                <div style={{ background: '#f8fafc', padding: '0.5rem 0.75rem', borderRadius: 6 }}>
                  <strong style={{ color: '#1e293b' }}>Ethics Approved</strong><br />
                  <span style={{ color: '#64748b' }}>Parental consent + institutional review</span>
                </div>
                <div style={{ background: '#f8fafc', padding: '0.5rem 0.75rem', borderRadius: 6 }}>
                  <strong style={{ color: '#1e293b' }}>Data Privacy</strong><br />
                  <span style={{ color: '#64748b' }}>All data on-device. Zero cloud transmission.</span>
                </div>
                <div style={{ background: '#f8fafc', padding: '0.5rem 0.75rem', borderRadius: 6 }}>
                  <strong style={{ color: '#1e293b' }}>Per-Child Reports</strong><br />
                  <span style={{ color: '#64748b' }}>Knowledge state, ISL patterns, progress curves</span>
                </div>
              </div>
            </div>

            {/* QR Code + Contact */}
            <div style={{ textAlign: 'center' }}>
              <div style={{
                background: '#f8fafc', borderRadius: 12, padding: '1rem',
                border: '1px solid #e2e8f0',
              }}>
                <div style={{ fontSize: '0.65rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
                  Try it now:
                </div>
                <img
                  src="https://quickchart.io/qr?text=https%3A%2F%2Fmulti-modal-gesture-grammar.vercel.app&size=160&margin=1"
                  alt="QR code to MLAF demo"
                  style={{ width: 120, height: 120, marginBottom: '0.4rem' }}
                />
                <div style={{
                  fontSize: '0.58rem', color: '#3b82f6', wordBreak: 'break-all',
                  fontWeight: 600,
                }}>
                  multi-modal-gesture-grammar.vercel.app
                </div>
              </div>

              <div style={{
                marginTop: '0.75rem', padding: '0.75rem',
                background: 'linear-gradient(135deg, #0f172a, #1e293b)',
                borderRadius: 8, color: '#e2e8f0',
              }}>
                <div style={{ fontWeight: 800, fontSize: '0.75rem', marginBottom: '0.5rem' }}>
                  Neil Shankar Ray
                </div>
                <div style={{ fontSize: '0.6rem', color: '#94a3b8', lineHeight: 1.6 }}>
                  NLP & Speech AI Engineer · Applied Linguist (MA, 14 yrs) · IIT Patna AI/ML<br />
                  roychinu45@gmail.com<br />
                  linkedin.com/in/neilsray
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ─── FOOTER ─── */}
        <div style={{
          borderTop: '1px solid #e2e8f0', paddingTop: '0.75rem',
          fontSize: '0.58rem', color: '#94a3b8', textAlign: 'center',
          lineHeight: 1.5,
        }}>
          MLAF — Multimodal Language Acquisition Framework · Patent-Pending TEMP/E-1/22951/2026-KOL, Indian Patent Office (2026)
          <br />All rights reserved. Commercial licensing inquiries: roychinu45@gmail.com
        </div>
      </div>
    </>
  );
}
