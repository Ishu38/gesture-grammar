/**
 * DemoHandoutBengali.jsx — Bengali-language printable demo handout
 * জন্য শিক্ষক, এনজিও ডিরেক্টর, এবং স্কুল প্রশাসকদের জন্য।
 * Print-optimized. Give alongside the English version during demos in Kolkata.
 */
import { useCallback } from 'react';

const B = {
  primary: '#0F172A', accent: '#FF5252', muted: '#475569',
};

function SectionTitle({ children, color = '#FF5252' }) {
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

function FeatureRow({ icon, title, desc, color = '#FF5252' }) {
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
        <div style={{ fontSize: '0.7rem', color: '#64748b', lineHeight: 1.5 }}>{desc}</div>
      </div>
    </div>
  );
}

export default function DemoHandoutBengali({ onBack, onNavigate }) {
  const handlePrint = useCallback(() => { window.print(); }, []);

  return (
    <>
      <style>{`
        @media print {
          body { background: #fff !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
          .no-print, header, footer, [role="contentinfo"], .app-footer, .app-header { display: none !important; }
          .handout-page { margin: 0; padding: 0; box-shadow: none; border: none; max-width: 100%; }
          @page { size: A4; margin: 12mm; }
        }
      `}</style>

      <div className="no-print" style={{
        display: 'flex', justifyContent: 'center', gap: '1rem',
        padding: '1rem', background: '#FFF6E6',
      }}>
        <button onClick={handlePrint} style={{
          padding: '0.8rem 2.5rem', fontSize: '1rem', fontWeight: 700,
          color: '#FFF6E6', background: '#FF5252',
          border: 'none', borderRadius: '10px', cursor: 'pointer',
          boxShadow: '0 4px 20px rgba(74,222,128,0.3)',
        }}>
          প্রিন্ট
        </button>
        {onNavigate && (
          <button onClick={() => onNavigate('HANDOUT')} style={{
            padding: '0.8rem 2rem', fontSize: '0.95rem', fontWeight: 600,
            color: '#3b82f6', background: 'rgba(59,130,246,0.08)',
            border: '1px solid rgba(59,130,246,0.3)', borderRadius: '10px', cursor: 'pointer',
          }}>
            English Version
          </button>
        )}
        {onBack && (
          <button onClick={onBack} style={{
            padding: '0.8rem 2rem', fontSize: '0.95rem', fontWeight: 600,
            color: '#94a3b8', background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)', borderRadius: '10px', cursor: 'pointer',
          }}>
            ফিরে যান
          </button>
        )}
      </div>

      <div className="handout-page" style={{
        maxWidth: '740px', margin: '0 auto', padding: '2rem',
        background: '#ffffff', color: '#1e293b',
        fontFamily: "'Noto Sans Bengali', 'Inter', system-ui, -apple-system, sans-serif",
        borderRadius: '8px', boxShadow: '0 4px 30px rgba(0,0,0,0.15)',
      }}>
        {/* HEADER */}
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
          marginBottom: '1.5rem', paddingBottom: '1rem',
          borderBottom: '3px solid #FF5252',
        }}>
          <div>
            <h1 style={{ fontSize: '2rem', fontWeight: 900, color: '#0f172a', margin: 0, letterSpacing: '0.08em' }}>
              MLAF
            </h1>
            <p style={{ fontSize: '0.75rem', color: '#475569', margin: '0.25rem 0 0', letterSpacing: '0.04em' }}>
              মাল্টিমোডাল ভাষা অর্জন ফ্রেমওয়ার্ক
            </p>
            <p style={{ fontSize: '0.62rem', color: '#94a3b8', margin: '0.15rem 0 0' }}>
              পেটেন্ট পেন্ডিং — 202631020540 · ভারতীয় পেটেন্ট অফিস (২০২৬)
            </p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{
              padding: '4px 10px', background: '#FF525215', borderRadius: 6,
              fontSize: '0.62rem', color: '#16a34a', fontWeight: 700, letterSpacing: '0.05em',
            }}>
              পেটেন্ট পেন্ডিং
            </div>
          </div>
        </div>

        {/* SECTION 1: WHAT IT IS */}
        <div style={{ marginBottom: '1.25rem' }}>
          <SectionTitle>এটি কী?</SectionTitle>
          <p style={{
            fontSize: '0.82rem', color: '#475569', lineHeight: 1.8, marginBottom: '1rem',
          }}>
            <strong>MLAF</strong> একটি ব্রাউজার-ভিত্তিক অ্যাপ্লিকেশন যা হাতের ইশারার মাধ্যমে ইংরেজি ব্যাকরণ শেখায়।
            এটি তৈরি করা হয়েছে <strong>সেরিব্রাল পলসি, নন-ভার্বাল অটিজম, মোটর অক্ষমতা, এবং বধির/শ্রবণ-প্রতিবন্ধী</strong>
            শিক্ষার্থীদের জন্য — যারা লিখতে, টাইপ করতে, বা কথা বলতে পারেন না।
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.3rem 1rem' }}>
            <FeatureRow icon="A" title="অভিযোজিত AI" color="#FF5252"
              desc="হাতের কাঁপুনি ও মানসিক চাপ স্বয়ংক্রিয়ভাবে সনাক্ত করে। প্রতিটি শিক্ষার্থী আলাদা ভাবে সমর্থিত।"
            />
            <FeatureRow icon="I" title="ISL ট্রান্সফার সনাক্তকরণ" color="#8b5cf6"
              desc="ভারতীয় সাংকেতিক ভাষার শব্দক্রম সনাক্ত করে ইংরেজি SVO ক্রমে রূপান্তর করে।"
            />
            <FeatureRow icon="D" title="ডায়াগনস্টিক রিপোর্ট" color="#3b82f6"
              desc="প্রতিটি কনসেপ্টের জ্ঞানের মাত্রা। শিক্ষক জানেন ঠিক কী শেখাতে হবে।"
            />
            <FeatureRow icon="O" title="অফলাইন কাজ করে" color="#f59e0b"
              desc="PWA — যেকোনো ফোনে ইন্সটল। ইন্টারনেট ছাড়াই সমস্ত ফিচার কাজ করে।"
            />
            <FeatureRow icon="T" title="টেক্সট-টু-স্পিচ" color="#ec4899"
              desc="সম্পূর্ণ বাক্যটি জোরে পড়ে। যারা কথা বলতে পারেন না তাদের কণ্ঠস্বর।"
            />
            <FeatureRow icon="N" title="নিউরো-সিম্বলিক AI" color="#14b8a6"
              desc="ভিশন + অ্যাকোস্টিক + গেজ — বায়েসীয় ফিউশন। ফর্মাল গ্রামার ইঞ্জিন রিয়েল-টাইম যাচাই করে।"
            />
          </div>
        </div>

        {/* SECTION 2: HOW IT WORKS */}
        <div style={{ marginBottom: '1.25rem' }}>
          <SectionTitle color="#3b82f6">কীভাবে কাজ করে?</SectionTitle>
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '0.5rem',
            textAlign: 'center',
          }}>
            {[
              { num: 1, title: 'প্রোফাইল নির্বাচন', desc: 'মোটর / বধির / অন্ধ / ডিস্লেক্সিয়া' },
              { num: 2, title: 'ইশারা করুন', desc: 'ক্যামেরার সামনে হাতের ইশারা' },
              { num: 3, title: 'শব্দ আসে', desc: 'AI ইশারার আকৃতি সনাক্ত করে' },
              { num: 4, title: 'বাক্য তৈরি', desc: 'Subject-Verb-Object ক্রমে' },
              { num: 5, title: 'জোরে পড়ে', desc: 'TTS তাদের কণ্ঠস্বর হয়ে ওঠে' },
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
                <div style={{ fontSize: '0.6rem', color: '#94a3b8', lineHeight: 1.4 }}>
                  {step.desc}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* SECTION 3: PILOT INVITATION */}
        <div style={{ marginBottom: '1.25rem' }}>
          <SectionTitle color="#f59e0b">পাইলট স্টাডি আমন্ত্রণ</SectionTitle>
          <div style={{
            display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem',
            alignItems: 'start',
          }}>
            <div>
              <p style={{
                fontSize: '0.78rem', color: '#475569', lineHeight: 1.8, margin: '0 0 0.75rem',
              }}>
                আমরা খুঁজছি <strong>১০–১৫ জন শিক্ষার্থী</strong> (বয়স ৬–১৪) — সেরিব্রাল পলসি, নন-ভার্বাল অটিজম,
                বা শ্রবণ-প্রতিবন্ধী — একটি <strong>৮-সপ্তাহের পাইলট স্টাডির</strong> জন্য
                (সপ্তাহে ৩টি সেশন, প্রতি সেশন ২০ মিনিট)।
                আমরা ট্যাবলেট, প্রশিক্ষণ, এবং বিস্তারিত ডায়াগনস্টিক রিপোর্ট প্রদান করব।
                <strong>আপনার প্রতিষ্ঠানের কোনো খরচ নেই।</strong>
              </p>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.4rem', fontSize: '0.7rem' }}>
                <div style={{ background: '#f8fafc', padding: '0.5rem 0.75rem', borderRadius: 6 }}>
                  <strong style={{ color: '#1e293b' }}>প্রি/পোস্ট টেস্টিং</strong><br />
                  <span style={{ color: '#64748b' }}>২০-আইটেম বাক্য নির্মাণ পরীক্ষা</span>
                </div>
                <div style={{ background: '#f8fafc', padding: '0.5rem 0.75rem', borderRadius: 6 }}>
                  <strong style={{ color: '#1e293b' }}>নৈতিকতা অনুমোদিত</strong><br />
                  <span style={{ color: '#64748b' }}>অভিভাবকের সম্মতি + প্রাতিষ্ঠানিক রিভিউ</span>
                </div>
                <div style={{ background: '#f8fafc', padding: '0.5rem 0.75rem', borderRadius: 6 }}>
                  <strong style={{ color: '#1e293b' }}>ডেটা প্রাইভেসি</strong><br />
                  <span style={{ color: '#64748b' }}>সমস্ত ডেটা ডিভাইসেই থাকে। ক্লাউডে যায় না।</span>
                </div>
                <div style={{ background: '#f8fafc', padding: '0.5rem 0.75rem', borderRadius: 6 }}>
                  <strong style={{ color: '#1e293b' }}>প্রতি-শিশু রিপোর্ট</strong><br />
                  <span style={{ color: '#64748b' }}>নলেজ স্টেট, ISL প্যাটার্ন, প্রগ্রেস কার্ভ</span>
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
                  এখনই ট্রাই করুন:
                </div>
                <img
                  src="https://quickchart.io/qr?text=https%3A%2F%2Fmulti-modal-gesture-grammar.vercel.app&size=160&margin=1"
                  alt="MLAF ডেমো QR কোড"
                  style={{ width: 120, height: 120, marginBottom: '0.4rem' }}
                />
                <div style={{
                  fontSize: '0.58rem', color: '#3b82f6', wordBreak: 'break-all', fontWeight: 600,
                }}>
                  multi-modal-gesture-grammar.vercel.app
                </div>
              </div>

              <div style={{
                marginTop: '0.75rem', padding: '0.75rem',
                background: '#0F172A',
                borderRadius: 8, color: '#e2e8f0',
              }}>
                <div style={{ fontWeight: 800, fontSize: '0.75rem', marginBottom: '0.5rem' }}>
                  নীল শঙ্কর রায়
                </div>
                <div style={{ fontSize: '0.6rem', color: '#94a3b8', lineHeight: 1.6 }}>
                  NLP & Speech AI Engineer · Applied Linguist (MA, 14 yrs) · IIT Patna AI/ML<br />
                  neilshankarray@vaaani.in<br />
                  vaaani.in<br />
                  linkedin.com/in/neilsray
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* FOOTER */}
        <div style={{
          borderTop: '1px solid #e2e8f0', paddingTop: '0.75rem',
          fontSize: '0.58rem', color: '#94a3b8', textAlign: 'center', lineHeight: 1.6,
        }}>
          MLAF — মাল্টিমোডাল ভাষা অর্জন ফ্রেমওয়ার্ক · পেটেন্ট-পেন্ডিং 202631020540, ভারতীয় পেটেন্ট অফিস (২০২৬)
          <br />সর্বস্বত্ব সংরক্ষিত। বাণিজ্যিক লাইসেন্স অনুসন্ধান: neilshankarray@vaaani.in · vaaani.in
        </div>
      </div>
    </>
  );
}
