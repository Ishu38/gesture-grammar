/**
 * ContrastiveDisplay.jsx — ISL ↔ English Word Order Comparison
 *
 * When ISLInterferenceDetector detects a word-order error (SOV, topic-fronting),
 * this component shows the user's ISL-influenced order vs the correct English
 * SVO order with visual reordering arrows.
 */

const CATEGORY_COLORS = {
  SUBJECT: '#60a5fa',
  VERB: '#f87171',
  OBJECT: '#4ade80',
};

function WordBlock({ word, muted }) {
  const color = CATEGORY_COLORS[word.type] || '#94a3b8';

  return (
    <div style={{
      display: 'inline-flex', flexDirection: 'column', alignItems: 'center',
      padding: '6px 12px', borderRadius: 8,
      border: `2px solid ${muted ? '#4a4a6a' : color}`,
      background: muted ? 'rgba(74, 74, 106, 0.15)' : `${color}15`,
      minWidth: 60,
    }}>
      <span style={{
        fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
        letterSpacing: 0.8,
        color: muted ? '#64748b' : color,
      }}>
        {word.type}
      </span>
      <span style={{
        fontSize: 13, fontWeight: 600,
        color: muted ? '#94a3b8' : '#e2e8f0',
      }}>
        {word.word}
      </span>
    </div>
  );
}

function EmptySlot({ type }) {
  const color = CATEGORY_COLORS[type] || '#94a3b8';
  return (
    <div style={{
      display: 'inline-flex', flexDirection: 'column', alignItems: 'center',
      padding: '6px 12px', borderRadius: 8,
      border: `2px dashed ${color}`,
      background: `${color}08`,
      minWidth: 60,
    }}>
      <span style={{ fontSize: 9, fontWeight: 700, color, letterSpacing: 0.8 }}>
        {type}
      </span>
      <span style={{ fontSize: 13, color, fontWeight: 600 }}>?</span>
    </div>
  );
}

function WordRow({ words, label, muted }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        fontSize: 10, fontWeight: 600, color: muted ? '#64748b' : '#94a3b8',
        marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1,
      }}>
        {label}
      </div>
      <div style={{ display: 'flex', gap: 8, justifyContent: 'center', alignItems: 'center' }}>
        {words.map((w, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {w.empty
              ? <EmptySlot type={w.type} />
              : <WordBlock word={w} muted={muted} />}
            {i < words.length - 1 && (
              <span style={{ color: '#4a4a6a', fontSize: 14 }}>→</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ContrastiveDisplay({ interferenceReport, sentence }) {
  if (!interferenceReport || !interferenceReport.hasInterference) return null;
  if (!sentence || sentence.length === 0) return null;

  const pattern = interferenceReport.patterns[0]; // Primary pattern
  if (!pattern) return null;

  // Build the user's current order (ISL-influenced)
  const islOrder = sentence
    .filter(w => w && w.type)
    .map(w => ({
      word: w.word || w.display || '?',
      type: w.type,
      grammar_id: w.grammar_id,
    }));

  // Build the correct English SVO order
  const subject = sentence.find(w => w.type === 'SUBJECT');
  const verb = sentence.find(w => w.type === 'VERB');
  const object = sentence.find(w => w.type === 'OBJECT');

  const englishOrder = [];
  if (subject) englishOrder.push({ word: subject.word, type: 'SUBJECT' });
  else englishOrder.push({ word: '—', type: 'SUBJECT', empty: true });

  if (verb) englishOrder.push({ word: verb.word, type: 'VERB' });
  else englishOrder.push({ word: '—', type: 'VERB', empty: true });

  if (pattern.id === 'TRANSITIVE_OBJECT_DROP') {
    englishOrder.push({ word: '?', type: 'OBJECT', empty: true });
  } else if (object) {
    englishOrder.push({ word: object.word, type: 'OBJECT' });
  }

  // Determine the ISL pattern label
  const patternLabels = {
    SOV_ORDER: 'ISL uses Subject-Object-Verb order',
    TOPIC_FRONTING: 'ISL puts the topic (object) first',
    TRANSITIVE_OBJECT_DROP: 'ISL allows dropping the object (implied from context)',
  };

  return (
    <div style={{
      background: '#1a1a2e', borderRadius: 10, padding: '12px 16px',
      border: '1px solid #3a2a4e', marginTop: 8,
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10,
      }}>
        <span style={{
          background: '#f97316', color: '#0f0f1a', fontSize: 10, fontWeight: 700,
          padding: '2px 8px', borderRadius: 4, textTransform: 'uppercase',
        }}>
          ISL ↔ English
        </span>
        <span style={{ fontSize: 11, color: '#f97316', fontWeight: 600 }}>
          {pattern.title}
        </span>
      </div>

      {/* Side by side comparison */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
        {/* ISL order (user's current) */}
        <WordRow words={islOrder} label="Your order (ISL pattern)" muted />

        {/* Reorder arrows */}
        <div style={{ color: '#f97316', fontSize: 18, padding: '2px 0' }}>
          ↕
        </div>

        {/* English SVO order (correct) */}
        <WordRow words={englishOrder} label="English order (SVO)" muted={false} />
      </div>

      {/* Linguistic explanation */}
      <div style={{
        marginTop: 10, padding: '8px 10px',
        background: 'rgba(249, 115, 22, 0.08)',
        borderRadius: 6, borderLeft: '3px solid #f97316',
      }}>
        <div style={{ fontSize: 11, color: '#f97316', fontWeight: 600, marginBottom: 3 }}>
          Why this happens:
        </div>
        <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.4 }}>
          {patternLabels[pattern.id] || pattern.description}
          {'. '}In English, sentences follow <strong style={{ color: '#e2e8f0' }}>Subject → Verb → Object</strong> order.
        </div>
        {pattern.correction && (
          <div style={{ fontSize: 11, color: '#4ade80', marginTop: 4 }}>
            Correction: {pattern.correction}
          </div>
        )}
      </div>
    </div>
  );
}
