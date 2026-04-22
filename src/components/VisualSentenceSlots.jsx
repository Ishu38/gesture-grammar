/**
 * VisualSentenceSlots.jsx — SVO Slot-Based Sentence Builder Visualization
 * Shows three empty slots (Subject → Verb → Object) that fill as the user
 * builds a sentence. Highlights the "next expected" slot with a pulsing border.
 */

import { LEXICON } from '../utils/GrammarEngine';

const SLOT_DEFS = [
  { role: 'SUBJECT', label: 'Subject', color: '#60a5fa', emptyText: 'Who?' },
  { role: 'VERB',    label: 'Verb',    color: '#f87171', emptyText: 'Does what?' },
  { role: 'OBJECT',  label: 'Object',  color: '#4ade80', emptyText: 'To what?' },
];

function SlotBox({ slot, word, isNext, hidden }) {
  const filled = !!word;

  if (hidden) {
    return <div style={{ width: 110, height: 70 }} />;
  }

  return (
    <div style={{
      width: 110, minHeight: 70,
      border: filled
        ? `2px solid ${slot.color}`
        : `2px dashed ${isNext ? slot.color : '#4a4a6a'}`,
      borderRadius: 10,
      background: filled
        ? `${slot.color}15`
        : 'rgba(15, 15, 26, 0.5)',
      display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      padding: '6px 8px', gap: 2,
      transition: 'all 0.3s ease',
      animation: isNext && !filled ? 'slotPulse 2s ease-in-out infinite' : filled ? 'slotPopIn 0.35s cubic-bezier(0.34, 1.56, 0.64, 1)' : 'none',
    }}>
      {/* Category label */}
      <div style={{
        fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
        letterSpacing: 1, color: filled ? slot.color : '#64748b',
      }}>
        {slot.label}
      </div>

      {/* Word or empty prompt */}
      <div style={{
        fontSize: filled ? 14 : 11,
        fontWeight: filled ? 700 : 400,
        color: filled ? '#e2e8f0' : '#64748b',
        textAlign: 'center',
      }}>
        {filled ? word.word : slot.emptyText}
      </div>

      {/* Next indicator */}
      {isNext && !filled && (
        <div style={{
          fontSize: 9, color: slot.color, fontWeight: 600, marginTop: 2,
        }}>
          Next
        </div>
      )}
    </div>
  );
}

export default function VisualSentenceSlots({ sentence = [], currentGesture }) {
  // Determine filled state for each slot
  const subjectWord = sentence.find(w => w.type === 'SUBJECT');
  const verbWord = sentence.find(w => w.type === 'VERB');
  const objectWord = sentence.find(w => w.type === 'OBJECT');

  // Determine if the verb is transitive (needs an object)
  const verbEntry = verbWord ? LEXICON[verbWord.grammar_id] : null;
  const needsObject = verbEntry ? verbEntry.transitive !== false : true;

  // Determine next expected slot
  let nextSlot = null;
  if (!subjectWord) nextSlot = 'SUBJECT';
  else if (!verbWord) nextSlot = 'VERB';
  else if (!objectWord && needsObject) nextSlot = 'OBJECT';

  const slotWords = [subjectWord, verbWord, objectWord];

  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      gap: 6, padding: '8px 0',
    }}>
      {SLOT_DEFS.map((slot, i) => (
        <div key={slot.role} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <SlotBox
            slot={slot}
            word={slotWords[i]}
            isNext={nextSlot === slot.role}
            hidden={slot.role === 'OBJECT' && verbWord && !needsObject}
          />
          {/* Arrow between slots */}
          {i < SLOT_DEFS.length - 1 && !(slot.role === 'VERB' && verbWord && !needsObject) && (
            <div style={{
              color: '#4a4a6a', fontSize: 18, fontWeight: 300, userSelect: 'none',
            }}>
              ›
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
