/**
 * gestureCatalog.js — THE single source of truth for every recognized gesture.
 *
 * Reliability Program, Fix #1 + #2 foundation. Every layer (guide sidebar,
 * finger-pattern matcher, geometric detector) must agree with THIS file. The
 * test in src/__tests__/gesture-integrity.test.js fails the build if any layer
 * drifts, or if any two gestures share an identical signature (collision).
 *
 * signature = finger-curl pattern  +  a `discriminator` the camera can actually
 * measure. Finger-curl alone collides (6 gestures are "all curled"); the
 * discriminator is the extra, measurable dimension that makes each unique:
 *   pinch        — thumb-tip ↔ index-tip touching (GRAB)
 *   c_gap        — thumb + index apart forming a C (DRINK)
 *   fist         — tight fist, thumb across palm (I)
 *   spread_wide  — fingers curved but splayed apart (BALL)
 *   bowl         — all slightly curved, thumb tucked in (FOOD)
 *   half         — partially curled, reaching out (WANT)
 *   at_face      — hand held near face / upper frame (EAT)
 *   palm_down    — hand horizontal, palm to floor (APPLE)
 *   palm_up      — hand horizontal, palm up (BOOK)
 *   vertical     — hand vertical, palm to camera (STOP, YOU)
 *   horizontal   — hand not vertical (GO)
 *   spread       — fingers splayed apart (THEY, WATER, SEE)
 *   together     — extended fingers held close (WE)
 *   tips_touch   — two fingertips touching, Λ shape (HOUSE)
 *   default      — no extra discriminator needed (unique on fingers alone)
 *
 * `c` = curled, `e` = extended.
 */

const f = (thumb, index, middle, ring, pinky) => ({ thumb, index, middle, ring, pinky });

export const GESTURE_CATALOG = {
  // ── SUBJECTS ──
  SUBJECT_I:    { category: 'SUBJECT', label: 'I',       guide: 'All fingers curled into a fist — every tip tucked in, thumb across palm', fingers: f('c','c','c','c','c'), discriminator: 'fist' },
  SUBJECT_YOU:  { category: 'SUBJECT', label: 'YOU',     guide: 'Index finger extended and pointing up, all other fingers curled',         fingers: f('c','e','c','c','c'), discriminator: 'vertical' },
  SUBJECT_HE:   { category: 'SUBJECT', label: 'HE / SHE',guide: 'Index + Middle + Ring extended, Thumb + Pinky curled',                    fingers: f('c','e','e','e','c'), discriminator: 'default' },
  SUBJECT_SHE:  { category: 'SUBJECT', label: 'HE / SHE',guide: 'Index + Middle + Ring extended, Thumb + Pinky curled (context picks HE/SHE)', fingers: f('c','e','e','e','c'), discriminator: 'default', aliasOf: 'SUBJECT_HE' },
  SUBJECT_WE:   { category: 'SUBJECT', label: 'WE',      guide: 'Index + Middle extended and held close together, others curled',          fingers: f('c','e','e','c','c'), discriminator: 'together' },
  SUBJECT_THEY: { category: 'SUBJECT', label: 'THEY',    guide: 'All five fingers extended and spread apart',                              fingers: f('e','e','e','e','e'), discriminator: 'spread' },

  // ── VERBS ──
  GRAB:  { category: 'VERB', label: 'GRAB',  guide: 'Thumb tip touches index tip (pinch) — others relaxed/curled', fingers: f('c','c','c','c','c'), discriminator: 'pinch' },
  DRINK: { category: 'VERB', label: 'DRINK', guide: 'Thumb + index apart forming a C-gap, middle/ring/pinky curled', fingers: f('e','e','c','c','c'), discriminator: 'c_gap' },
  EAT:   { category: 'VERB', label: 'EAT',   guide: 'Fingertips bunched together, hand held near face (upper frame)', fingers: f('c','c','c','c','c'), discriminator: 'at_face' },
  WANT:  { category: 'VERB', label: 'WANT',  guide: 'Fingers partially curled (half-closed), reaching outward',     fingers: f('c','c','c','c','c'), discriminator: 'half' },
  SEE:   { category: 'VERB', label: 'SEE',   guide: 'Index + Middle extended and spread in a V, others curled',     fingers: f('c','e','e','c','c'), discriminator: 'spread' },
  GO:    { category: 'VERB', label: 'GO',    guide: 'Index extended, others curled, relaxed thumb — hand not vertical', fingers: f('c','e','c','c','c'), discriminator: 'horizontal' },
  STOP:  { category: 'VERB', label: 'STOP',  guide: 'All five fingers extended, palm facing camera, hand vertical', fingers: f('e','e','e','e','e'), discriminator: 'vertical' },

  // ── OBJECTS ──
  APPLE: { category: 'OBJECT', label: 'APPLE', guide: 'Hand horizontal, all fingers extended together, palm facing floor', fingers: f('e','e','e','e','e'), discriminator: 'palm_down' },
  BALL:  { category: 'OBJECT', label: 'BALL',  guide: 'Fingers curved and splayed wide apart, like holding a large ball',  fingers: f('c','c','c','c','c'), discriminator: 'spread_wide' },
  WATER: { category: 'OBJECT', label: 'WATER', guide: 'Index + Middle + Ring extended and spread, Thumb + Pinky curled',   fingers: f('c','e','e','e','c'), discriminator: 'spread' },
  FOOD:  { category: 'OBJECT', label: 'FOOD',  guide: 'All fingers slightly curved, thumb curled in — like holding a bowl', fingers: f('c','c','c','c','c'), discriminator: 'bowl' },
  BOOK:  { category: 'OBJECT', label: 'BOOK',  guide: 'All fingers extended flat, palm up, hand horizontal — open book',   fingers: f('e','e','e','e','e'), discriminator: 'palm_up' },
  HOUSE: { category: 'OBJECT', label: 'HOUSE', guide: 'Index + Middle extended with tips touching (Λ), others curled',     fingers: f('c','e','e','c','c'), discriminator: 'tips_touch' },
};

/** The full signature string used for collision-checking. */
export function signatureOf(entry) {
  const p = entry.fingers;
  return `${p.thumb}${p.index}${p.middle}${p.ring}${p.pinky}:${entry.discriminator}`;
}
