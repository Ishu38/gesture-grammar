/**
 * ISLInterferenceDetector.js — ISL-English Syntactic Transfer Error Detector
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Detects L1 syntactic interference patterns from Indian Sign Language (ISL)
 * in learners constructing English sentences via gesture input.
 *
 * ISL typological profile (relevant to this module):
 *   - Word order: SOV (Subject-Object-Verb) — canonical
 *   - Topic prominence: High — objects may be fronted for topicalization
 *   - Null elements: Copula (is/am/are) frequently omitted
 *   - Spatial grammar: Verb agreement encoded spatially, not morphologically
 *
 * English typological profile (target language):
 *   - Word order: SVO (Subject-Verb-Object) — rigid
 *   - Topic prominence: Low — fronting is marked and emphatic
 *   - Copula: Obligatory in predicative constructions
 *   - Morphological: Person/number agreement on verbs (S-form)
 *
 * Interference patterns detected:
 *   1. SOV_ORDER        — ISL canonical SOV word order transferred to English
 *   2. TOPIC_FRONTING   — ISL topic-prominence: object fronted before subject
 *   3. TRANSITIVE_OBJECT_DROP — Object omitted from transitive verb construction
 *
 * Cognitive science grounding:
 *   Birsh & Carreker (2018), Ch. 19: "Language and Literacy Development Among
 *   English Language Learners." L1 syntactic transfer theory; contrastive
 *   analysis of L1 and L2 grammar as basis for error prediction and correction.
 *   Cummins (1979): Linguistic Interdependence Hypothesis.
 *   Odlin (1989): Language Transfer — Cross-Linguistic Influence in Language Learning.
 *
 * Patent claim:
 *   "A method and system for real-time detection and correction of syntactic
 *   transfer errors in L2 English learners whose L1 is a visuo-spatial language
 *   (Indian Sign Language), using gesture sequence analysis within a multimodal
 *   language acquisition framework running locally on a mobile device."
 */

// =============================================================================
// INTERFERENCE PATTERN DEFINITIONS
// =============================================================================

/**
 * Canonical ISL interference patterns with linguistic grounding and corrections.
 * Each entry defines: the ISL structural pattern, why it occurs typologically,
 * and how to correct it toward English target grammar.
 */
export const ISL_INTERFERENCE_PATTERNS = {
  SOV_ORDER: {
    id: 'SOV_ORDER',
    severity: 'error',
    title: 'ISL Word Order (SOV)',
    isl_structure: 'Subject → Object → Verb',
    english_structure: 'Subject → Verb → Object',
    example_isl: '"I apple eat"',
    example_english: '"I eat apple"',
    linguistic_note:
      'ISL uses Subject-Object-Verb (SOV) order. English requires Subject-Verb-Object ' +
      '(SVO). This is the most frequent syntactic transfer error for ISL-background ' +
      'learners of English (Birsh & Carreker, 2018, Ch. 19).',
    correction_template: (subject, verb, object) =>
      `In English, the verb comes before the object: "${subject} ${verb} ${object}"`,
  },

  TOPIC_FRONTING: {
    id: 'TOPIC_FRONTING',
    severity: 'error',
    title: 'ISL Topic Fronting',
    isl_structure: 'Object → Subject → Verb  (topic-prominent)',
    english_structure: 'Subject → Verb → Object',
    example_isl: '"Apple, I eat"',
    example_english: '"I eat apple"',
    linguistic_note:
      'ISL is a topic-prominent language: the discourse topic is placed first, ' +
      'regardless of its grammatical role. In English this construction is ' +
      'marked and sounds unusual in everyday speech.',
    correction_template: (subject, verb, object) =>
      `Start with the subject: "${subject} ${verb} ${object}"`,
  },

  TRANSITIVE_OBJECT_DROP: {
    id: 'TRANSITIVE_OBJECT_DROP',
    severity: 'warning',
    title: 'Missing Object',
    isl_structure: 'Subject → Verb  (object implied from context)',
    english_structure: 'Subject → Verb → Object  (required)',
    example_isl: '"I eat"  (the food is visible in the environment)',
    example_english: '"I eat apple"  (object must be stated)',
    linguistic_note:
      'ISL can omit objects whose referent is established in the signing ' +
      'environment or context. English transitive verbs require an explicit ' +
      'object even when the referent is contextually clear.',
    correction_template: (subject, verb) =>
      `"${verb}" is transitive — add what is being ${verb}ed: "${subject} ${verb} [object]"`,
  },
};

// =============================================================================
// ISL INTERFERENCE DETECTOR
// =============================================================================

/**
 * Detects ISL syntactic interference patterns in a gesture-built sentence.
 *
 * Usage:
 *   const detector = new ISLInterferenceDetector();
 *
 *   // Called reactively as the sentence array changes:
 *   const report = detector.analyze(sentence);
 *   // sentence = word objects from useSentenceBuilder:
 *   // [{ grammar_id, type, word, transitive, ... }, ...]
 *
 *   // For session-level trend data:
 *   const trend = detector.getTrend();
 */
export class ISLInterferenceDetector {
  constructor() {
    // History of interference events across sentences in the session.
    // Used for adaptive curriculum adjustment and patent-supporting data collection.
    this.detectionHistory = [];
  }

  /**
   * Analyze a sentence array for ISL interference patterns.
   * Called reactively on every sentence change.
   *
   * @param {Array} sentence — word objects from useSentenceBuilder
   * @returns {ISLInterferenceReport}
   */
  analyze(sentence) {
    if (!sentence || sentence.length === 0) {
      return this._emptyReport();
    }

    const patterns = [];

    const sovResult = this._detectSOVOrder(sentence);
    if (sovResult) patterns.push(sovResult);

    const topicResult = this._detectTopicFronting(sentence);
    if (topicResult) patterns.push(topicResult);

    const dropResult = this._detectTransitiveObjectDrop(sentence);
    if (dropResult) patterns.push(dropResult);

    const report = {
      hasInterference: patterns.length > 0,
      patterns,
      severity: this._computeOverallSeverity(patterns),
      sentence_display: sentence.map(w => w.word).join(' '),
      timestamp: Date.now(),
    };

    if (report.hasInterference) {
      this.detectionHistory.push({
        timestamp: report.timestamp,
        pattern_ids: patterns.map(p => p.id),
        sentence: report.sentence_display,
      });
    }

    return report;
  }

  /**
   * Return frequency analysis of interference patterns across the session.
   * This data can be used for:
   *   - Adaptive curriculum: focus on the most-repeated error type
   *   - Session logging: feed into empirical data for patent evidence
   *
   * @returns {{ total_detections, pattern_frequency, most_frequent }}
   */
  getTrend() {
    const frequency = {};
    for (const event of this.detectionHistory) {
      for (const id of event.pattern_ids) {
        frequency[id] = (frequency[id] || 0) + 1;
      }
    }
    return {
      total_detections: this.detectionHistory.length,
      pattern_frequency: frequency,
      most_frequent: Object.entries(frequency)
        .sort(([, a], [, b]) => b - a)[0]?.[0] || null,
      history: this.detectionHistory,
    };
  }

  /**
   * Clear session history.
   */
  resetHistory() {
    this.detectionHistory = [];
  }

  // ===========================================================================
  // PRIVATE — Detection algorithms
  // ===========================================================================

  /**
   * SOV ORDER DETECTION
   *
   * ISL canonical order: Subject → Object → Verb
   * English target:      Subject → Verb   → Object
   *
   * Algorithm: Locate the first OBJECT token and first VERB token by position.
   * If OBJECT precedes VERB, SOV transfer is detected.
   * (Does not double-fire with TOPIC_FRONTING — only fires when subject is first.)
   */
  _detectSOVOrder(sentence) {
    const firstObjectIndex = sentence.findIndex(w => w.type === 'OBJECT');
    const firstVerbIndex = sentence.findIndex(w => w.type === 'VERB');

    if (firstObjectIndex === -1 || firstVerbIndex === -1) return null;
    if (firstObjectIndex >= firstVerbIndex) return null;

    // Exclude topic-fronting case (object is the very first token)
    // That case is handled by _detectTopicFronting
    if (firstObjectIndex === 0) return null;

    const subject = sentence.find(w => w.type === 'SUBJECT');
    const verb = sentence.find(w => w.type === 'VERB');
    const object = sentence.find(w => w.type === 'OBJECT');

    const pattern = ISL_INTERFERENCE_PATTERNS.SOV_ORDER;
    return {
      id: pattern.id,
      severity: pattern.severity,
      title: pattern.title,
      detected_at_position: firstObjectIndex,
      description:
        `"${object?.word}" (object) appears before "${verb?.word}" (verb) — ` +
        `ISL SOV word order detected.`,
      correction: pattern.correction_template(
        subject?.word || '[subject]',
        verb?.word || '[verb]',
        object?.word || '[object]'
      ),
      linguistic_note: pattern.linguistic_note,
      example_isl: pattern.example_isl,
      example_english: pattern.example_english,
    };
  }

  /**
   * TOPIC FRONTING DETECTION
   *
   * ISL topic-prominent pattern: Object → Subject → Verb
   * English target:              Subject → Verb   → Object
   *
   * Algorithm: If the first token in the sentence is an OBJECT, topic fronting
   * is detected. (ISL places the topic — often the object — sentence-initially.)
   */
  _detectTopicFronting(sentence) {
    if (sentence.length === 0) return null;
    if (sentence[0].type !== 'OBJECT') return null;

    const subject = sentence.find(w => w.type === 'SUBJECT');
    const verb = sentence.find(w => w.type === 'VERB');
    const object = sentence[0]; // the fronted object

    const pattern = ISL_INTERFERENCE_PATTERNS.TOPIC_FRONTING;
    return {
      id: pattern.id,
      severity: pattern.severity,
      title: pattern.title,
      detected_at_position: 0,
      description:
        `Sentence begins with "${object.word}" (object) — ` +
        `ISL topic fronting detected.`,
      correction: pattern.correction_template(
        subject?.word || '[subject]',
        verb?.word || '[verb]',
        object?.word || '[object]'
      ),
      linguistic_note: pattern.linguistic_note,
      example_isl: pattern.example_isl,
      example_english: pattern.example_english,
    };
  }

  /**
   * TRANSITIVE OBJECT DROP DETECTION
   *
   * ISL context-drop pattern: Subject → Verb  (object recoverable from environment)
   * English target:           Subject → Verb → Object  (explicit object required)
   *
   * Algorithm: If the last token is a transitive VERB and no OBJECT exists
   * in the sentence, object drop is detected. Only fires when sentence has
   * ≥ 2 tokens (subject + verb minimum) to avoid false positives mid-gesture.
   */
  _detectTransitiveObjectDrop(sentence) {
    if (sentence.length < 2) return null;

    const lastToken = sentence[sentence.length - 1];
    if (lastToken.type !== 'VERB') return null;
    if (!lastToken.transitive) return null;

    const hasObject = sentence.some(w => w.type === 'OBJECT');
    if (hasObject) return null;

    const subject = sentence.find(w => w.type === 'SUBJECT');
    const pattern = ISL_INTERFERENCE_PATTERNS.TRANSITIVE_OBJECT_DROP;

    return {
      id: pattern.id,
      severity: pattern.severity,
      title: pattern.title,
      detected_at_position: sentence.length - 1,
      description:
        `"${lastToken.word}" is a transitive verb — an object is required in English.`,
      correction: pattern.correction_template(
        subject?.word || '[subject]',
        lastToken.word
      ),
      linguistic_note: pattern.linguistic_note,
      example_isl: pattern.example_isl,
      example_english: pattern.example_english,
    };
  }

  _computeOverallSeverity(patterns) {
    if (patterns.length === 0) return 'none';
    if (patterns.some(p => p.severity === 'error')) return 'error';
    return 'warning';
  }

  _emptyReport() {
    return {
      hasInterference: false,
      patterns: [],
      severity: 'none',
      sentence_display: '',
      timestamp: Date.now(),
    };
  }
}

export default ISLInterferenceDetector;
