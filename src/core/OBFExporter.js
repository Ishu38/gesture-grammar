/**
 * OBFExporter.js — Open Board Format (.obf) Export for AAC Interoperability
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Exports MLAF's grammar vocabulary and constructed sentences as Open Board
 * Format (OBF) JSON files. OBF is the open standard for AAC communication
 * boards, supported by CoughDrop, Open Board Project, AACKeys, and others.
 *
 * OBF Specification: https://www.openboardformat.org/
 *
 * Export modes:
 *   1. Vocabulary Board — full MLAF lexicon as a grid of labeled buttons
 *   2. Sentence Board — a specific constructed sentence as a single-row board
 *   3. Phrase Collection — multiple saved phrases as a multi-page board set
 *
 * Each button includes:
 *   - label (display text)
 *   - vocalization (what to speak aloud)
 *   - semantic category (subject/verb/object)
 *   - background color coded by grammar role
 */

import { LEXICON } from '../utils/GrammarEngine.js';

// =============================================================================
// OBF COLOR SCHEME (matches MLAF grammar role colors)
// =============================================================================

const OBF_COLORS = {
  SUBJECT: { r: 37,  g: 99,  b: 235, a: 255 },  // Blue
  VERB:    { r: 220, g: 38,  b: 38,  a: 255 },   // Red
  OBJECT:  { r: 22,  g: 163, b: 74,  a: 255 },   // Green
};

// =============================================================================
// OBF EXPORTER
// =============================================================================

export class OBFExporter {

  /**
   * Export the full MLAF vocabulary as an OBF board.
   * Organizes tokens into rows by grammatical role:
   *   Row 1: Subjects (I, You, He, She, We, They)
   *   Row 2: Verbs (grab, go, eat, want, stop, drink, see)
   *   Row 3: Objects (apple, ball, water, food, book, house)
   *
   * @returns {object} OBF-compliant board JSON
   */
  static exportVocabularyBoard() {
    const subjects = [];
    const verbs = [];
    const objects = [];

    for (const [id, entry] of Object.entries(LEXICON)) {
      // Skip S-form verbs — export only base forms for AAC simplicity
      if (entry.requires_s_form) continue;

      const button = OBFExporter._createButton(id, entry);

      if (entry.type === 'SUBJECT') subjects.push(button);
      else if (entry.type === 'VERB') verbs.push(button);
      else if (entry.type === 'OBJECT') objects.push(button);
    }

    const allButtons = [...subjects, ...verbs, ...objects];
    const columns = Math.max(subjects.length, verbs.length, objects.length);

    // Build grid: assign row/column positions
    const gridOrder = [];
    [subjects, verbs, objects].forEach((row, rowIdx) => {
      row.forEach((btn, colIdx) => {
        gridOrder.push(btn.id);
      });
      // Fill remaining columns with null
      for (let i = row.length; i < columns; i++) {
        gridOrder.push(null);
      }
    });

    return {
      format: 'open-board-0.1',
      id: 'mlaf-vocabulary',
      locale: 'en',
      name: 'MLAF Grammar Vocabulary',
      description_html: 'Grammar tokens from the Multimodal Language Acquisition Framework. Subjects (blue), Verbs (red), Objects (green).',
      buttons: allButtons,
      grid: {
        rows: 3,
        columns: columns,
        order: OBFExporter._chunkArray(gridOrder, columns),
      },
      license: { type: 'private', author: 'MLAF / Neil Shankar Ray' },
    };
  }

  /**
   * Export a constructed sentence as a single-row OBF board.
   * Each word in the sentence becomes a button that speaks that word.
   * A final "Speak All" button vocalizes the full sentence.
   *
   * @param {Array} sentence — word objects from useSentenceBuilder
   *   Each: { grammar_id, word, display, category, ... }
   * @returns {object} OBF-compliant board JSON
   */
  static exportSentenceBoard(sentence) {
    if (!sentence || sentence.length === 0) {
      return null;
    }

    const buttons = sentence.map((word, idx) => {
      const lexEntry = LEXICON[word.grammar_id] || {};
      return OBFExporter._createButton(
        word.grammar_id || `word_${idx}`,
        {
          type: word.category || lexEntry.type || 'OBJECT',
          display: word.word || word.display || word.grammar_id,
        }
      );
    });

    // Add "Speak All" button
    const fullSentence = sentence.map(w => w.word || w.display || w.grammar_id).join(' ');
    buttons.push({
      id: 'speak_all',
      label: 'Speak All',
      vocalization: fullSentence,
      background_color: { r: 147, g: 51, b: 234, a: 255 }, // Purple
      border_color: { r: 107, g: 33, b: 168, a: 255 },
    });

    const columns = buttons.length;

    return {
      format: 'open-board-0.1',
      id: `mlaf-sentence-${Date.now()}`,
      locale: 'en',
      name: fullSentence,
      description_html: `Sentence constructed in MLAF: "${fullSentence}"`,
      buttons,
      grid: {
        rows: 1,
        columns,
        order: [buttons.map(b => b.id)],
      },
      license: { type: 'private', author: 'MLAF / Neil Shankar Ray' },
    };
  }

  /**
   * Export a collection of saved phrases as a multi-page OBF board set (.obz manifest).
   *
   * @param {Array<{sentence: string, words: Array}>} phrases — saved phrases
   * @returns {object} OBF board set manifest
   */
  static exportPhraseCollection(phrases) {
    if (!phrases || phrases.length === 0) return null;

    const boards = phrases.map((phrase, idx) => {
      const board = OBFExporter.exportSentenceBoard(phrase.words);
      if (board) {
        board.id = `mlaf-phrase-${idx}`;
        board.name = phrase.sentence;
      }
      return board;
    }).filter(Boolean);

    // Create index board — one button per phrase
    const indexButtons = phrases.map((phrase, idx) => ({
      id: `phrase_${idx}`,
      label: phrase.sentence,
      vocalization: phrase.sentence,
      background_color: { r: 59, g: 130, b: 246, a: 255 },
      border_color: { r: 37, g: 99, b: 235, a: 255 },
      load_board: { id: `mlaf-phrase-${idx}` },
    }));

    const indexBoard = {
      format: 'open-board-0.1',
      id: 'mlaf-phrases-index',
      locale: 'en',
      name: 'MLAF Saved Phrases',
      description_html: 'Phrases constructed and saved in MLAF sessions.',
      buttons: indexButtons,
      grid: {
        rows: Math.ceil(indexButtons.length / 4),
        columns: Math.min(4, indexButtons.length),
        order: OBFExporter._chunkArray(
          indexButtons.map(b => b.id),
          Math.min(4, indexButtons.length)
        ),
      },
      license: { type: 'private', author: 'MLAF / Neil Shankar Ray' },
    };

    return {
      format: 'open-board-0.1',
      root: 'mlaf-phrases-index',
      boards: [indexBoard, ...boards],
    };
  }

  /**
   * Download an OBF board as a .obf JSON file.
   *
   * @param {object} board — OBF board object
   * @param {string} [filename='mlaf-board.obf'] — download filename
   */
  static downloadBoard(board, filename = 'mlaf-board.obf') {
    if (!board) return;
    const json = JSON.stringify(board, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  // ===========================================================================
  // PRIVATE
  // ===========================================================================

  static _createButton(id, entry) {
    const type = entry.type || 'OBJECT';
    const color = OBF_COLORS[type] || OBF_COLORS.OBJECT;
    const display = entry.display || id;

    return {
      id: id.toLowerCase(),
      label: display.charAt(0).toUpperCase() + display.slice(1),
      vocalization: display.toLowerCase(),
      background_color: { ...color },
      border_color: {
        r: Math.max(0, color.r - 30),
        g: Math.max(0, color.g - 30),
        b: Math.max(0, color.b - 30),
        a: 255,
      },
    };
  }

  static _chunkArray(arr, size) {
    const chunks = [];
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size));
    }
    return chunks;
  }
}

export default OBFExporter;
