/**
 * PhraseBankManager.js — Sentence Phrase Bank with Persistence
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Persists commonly constructed sentences to localStorage so a non-speaking
 * child's grammar work survives beyond the session. Provides:
 *
 *   - Save: store a completed sentence with its word objects
 *   - Load: retrieve all saved phrases
 *   - Delete: remove a phrase
 *   - Copy: copy sentence text to clipboard
 *   - Share: share via Web Share API (WhatsApp, SMS, email on mobile)
 *   - Speak: replay sentence via Web Speech API
 *
 * Storage format (localStorage key: 'mlaf_phrase_bank'):
 *   [{ id, sentence, words, savedAt, timesSpoken }]
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const STORAGE_KEY = 'mlaf_phrase_bank';
const MAX_PHRASES = 50;

// =============================================================================
// PHRASE BANK MANAGER
// =============================================================================

export class PhraseBankManager {
  constructor() {
    this._phrases = this._load();
  }

  // ===========================================================================
  // PUBLIC — CRUD
  // ===========================================================================

  /**
   * Save a completed sentence to the phrase bank.
   *
   * @param {Array} wordObjects — word objects from useSentenceBuilder
   *   Each: { grammar_id, word, display, category, ... }
   * @returns {object} The saved phrase entry
   */
  save(wordObjects) {
    if (!wordObjects || wordObjects.length === 0) return null;

    const sentence = wordObjects
      .map(w => w.word || w.display || w.grammar_id)
      .join(' ');

    // Deduplicate — don't save the same sentence twice
    const existing = this._phrases.find(p => p.sentence === sentence);
    if (existing) {
      existing.savedAt = Date.now();
      this._persist();
      return existing;
    }

    const entry = {
      id: `phrase_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      sentence,
      words: wordObjects.map(w => ({
        grammar_id: w.grammar_id,
        word: w.word || w.display,
        category: w.category || w.type,
      })),
      savedAt: Date.now(),
      timesSpoken: 0,
    };

    this._phrases.unshift(entry);

    // Cap at MAX_PHRASES
    if (this._phrases.length > MAX_PHRASES) {
      this._phrases = this._phrases.slice(0, MAX_PHRASES);
    }

    this._persist();
    return entry;
  }

  /**
   * Get all saved phrases.
   * @returns {Array<object>}
   */
  getAll() {
    return [...this._phrases];
  }

  /**
   * Get phrase count.
   * @returns {number}
   */
  count() {
    return this._phrases.length;
  }

  /**
   * Delete a phrase by ID.
   * @param {string} phraseId
   * @returns {boolean} true if deleted
   */
  delete(phraseId) {
    const before = this._phrases.length;
    this._phrases = this._phrases.filter(p => p.id !== phraseId);
    if (this._phrases.length < before) {
      this._persist();
      return true;
    }
    return false;
  }

  /**
   * Clear all saved phrases.
   */
  clearAll() {
    this._phrases = [];
    this._persist();
  }

  // ===========================================================================
  // PUBLIC — Output Actions
  // ===========================================================================

  /**
   * Copy sentence text to clipboard.
   * @param {string} sentence
   * @returns {Promise<boolean>}
   */
  async copyToClipboard(sentence) {
    if (!sentence) return false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(sentence);
        return true;
      }
      // Fallback for older browsers
      return this._fallbackCopy(sentence);
    } catch {
      return this._fallbackCopy(sentence);
    }
  }

  /**
   * Share sentence via Web Share API (mobile: WhatsApp, SMS, email).
   * @param {string} sentence
   * @returns {Promise<boolean>}
   */
  async share(sentence) {
    if (!sentence) return false;

    if (navigator.share) {
      try {
        await navigator.share({
          title: 'MLAF Sentence',
          text: sentence,
        });
        return true;
      } catch (err) {
        // User cancelled share — not an error
        if (err.name === 'AbortError') return false;
        // Fallback to clipboard
        return this.copyToClipboard(sentence);
      }
    }

    // No Web Share API — fallback to clipboard
    return this.copyToClipboard(sentence);
  }

  /**
   * Speak a sentence aloud via Web Speech API.
   * @param {string} sentence
   * @param {number} [rate=0.9]
   * @returns {boolean} true if speech started
   */
  speak(sentence, rate = 0.9) {
    if (!sentence || typeof window === 'undefined' || !('speechSynthesis' in window)) {
      return false;
    }

    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(sentence);
    utterance.rate = rate;
    utterance.pitch = 1.0;

    // Prefer English voice
    const voices = speechSynthesis.getVoices();
    const english = voices.find(v => v.lang.startsWith('en'));
    if (english) utterance.voice = english;

    speechSynthesis.speak(utterance);

    // Track usage
    const phrase = this._phrases.find(p => p.sentence === sentence);
    if (phrase) {
      phrase.timesSpoken++;
      this._persist();
    }

    return true;
  }

  /**
   * Check if Web Share API is available.
   * @returns {boolean}
   */
  canShare() {
    return typeof navigator !== 'undefined' && !!navigator.share;
  }

  /**
   * Check if clipboard API is available.
   * @returns {boolean}
   */
  canCopy() {
    return typeof navigator !== 'undefined' &&
      !!(navigator.clipboard?.writeText || document.execCommand);
  }

  // ===========================================================================
  // PRIVATE
  // ===========================================================================

  _load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  _persist() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this._phrases));
    } catch {
      // localStorage full or unavailable — silent fail
    }
  }

  _fallbackCopy(text) {
    try {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.left = '-9999px';
      document.body.appendChild(textarea);
      textarea.select();
      const success = document.execCommand('copy');
      document.body.removeChild(textarea);
      return success;
    } catch {
      return false;
    }
  }
}

export default PhraseBankManager;
