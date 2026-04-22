/**
 * GazeDwellSelector.js — Eye-Gaze Dwell-Click Token Selection for AAC
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Provides an alternative input modality for children who cannot form hand
 * gestures (e.g., severe CP, Rett syndrome, SCI). Uses the existing
 * EyeGazeTracker's gaze direction (gazeX, gazeY) to select grammar tokens
 * from an on-screen grid via sustained fixation (dwell-click).
 *
 * Screen layout for gaze selection (3x3 grid):
 *
 *   ┌─────────────┬─────────────┬─────────────┐
 *   │  SUBJECT_I  │ SUBJECT_YOU │ SUBJECT_HE  │  ← Row 0: Subjects
 *   ├─────────────┼─────────────┼─────────────┤
 *   │    GRAB     │     EAT     │    WANT     │  ← Row 1: Verbs
 *   ├─────────────┼─────────────┼─────────────┤
 *   │    APPLE    │    WATER    │    FOOD     │  ← Row 2: Objects
 *   └─────────────┴─────────────┴─────────────┘
 *
 * Pagination: LEFT/RIGHT arrows to scroll through tokens in each row.
 *
 * Dwell mechanism:
 *   1. Gaze enters a cell → start dwell timer
 *   2. Gaze stays in same cell for DWELL_FRAMES → trigger selection
 *   3. After selection → cooldown period (prevents double-select)
 *   4. Gaze moves to different cell → reset timer
 *
 * Visual feedback:
 *   - Progress ring around dwelled cell (fills as dwell progresses)
 *   - Selected cell flashes briefly
 *   - Audio chime on selection (if audio enabled)
 */

import { LEXICON } from '../utils/GrammarEngine.js';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Frames of sustained fixation to trigger selection. At 30fps ≈ 1.5 seconds. */
const DEFAULT_DWELL_FRAMES = 45;

/** Cooldown frames after a selection before another can be made. */
const COOLDOWN_FRAMES = 30;

/** Grid layout configuration. */
const GRID_ROWS = 3;
const GRID_COLS = 3;

// =============================================================================
// TOKEN PAGES (paginated vocabulary per row)
// =============================================================================

/**
 * Build the token pages from the LEXICON.
 * Each row category is split into pages of GRID_COLS items.
 */
function buildTokenPages() {
  const subjects = [];
  const verbs = [];
  const objects = [];

  for (const [id, entry] of Object.entries(LEXICON)) {
    if (entry.requires_s_form) continue; // skip S-forms
    if (entry.type === 'SUBJECT') subjects.push(id);
    else if (entry.type === 'VERB') verbs.push(id);
    else if (entry.type === 'OBJECT') objects.push(id);
  }

  const paginate = (arr) => {
    const pages = [];
    for (let i = 0; i < arr.length; i += GRID_COLS) {
      pages.push(arr.slice(i, i + GRID_COLS));
    }
    return pages.length > 0 ? pages : [[]];
  };

  return {
    subjects: paginate(subjects),
    verbs: paginate(verbs),
    objects: paginate(objects),
  };
}

// =============================================================================
// GAZE DWELL SELECTOR
// =============================================================================

export class GazeDwellSelector {
  /**
   * @param {object} [config]
   * @param {number} [config.dwellFrames=45] — frames to dwell before selection
   * @param {function} [config.onTokenSelected] — callback: (grammarId, cellInfo) => void
   * @param {function} [config.onDwellProgress] — callback: (cellKey, progress) => void
   * @param {function} [config.onPageChange] — callback: (row, pageIndex) => void
   */
  constructor(config = {}) {
    this._dwellFrames = config.dwellFrames || DEFAULT_DWELL_FRAMES;
    this._onTokenSelected = config.onTokenSelected || null;
    this._onDwellProgress = config.onDwellProgress || null;
    this._onPageChange = config.onPageChange || null;

    // Token pages
    this._tokenPages = buildTokenPages();
    this._pageIndices = { subjects: 0, verbs: 0, objects: 0 };

    // Dwell state
    this._currentCell = null;     // { row, col } or null
    this._dwellCount = 0;
    this._cooldownCount = 0;

    // Selection history for undo
    this._selectionHistory = [];

    // Active state
    this._isActive = false;
  }

  // ===========================================================================
  // PUBLIC — Lifecycle
  // ===========================================================================

  /**
   * Activate gaze-dwell input mode.
   */
  activate() {
    this._isActive = true;
    this._resetDwell();
  }

  /**
   * Deactivate gaze-dwell input mode.
   */
  deactivate() {
    this._isActive = false;
    this._resetDwell();
  }

  /**
   * Check if active.
   * @returns {boolean}
   */
  isActive() {
    return this._isActive;
  }

  // ===========================================================================
  // PUBLIC — Per-Frame Processing
  // ===========================================================================

  /**
   * Process one gaze frame. Call every frame when gaze-dwell mode is active.
   *
   * @param {object} gazeResult — from EyeGazeTracker.processFrame()
   *   Expected: { features: { gazeX, gazeY, fixationDuration }, gaze_state, face_detected }
   * @returns {object} { cell, dwellProgress, selected, selectedToken }
   */
  processFrame(gazeResult) {
    if (!this._isActive) {
      return { cell: null, dwellProgress: 0, selected: false, selectedToken: null };
    }

    if (!gazeResult || !gazeResult.face_detected) {
      this._resetDwell();
      return { cell: null, dwellProgress: 0, selected: false, selectedToken: null };
    }

    // Cooldown after selection
    if (this._cooldownCount > 0) {
      this._cooldownCount--;
      return {
        cell: this._currentCell,
        dwellProgress: 0,
        selected: false,
        selectedToken: null,
        cooldown: true,
      };
    }

    const { gazeX, gazeY } = gazeResult.features;

    // Map gaze position to grid cell
    const cell = this._gazeToCell(gazeX, gazeY);

    if (!cell) {
      this._resetDwell();
      return { cell: null, dwellProgress: 0, selected: false, selectedToken: null };
    }

    // Check if gaze is still on the same cell
    const sameCell = this._currentCell &&
      this._currentCell.row === cell.row &&
      this._currentCell.col === cell.col;

    if (sameCell) {
      this._dwellCount++;
    } else {
      // Gaze moved to new cell — reset
      this._currentCell = cell;
      this._dwellCount = 1;
    }

    const progress = Math.min(1, this._dwellCount / this._dwellFrames);

    // Emit dwell progress
    if (this._onDwellProgress) {
      const cellKey = `${cell.row}_${cell.col}`;
      try { this._onDwellProgress(cellKey, progress); } catch (_) {}
    }

    // Check if dwell threshold reached
    if (this._dwellCount >= this._dwellFrames) {
      const token = this._getCellToken(cell.row, cell.col);
      this._cooldownCount = COOLDOWN_FRAMES;
      this._dwellCount = 0;

      if (token) {
        this._selectionHistory.push(token);

        if (this._onTokenSelected) {
          try {
            this._onTokenSelected(token, {
              row: cell.row,
              col: cell.col,
              display: LEXICON[token]?.display || token,
              type: LEXICON[token]?.type || 'UNKNOWN',
            });
          } catch (_) {}
        }
      }

      return {
        cell,
        dwellProgress: 1,
        selected: true,
        selectedToken: token,
      };
    }

    return {
      cell,
      dwellProgress: progress,
      selected: false,
      selectedToken: null,
    };
  }

  // ===========================================================================
  // PUBLIC — Navigation
  // ===========================================================================

  /**
   * Navigate to next page for a row category.
   * @param {string} category — 'subjects', 'verbs', or 'objects'
   */
  nextPage(category) {
    const pages = this._tokenPages[category];
    if (!pages) return;
    this._pageIndices[category] = (this._pageIndices[category] + 1) % pages.length;
    if (this._onPageChange) {
      try { this._onPageChange(category, this._pageIndices[category]); } catch (_) {}
    }
  }

  /**
   * Navigate to previous page for a row category.
   * @param {string} category — 'subjects', 'verbs', or 'objects'
   */
  prevPage(category) {
    const pages = this._tokenPages[category];
    if (!pages) return;
    this._pageIndices[category] =
      (this._pageIndices[category] - 1 + pages.length) % pages.length;
    if (this._onPageChange) {
      try { this._onPageChange(category, this._pageIndices[category]); } catch (_) {}
    }
  }

  /**
   * Get the current grid layout (what tokens are visible in each cell).
   * @returns {Array<Array<{token: string, display: string, type: string}|null>>}
   */
  getCurrentGrid() {
    const categories = ['subjects', 'verbs', 'objects'];
    const grid = [];

    for (let row = 0; row < GRID_ROWS; row++) {
      const cat = categories[row];
      const pageIdx = this._pageIndices[cat];
      const page = this._tokenPages[cat][pageIdx] || [];
      const rowCells = [];

      for (let col = 0; col < GRID_COLS; col++) {
        const token = page[col] || null;
        if (token) {
          rowCells.push({
            token,
            display: LEXICON[token]?.display || token,
            type: LEXICON[token]?.type || 'UNKNOWN',
          });
        } else {
          rowCells.push(null);
        }
      }
      grid.push(rowCells);
    }

    return grid;
  }

  /**
   * Get pagination info for each row.
   * @returns {object}
   */
  getPaginationInfo() {
    return {
      subjects: {
        page: this._pageIndices.subjects,
        totalPages: this._tokenPages.subjects.length,
        hasNext: this._tokenPages.subjects.length > 1,
      },
      verbs: {
        page: this._pageIndices.verbs,
        totalPages: this._tokenPages.verbs.length,
        hasNext: this._tokenPages.verbs.length > 1,
      },
      objects: {
        page: this._pageIndices.objects,
        totalPages: this._tokenPages.objects.length,
        hasNext: this._tokenPages.objects.length > 1,
      },
    };
  }

  /**
   * Get selection history.
   * @returns {string[]}
   */
  getSelectionHistory() {
    return [...this._selectionHistory];
  }

  /**
   * Clear selection history.
   */
  clearHistory() {
    this._selectionHistory = [];
  }

  /**
   * Update dwell threshold at runtime (accessibility adjustment).
   * @param {number} frames
   */
  setDwellFrames(frames) {
    this._dwellFrames = Math.max(15, Math.min(120, frames));
  }

  /**
   * Get dwell configuration.
   * @returns {object}
   */
  getConfig() {
    return {
      dwellFrames: this._dwellFrames,
      cooldownFrames: COOLDOWN_FRAMES,
      gridRows: GRID_ROWS,
      gridCols: GRID_COLS,
      isActive: this._isActive,
    };
  }

  // ===========================================================================
  // PRIVATE
  // ===========================================================================

  /**
   * Map gazeX/gazeY (0-1 range) to a grid cell.
   * @param {number} gazeX — 0=looking left, 1=looking right
   * @param {number} gazeY — 0=looking up, 1=looking down
   * @returns {{ row: number, col: number }|null}
   */
  _gazeToCell(gazeX, gazeY) {
    // Clamp to valid range
    const x = Math.max(0, Math.min(1, gazeX));
    const y = Math.max(0, Math.min(1, gazeY));

    // Map to grid cell
    const col = Math.min(GRID_COLS - 1, Math.floor(x * GRID_COLS));
    const row = Math.min(GRID_ROWS - 1, Math.floor(y * GRID_ROWS));

    return { row, col };
  }

  /**
   * Get the grammar token for a specific cell.
   * @param {number} row
   * @param {number} col
   * @returns {string|null} grammar_id or null
   */
  _getCellToken(row, col) {
    const categories = ['subjects', 'verbs', 'objects'];
    const cat = categories[row];
    if (!cat) return null;

    const pageIdx = this._pageIndices[cat];
    const page = this._tokenPages[cat][pageIdx];
    if (!page) return null;

    return page[col] || null;
  }

  _resetDwell() {
    this._currentCell = null;
    this._dwellCount = 0;
  }
}

export default GazeDwellSelector;
