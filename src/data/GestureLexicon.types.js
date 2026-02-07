/**
 * GestureLexicon Type Definitions
 *
 * These JSDoc types define the schema for the Gesture Lexicon.
 * Use these types when implementing the gesture detection system.
 */

/**
 * @typedef {'<' | '>' | 'distance <' | 'distance >' | 'velocity >' | '~='} ConstraintOperator
 */

/**
 * @typedef {'AND' | 'OR'} LogicalJoin
 */

/**
 * @typedef {'STATIC' | 'DYNAMIC'} GestureType
 */

/**
 * @typedef {'SUBJECT' | 'VERB' | 'OBJECT' | 'MODIFIER' | 'GRAMMATICAL_MARKER' | 'COMMAND'} SemanticType
 */

/**
 * @typedef {Object} LandmarkCondition
 * @property {string} landmark_a - First landmark reference (e.g., "index_tip_y")
 * @property {ConstraintOperator} operator - Comparison operator
 * @property {string} [landmark_b] - Second landmark reference for comparison
 * @property {number} [threshold] - Numeric threshold for distance/velocity operators
 * @property {number} [tolerance] - Tolerance for approximate equality (~=)
 * @property {'positive_x' | 'negative_x' | 'positive_y' | 'negative_y'} [direction] - For velocity checks
 * @property {number} [time_window_ms] - Time window for dynamic gestures
 */

/**
 * @typedef {Object} TemporalRequirements
 * @property {number} min_duration_ms - Minimum gesture duration
 * @property {number} max_duration_ms - Maximum gesture duration
 */

/**
 * @typedef {Object} LandmarkConstraints
 * @property {GestureType} type - Whether gesture is static or dynamic
 * @property {Object.<string, LandmarkCondition>} conditions - Named conditions to check
 * @property {LogicalJoin} logical_join - How to combine conditions
 * @property {TemporalRequirements} [temporal_requirements] - For dynamic gestures
 */

/**
 * @typedef {Object} GrammaticalProperties
 * @property {1 | 2 | 3} [person] - Grammatical person
 * @property {'singular' | 'plural'} [number] - Grammatical number
 * @property {'transitive' | 'intransitive'} [transitivity] - Verb transitivity
 * @property {boolean} [requires_object] - Whether verb needs object
 * @property {'active' | 'passive'} [voice] - Voice marker
 * @property {'subject' | 'object'} [agent_position] - Agent position in sentence
 * @property {'suffix' | 'prefix'} [morphological_role] - Morphological role
 * @property {'concrete' | 'abstract'} [noun_type] - Noun classification
 * @property {boolean} [countable] - Whether noun is countable
 */

/**
 * @typedef {Object} SemanticMapping
 * @property {string} grammar_id - ID used in GrammarEngine LEXICON
 * @property {string} concept - Abstract linguistic concept
 * @property {SemanticType} type - Grammatical category
 * @property {string} english_gloss - Human-readable translation
 * @property {string} usage_context - When/how to use this gesture
 * @property {GrammaticalProperties} [grammatical_properties] - Detailed grammar info
 */

/**
 * @typedef {Object} GestureDefinition
 * @property {string} gesture_id - Unique identifier (e.g., "GST_FIST")
 * @property {string} human_description - Plain English description
 * @property {LandmarkConstraints} landmark_constraints - Detection constraints
 * @property {SemanticMapping} semantic_mapping - Grammatical meaning
 * @property {number} priority - Detection priority (lower = higher priority)
 * @property {string} [notes] - Additional notes
 */

/**
 * @typedef {Object} GestureLexicon
 * @property {string} $schema - Schema version identifier
 * @property {Object} meta - Metadata about the lexicon
 * @property {GestureDefinition[]} gestures - Array of gesture definitions
 * @property {Object} type_hierarchy - Grouping of grammar IDs by type
 * @property {Object} detection_notes - Implementation notes
 */

// Landmark index constants for reference
export const LANDMARK_INDICES = {
  WRIST: 0,

  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,

  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,

  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,

  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,

  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
};

// Semantic type constants
export const SEMANTIC_TYPES = {
  SUBJECT: 'SUBJECT',
  VERB: 'VERB',
  OBJECT: 'OBJECT',
  MODIFIER: 'MODIFIER',
  GRAMMATICAL_MARKER: 'GRAMMATICAL_MARKER',
  COMMAND: 'COMMAND',
};

// Gesture type constants
export const GESTURE_TYPES = {
  STATIC: 'STATIC',
  DYNAMIC: 'DYNAMIC',
};

/**
 * Maps gesture_id to grammar_id for quick lookup
 */
export const GESTURE_TO_GRAMMAR = {
  'GST_OPEN_PALM': 'STOP',
  'GST_FIST': 'SUBJECT_I',
  'GST_TWO_FINGERS': 'PLURAL',
  'GST_PINCH': 'GRAB',
  'GST_SWIPE_RIGHT': 'ACTIVE_VOICE',
  'GST_SWIPE_LEFT': 'PASSIVE_VOICE',
  'GST_POINT': 'SUBJECT_YOU',
  'GST_THREE_FINGERS': 'SUBJECT_HE',
  'GST_THUMBS_UP': 'AFFIRMATIVE',
  'GST_FLAT_HAND_DOWN': 'OBJECT_APPLE',
};
