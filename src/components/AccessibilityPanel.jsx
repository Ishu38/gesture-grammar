/**
 * AccessibilityPanel.jsx — Profile selector and adaptive settings
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Allows users to select their accessibility profile, which auto-adjusts
 * all gesture thresholds, feedback modes, and UI adaptations.
 */

import { ACCESSIBILITY_PROFILES } from '../core/AccessibilityProfile';

function AccessibilityPanel({ currentProfile, onProfileChange }) {
  const profiles = Object.values(ACCESSIBILITY_PROFILES);

  return (
    <div className="accessibility-panel" role="region" aria-label="Accessibility Settings">
      <div className="accessibility-header">
        <h3 className="accessibility-title">Accessibility</h3>
      </div>

      <div className="profile-selector">
        {profiles.map(profile => (
          <button
            key={profile.id}
            className={`profile-option ${currentProfile === profile.id ? 'active' : ''}`}
            onClick={() => onProfileChange(profile.id)}
            aria-pressed={currentProfile === profile.id}
            aria-label={`${profile.label}: ${profile.description}`}
          >
            <span className="profile-icon">{getProfileIcon(profile.id)}</span>
            <div className="profile-info">
              <span className="profile-label">{profile.label}</span>
              <span className="profile-desc">{profile.description}</span>
            </div>
            {currentProfile === profile.id && (
              <span className="profile-check" aria-hidden="true">&#10003;</span>
            )}
          </button>
        ))}
      </div>

      {currentProfile !== 'default' && (
        <div className="profile-active-info" role="status" aria-live="polite">
          <span className="active-badge">Active</span>
          <span className="active-detail">
            {getActiveDescription(currentProfile)}
          </span>
        </div>
      )}
    </div>
  );
}

function getProfileIcon(profileId) {
  switch (profileId) {
    case 'speech-impaired': return '\u{1F5E3}';
    case 'hearing-impaired': return '\u{1F442}';
    case 'motor-impaired': return '\u{270B}';
    case 'cp-spastic': return '\u{1F9BE}';
    case 'cp-athetoid': return '\u{1F9BE}';
    case 'cp-ataxic': return '\u{1F9BE}';
    case 'cp-mixed': return '\u{1F9BE}';
    case 'asd-low-stimulus': return '\u{1F9E9}';
    case 'asd-structured': return '\u{1F9E9}';
    case 'low-vision': return '\u{1F441}';
    case 'cochlear-implant': return '\u{1F9BB}';
    case 'eye-gaze-aac': return '\u{1F440}';
    default: return '\u{1F464}';
  }
}

function getActiveDescription(profileId) {
  const profile = ACCESSIBILITY_PROFILES[profileId];
  if (!profile) return '';

  const details = [];
  if (profile.toleranceMultiplier > 1) details.push(`${profile.toleranceMultiplier}x gesture tolerance`);
  if (profile.confidenceFrames > 45) details.push(`${profile.confidenceFrames} frame hold time`);
  if (profile.outputMode === 'text-to-speech') details.push('Text-to-speech enabled');
  if (profile.uiAdaptations?.noAudioFeedback) details.push('Audio suppressed');
  if (profile.uiAdaptations?.highContrast) details.push('High contrast mode');
  if (profile.uiAdaptations?.audioCorrectionFeedback) details.push('Audio corrections');
  if (profile.uiAdaptations?.hapticFeedback) details.push('Haptic vibration');
  if (profile.noiseFloorCalibration) details.push('Noise-adaptive');
  if (profile.signLanguageBridge) details.push('ISL bridge');
  if (profile.lowStimulus) details.push('Low stimulus');
  if (profile.perseverationDetection) details.push('Perseveration guard');

  return details.join(' | ');
}

export default AccessibilityPanel;
