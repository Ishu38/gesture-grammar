import { useState, useMemo, useCallback, useRef } from 'react'
import ErrorBoundary from './components/ErrorBoundary'
import WelcomeScreen from './components/WelcomeScreen'
import ModeSelect from './components/ModeSelect'
import AccessibilityPanel from './components/AccessibilityPanel'
import SandboxMode from './components/SandboxMode'
import GestureRecorder from './components/GestureRecorder'
import SessionReport from './components/SessionReport'
import { AccessibilityProfile, saveProfileSelection, loadProfileSelection } from './core/AccessibilityProfile'
import UpdatePrompt, { getAppVersion } from './components/UpdatePrompt'
import './App.css'

/**
 * App State Machine:
 *
 *   WELCOME  →  PROFILE  →  MODE_SELECT  →  GUIDED / SANDBOX  →  REPORT
 *      ↑                                                             │
 *      └─────────────────────────────────────────────────────────────┘
 *
 * GUIDED mode sets SandboxMode to start in guided practice by default.
 * SANDBOX mode gives full open-ended access.
 * REPORT shows learning analytics at end of session.
 */

const SCREENS = {
  WELCOME: 'WELCOME',
  PROFILE: 'PROFILE',
  MODE_SELECT: 'MODE_SELECT',
  GUIDED: 'GUIDED',
  SANDBOX: 'SANDBOX',
  RECORDER: 'RECORDER',
  REPORT: 'REPORT',
};

function App() {
  const [screen, setScreen] = useState(SCREENS.WELCOME);
  const [sessionKey, setSessionKey] = useState(0); // Forces full remount of SandboxMode
  const [profileType, setProfileType] = useState(() => loadProfileSelection());

  // Session data refs — collected during GUIDED/SANDBOX, shown in REPORT
  const sessionDataRef = useRef({
    sessionStats: null,
    masteryReport: null,
    automaticitySummary: null,
  });

  const accessibilityProfile = useMemo(
    () => new AccessibilityProfile(profileType),
    [profileType]
  );
  const uiAdaptations = accessibilityProfile.getUIAdaptations();

  const handleProfileChange = useCallback((newType) => {
    setProfileType(newType);
    saveProfileSelection(newType);
  }, []);

  const handleProfileDone = useCallback(() => {
    setScreen(SCREENS.MODE_SELECT);
  }, []);

  const handleEndSession = useCallback((data) => {
    if (data) {
      sessionDataRef.current = data;
    }
    setScreen(SCREENS.REPORT);
  }, []);

  const handleBackToMenu = useCallback(() => {
    setScreen(SCREENS.MODE_SELECT);
  }, []);

  const handleNewSession = useCallback(() => {
    sessionDataRef.current = { sessionStats: null, masteryReport: null, automaticitySummary: null };
    setSessionKey(k => k + 1); // Force full remount of SandboxMode, clearing all async state
    setScreen(SCREENS.MODE_SELECT);
  }, []);

  return (
    <ErrorBoundary>
      <UpdatePrompt />
      <div className={`app ${uiAdaptations.highContrast ? 'high-contrast' : ''} ${accessibilityProfile.isLowStimulus() ? 'low-stimulus' : ''}`}>

        {/* ============== WELCOME ============== */}
        {screen === SCREENS.WELCOME && (
          <WelcomeScreen onStart={() => setScreen(SCREENS.PROFILE)} />
        )}

        {/* ============== PROFILE SELECTION ============== */}
        {screen === SCREENS.PROFILE && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '100vh',
            padding: '2rem',
          }}>
            <h2 style={{
              fontSize: '1.5rem', fontWeight: 700, color: '#e2e8f0',
              marginBottom: '0.5rem',
            }}>
              Accessibility Profile
            </h2>
            <p style={{
              fontSize: '0.85rem', color: '#94a3b8', marginBottom: '2rem',
              textAlign: 'center', maxWidth: '400px',
            }}>
              Select the profile that best matches the learner.
              This adjusts error tolerance, confidence thresholds, and feedback intensity.
            </p>
            <AccessibilityPanel
              currentProfile={profileType}
              onProfileChange={handleProfileChange}
            />
            <button
              onClick={handleProfileDone}
              style={{
                marginTop: '2rem',
                padding: '0.85rem 2.5rem',
                fontSize: '1rem',
                fontWeight: 700,
                color: '#0f0f1a',
                background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
                border: 'none',
                borderRadius: '10px',
                cursor: 'pointer',
                boxShadow: '0 4px 20px rgba(74, 222, 128, 0.2)',
              }}
            >
              Continue
            </button>
          </div>
        )}

        {/* ============== MODE SELECT ============== */}
        {screen === SCREENS.MODE_SELECT && (
          <ModeSelect
            profileType={profileType}
            onSelectGuided={() => setScreen(SCREENS.GUIDED)}
            onSelectSandbox={() => setScreen(SCREENS.SANDBOX)}
            onSelectRecorder={() => setScreen(SCREENS.RECORDER)}
          />
        )}

        {/* ============== GUIDED PRACTICE ============== */}
        {screen === SCREENS.GUIDED && (
          <>
            <header className="app-header">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
                <h1>MLAF</h1>
                <span style={{
                  background: '#4ade80', color: '#0f0f1a',
                  padding: '2px 8px', borderRadius: 4,
                  fontSize: '0.6rem', fontWeight: 800, letterSpacing: '0.1em',
                }}>
                  GUIDED
                </span>
              </div>
              <p className="subtitle">Multimodal Language Acquisition Framework</p>
            </header>
            <ErrorBoundary>
              <SandboxMode
                key={`guided-${sessionKey}`}
                accessibilityProfile={accessibilityProfile}
                initialMode="guided"
                onEndSession={handleEndSession}
              />
            </ErrorBoundary>
          </>
        )}

        {/* ============== SANDBOX (ADVANCED) ============== */}
        {screen === SCREENS.SANDBOX && (
          <>
            <header className="app-header">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
                <h1>MLAF</h1>
                <span style={{
                  background: 'rgba(255,255,255,0.1)', color: '#94a3b8',
                  padding: '2px 8px', borderRadius: 4,
                  fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.1em',
                }}>
                  SANDBOX
                </span>
              </div>
              <p className="subtitle">Multimodal Language Acquisition Framework</p>
            </header>
            <ErrorBoundary>
              <SandboxMode
                key={`sandbox-${sessionKey}`}
                accessibilityProfile={accessibilityProfile}
                initialMode="sandbox"
                onEndSession={handleEndSession}
              />
            </ErrorBoundary>
          </>
        )}

        {/* ============== GESTURE RECORDER ============== */}
        {screen === SCREENS.RECORDER && (
          <>
            <header className="app-header">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
                <h1>MLAF</h1>
                <span style={{
                  background: '#fbbf24', color: '#0f0f1a',
                  padding: '2px 8px', borderRadius: 4,
                  fontSize: '0.6rem', fontWeight: 800, letterSpacing: '0.1em',
                }}>
                  RECORDER
                </span>
              </div>
              <p className="subtitle">Capture Training Data</p>
            </header>
            <ErrorBoundary>
              <GestureRecorder onBack={handleBackToMenu} />
            </ErrorBoundary>
          </>
        )}

        {/* ============== SESSION REPORT ============== */}
        {screen === SCREENS.REPORT && (
          <SessionReport
            sessionStats={sessionDataRef.current.sessionStats}
            masteryReport={sessionDataRef.current.masteryReport}
            automaticitySummary={sessionDataRef.current.automaticitySummary}
            onNewSession={handleNewSession}
            onBackToMenu={handleBackToMenu}
          />
        )}

        {/* Footer — only show in active modes */}
        {(screen === SCREENS.GUIDED || screen === SCREENS.SANDBOX || screen === SCREENS.RECORDER) && (
          <footer className="app-footer">
            Designed &amp; Created by Neil Shankar Ray
            <span style={{
              marginLeft: 8,
              fontSize: '0.6rem',
              color: '#475569',
              fontWeight: 600,
            }}>
              v{getAppVersion()}
            </span>
          </footer>
        )}

        {/* Always-visible patent notice — legal */}
        <div
          role="contentinfo"
          aria-label="Patent notice"
          style={{
            position: 'fixed',
            bottom: 0,
            left: 0,
            right: 0,
            padding: '6px 12px',
            background: 'rgba(15, 15, 26, 0.92)',
            borderTop: '1px solid rgba(148, 163, 184, 0.15)',
            color: '#64748b',
            fontSize: '0.65rem',
            textAlign: 'center',
            lineHeight: 1.4,
            zIndex: 1000,
            backdropFilter: 'blur(6px)',
            WebkitBackdropFilter: 'blur(6px)',
            pointerEvents: 'none',
          }}
        >
          Patent-Pending — Provisional application TEMP/E-1/22951/2026-KOL, Indian Patent Office (2026). All rights reserved.
        </div>
      </div>
    </ErrorBoundary>
  )
}

export default App
