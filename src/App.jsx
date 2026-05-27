import { useState, useMemo, useCallback } from 'react'
import ErrorBoundary from './components/ErrorBoundary'
import WelcomeScreen from './components/WelcomeScreen'
import ModeSelect from './components/ModeSelect'
import AccessibilityPanel from './components/AccessibilityPanel'
import SandboxMode from './components/SandboxMode'
import GestureRecorder from './components/GestureRecorder'
import SessionReport from './components/SessionReport'
import DemoHandout from './components/DemoHandout'
import DemoHandoutBengali from './components/DemoHandoutBengali'
import AboutUs from './components/AboutUs'
import ContactUs from './components/ContactUs'
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
  HANDOUT: 'HANDOUT',
  BENGALI_HANDOUT: 'BENGALI_HANDOUT',
  ABOUT: 'ABOUT',
  CONTACT: 'CONTACT',
};

function App() {
  const [screen, setScreen] = useState(SCREENS.WELCOME);
  const [sessionKey, setSessionKey] = useState(0); // Forces full remount of SandboxMode
  const [profileType, setProfileType] = useState(() => loadProfileSelection());

  // Session data — collected during GUIDED/SANDBOX, shown in REPORT
  const [sessionData, setSessionData] = useState({
    sessionStats: null,
    masteryReport: null,
    automaticitySummary: null,
    knowledgeReport: null,
    sessionNarrative: null,
    learnerModel: null,
    allExplanations: null,
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
      setSessionData(data);
    }
    setScreen(SCREENS.REPORT);
  }, []);

  const handleBackToMenu = useCallback(() => {
    setScreen(SCREENS.MODE_SELECT);
  }, []);

  const handleNewSession = useCallback(() => {
    setSessionData({ sessionStats: null, masteryReport: null, automaticitySummary: null, knowledgeReport: null, sessionNarrative: null, learnerModel: null, allExplanations: null });
    setSessionKey(k => k + 1);
    setScreen(SCREENS.MODE_SELECT);
  }, []);

  const handleBackToWelcome = useCallback(() => {
    setScreen(SCREENS.WELCOME);
  }, []);

  const handleNavigate = useCallback((screen) => {
    switch (screen) {
      case 'HOME':    setScreen(SCREENS.WELCOME); break;
      case 'ABOUT':   setScreen(SCREENS.ABOUT); break;
      case 'CONTACT': setScreen(SCREENS.CONTACT); break;
      case 'HANDOUT': setScreen(SCREENS.HANDOUT); break;
      case 'BENGALI': setScreen(SCREENS.BENGALI_HANDOUT); break;
      default:        setScreen(SCREENS.WELCOME);
    }
  }, []);

  const isActiveScreen = screen !== SCREENS.WELCOME && screen !== SCREENS.HANDOUT && screen !== SCREENS.BENGALI_HANDOUT && screen !== SCREENS.ABOUT && screen !== SCREENS.CONTACT;

  return (
    <ErrorBoundary>
      <UpdatePrompt />
      <div className={`app ${uiAdaptations.highContrast ? 'high-contrast' : ''} ${accessibilityProfile.isLowStimulus() ? 'low-stimulus' : ''}`}>

        {/* Persistent Home button — visible on all active screens */}
        {isActiveScreen && (
          <button
            onClick={handleBackToWelcome}
            aria-label="Return to home screen"
            className="home-nav-btn"
            style={{
              position: 'fixed', top: 14, left: 14, zIndex: 9999,
              padding: '7px 14px 7px 12px',
              borderRadius: 10,
              background: '#FFF6E6',
              border: '2px solid #0F172A',
              color: '#0F172A',
              fontFamily: "'Space Grotesk', 'Inter', system-ui, sans-serif",
              fontSize: '0.82rem', fontWeight: 700,
              cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: '0.35rem',
              boxShadow: '3px 3px 0 0 #FF5252',
              transition: 'transform 0.12s ease, box-shadow 0.12s ease',
            }}
            onMouseEnter={e => { e.currentTarget.style.transform = 'translate(-1px, -1px)'; e.currentTarget.style.boxShadow = '4px 4px 0 0 #FF5252'; }}
            onMouseLeave={e => { e.currentTarget.style.transform = 'translate(0, 0)'; e.currentTarget.style.boxShadow = '3px 3px 0 0 #FF5252'; }}
          >
            <span aria-hidden="true">←</span> Home
          </button>
        )}

        {/* ============== WELCOME ============== */}
        {screen === SCREENS.WELCOME && (
          <WelcomeScreen
            onStart={() => setScreen(SCREENS.PROFILE)}
            onNavigate={handleNavigate}
          />
        )}

        {/* ============== PROFILE SELECTION ============== */}
        {screen === SCREENS.PROFILE && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            minHeight: '100vh',
            padding: '70px 1.5rem 3rem',
            background: '#FFF6E6',
            position: 'relative',
            overflow: 'hidden',
          }}>
            {/* Background splats */}
            <svg viewBox="0 0 120 120" aria-hidden="true" style={{ position: 'absolute', top: 80, right: -30, width: 180, height: 180, opacity: 0.16, transform: 'rotate(20deg)', pointerEvents: 'none' }}>
              <path d="M62 8 C 80 6, 95 22, 92 38 C 105 42, 116 56, 108 72 C 116 88, 100 105, 82 100 C 78 116, 56 118, 48 104 C 30 110, 14 96, 20 78 C 6 70, 4 50, 22 44 C 22 24, 44 12, 62 8 Z" fill="#A855F7"/>
            </svg>
            <svg viewBox="0 0 120 120" aria-hidden="true" style={{ position: 'absolute', bottom: 60, left: -40, width: 160, height: 160, opacity: 0.14, transform: 'rotate(-20deg)', pointerEvents: 'none' }}>
              <path d="M62 8 C 80 6, 95 22, 92 38 C 105 42, 116 56, 108 72 C 116 88, 100 105, 82 100 C 78 116, 56 118, 48 104 C 30 110, 14 96, 20 78 C 6 70, 4 50, 22 44 C 22 24, 44 12, 62 8 Z" fill="#14B8A6"/>
            </svg>

            <div style={{ maxWidth: 760, width: '100%', position: 'relative', zIndex: 1 }}>
              <div style={{
                display: 'inline-block', background: '#FACC15', color: '#0F172A',
                fontFamily: "'Space Grotesk', 'Inter', system-ui, sans-serif",
                fontSize: '0.72rem', fontWeight: 700,
                letterSpacing: '0.1em', textTransform: 'uppercase',
                padding: '6px 12px', border: '2px solid #0F172A', borderRadius: 4,
                marginBottom: '1.2rem',
                boxShadow: '3px 3px 0 0 #0F172A',
              }}>
                Step 1 of 3
              </div>
              <h2 style={{
                fontFamily: "'Bungee', 'Impact', system-ui, sans-serif",
                fontSize: 'clamp(1.8rem, 4.5vw, 2.6rem)',
                color: '#0F172A',
                margin: '0 0 0.6rem',
                letterSpacing: '-0.005em',
                lineHeight: 1.1,
              }}>
                Who's <span style={{ color: '#FF5252' }}>learning?</span>
              </h2>
              <p style={{
                fontFamily: "'Space Grotesk', 'Inter', system-ui, sans-serif",
                fontSize: '0.98rem', color: '#475569',
                marginBottom: '1.8rem',
                maxWidth: 540, lineHeight: 1.55,
              }}>
                Pick the profile that fits the learner. MLAF adjusts hand-tolerance,
                hold-time, and feedback automatically — so you don't have to think
                about it again.
              </p>
            </div>

            <AccessibilityPanel
              currentProfile={profileType}
              onProfileChange={handleProfileChange}
            />

            <button
              onClick={handleProfileDone}
              style={{
                marginTop: '2rem',
                fontFamily: "'Bungee', 'Impact', system-ui, sans-serif",
                fontSize: '1.05rem',
                color: '#FFF6E6',
                background: '#0F172A',
                border: '2px solid #0F172A',
                borderRadius: 12,
                padding: '14px 36px',
                cursor: 'pointer',
                letterSpacing: '0.03em',
                boxShadow: '5px 5px 0 0 #FF5252',
                position: 'relative', zIndex: 1,
              }}
            >
              Continue →
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
            sessionStats={sessionData.sessionStats}
            masteryReport={sessionData.masteryReport}
            automaticitySummary={sessionData.automaticitySummary}
            knowledgeReport={sessionData.knowledgeReport}
            sessionNarrative={sessionData.sessionNarrative}
            learnerModel={sessionData.learnerModel}
            allExplanations={sessionData.allExplanations}
            onNewSession={handleNewSession}
            onBackToMenu={handleBackToMenu}
          />
        )}

        {/* ============== DEMO HANDOUT (ENGLISH) ============== */}
        {screen === SCREENS.HANDOUT && (
          <DemoHandout onBack={handleBackToWelcome} onNavigate={handleNavigate} />
        )}

        {/* ============== DEMO HANDOUT (BENGALI) ============== */}
        {screen === SCREENS.BENGALI_HANDOUT && (
          <DemoHandoutBengali onBack={handleBackToWelcome} onNavigate={handleNavigate} />
        )}

        {/* ============== ABOUT US ============== */}
        {screen === SCREENS.ABOUT && (
          <AboutUs onNavigate={handleNavigate} />
        )}

        {/* ============== CONTACT US ============== */}
        {screen === SCREENS.CONTACT && (
          <ContactUs onNavigate={handleNavigate} />
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
