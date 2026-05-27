/**
 * SessionReport.jsx — End-of-Session Summary
 * Shows learning analytics collected by SessionDataLogger, AutomaticityTracker,
 * and GestureMasteryGate during the session. This is what funders want to see:
 * measurable learning outcomes.
 */

function StatCard({ label, value, sub, color = '#e2e8f0' }) {
  return (
    <div style={{
      background: 'rgba(255,255,255,0.03)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: '12px',
      padding: '1.25rem',
      textAlign: 'center',
    }}>
      <div style={{ fontSize: '2rem', fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.25rem' }}>{label}</div>
      {sub && <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '0.25rem' }}>{sub}</div>}
    </div>
  );
}

function ProgressBar({ label, percent, color = '#4ade80' }) {
  return (
    <div style={{ marginBottom: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{label}</span>
        <span style={{ fontSize: '0.75rem', color }}>{percent}%</span>
      </div>
      <div style={{ height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          width: `${Math.min(100, percent)}%`,
          height: '100%',
          background: color,
          borderRadius: 3,
          transition: 'width 0.8s ease',
        }} />
      </div>
    </div>
  );
}

export default function SessionReport({
  sessionStats,
  masteryReport,
  automaticitySummary,
  knowledgeReport,
  sessionNarrative,
  learnerModel,
  allExplanations,
  onNewSession,
  onBackToMenu,
}) {
  const stats = sessionStats || {};
  const mastery = masteryReport || {};
  const auto = automaticitySummary || {};

  const totalGestures = stats.total_gesture_locks || 0;
  const totalSentences = stats.total_sentences_completed || 0;
  const accuracy = stats.accuracy != null ? Math.round(stats.accuracy * 100) : 0;
  const islErrors = stats.isl_interference_count || 0;
  const sessionDuration = stats.duration_ms
    ? Math.round(stats.duration_ms / 60000)
    : 0;

  // Mastery progress
  const currentStage = mastery.currentStage || 1;
  const highestMastered = mastery.highestMastered || 0;
  const stageProgress = mastery.stages
    ? mastery.stages.find(s => s.stage === currentStage)
    : null;
  const stagePercent = stageProgress && stageProgress.gestureStatuses.length > 0
    ? Math.round(
        (stageProgress.gestureStatuses.filter(g => g.mastered).length /
          stageProgress.gestureStatuses.length) * 100
      )
    : 0;

  // Cognitive load distribution
  const loadDist = stats.cognitive_load_distribution || {};

  return (
    <div role="main" aria-label="Session Report" style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      minHeight: '100vh',
      padding: '2rem',
      background: '#05070d',
      fontFamily: 'system-ui, -apple-system, sans-serif',
    }}>
      {/* Header */}
      <h2 style={{
        fontSize: '1.75rem',
        fontWeight: 800,
        color: '#e2e8f0',
        marginBottom: '0.25rem',
      }}>
        Session Complete
      </h2>
      <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '2rem' }}>
        {sessionDuration > 0 ? `${sessionDuration} minute${sessionDuration !== 1 ? 's' : ''} of practice` : 'Session summary'}
      </p>

      {/* Top-level stats */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '1rem',
        maxWidth: '640px',
        width: '100%',
        marginBottom: '2rem',
      }}>
        <StatCard label="Gestures" value={totalGestures} color="#4ade80" />
        <StatCard label="Sentences" value={totalSentences} color="#60a5fa" />
        <StatCard label="Accuracy" value={`${accuracy}%`} color={accuracy >= 80 ? '#4ade80' : accuracy >= 50 ? '#fbbf24' : '#f87171'} />
        <StatCard label="ISL Errors" value={islErrors} sub="Transfer detected" color={islErrors === 0 ? '#4ade80' : '#fbbf24'} />
      </div>

      {/* Two-column detail */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '1.5rem',
        maxWidth: '640px',
        width: '100%',
        marginBottom: '2rem',
      }}>
        {/* Mastery Progress */}
        <div style={{
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '12px',
          padding: '1.25rem',
        }}>
          <h3 style={{ color: '#e2e8f0', fontSize: '0.9rem', fontWeight: 700, margin: '0 0 1rem' }}>
            Curriculum Progress
          </h3>
          <div style={{
            fontSize: '0.75rem', color: '#60a5fa', marginBottom: '1rem',
            padding: '4px 8px', background: 'rgba(96,165,250,0.1)', borderRadius: 6,
            display: 'inline-block',
          }}>
            Stage {currentStage} of 5
          </div>

          {stageProgress && stageProgress.gestureStatuses.map(g => (
            <ProgressBar
              key={g.id}
              label={g.id.replace(/_/g, ' ')}
              percent={Math.round(g.progress * 100)}
              color={g.mastered ? '#4ade80' : '#60a5fa'}
            />
          ))}

          {highestMastered > 0 && (
            <div style={{ fontSize: '0.7rem', color: '#4ade80', marginTop: '0.5rem' }}>
              Stages 1{highestMastered > 1 ? `–${highestMastered}` : ''} fully mastered
            </div>
          )}
        </div>

        {/* Cognitive Load & Automaticity */}
        <div style={{
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '12px',
          padding: '1.25rem',
        }}>
          <h3 style={{ color: '#e2e8f0', fontSize: '0.9rem', fontWeight: 700, margin: '0 0 1rem' }}>
            Cognitive & Motor Analysis
          </h3>

          {/* Load distribution bar */}
          <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
            Cognitive Load Distribution
          </div>
          <div style={{
            display: 'flex', height: 24, borderRadius: 6, overflow: 'hidden',
            marginBottom: '1rem', border: '1px solid rgba(255,255,255,0.06)',
          }}>
            <div style={{
              width: `${loadDist.LOW ?? 0}%`,
              background: '#4ade80',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.6rem', color: '#0f0f1a', fontWeight: 700,
            }}>
              {(loadDist.LOW || 0) > 15 ? `LOW ${loadDist.LOW}%` : ''}
            </div>
            <div style={{
              width: `${loadDist.MEDIUM || 0}%`,
              background: '#fbbf24',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.6rem', color: '#0f0f1a', fontWeight: 700,
            }}>
              {(loadDist.MEDIUM || 0) > 15 ? `MED ${loadDist.MEDIUM}%` : ''}
            </div>
            <div style={{
              width: `${loadDist.HIGH || 0}%`,
              background: '#f87171',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.6rem', color: '#0f0f1a', fontWeight: 700,
            }}>
              {(loadDist.HIGH || 0) > 15 ? `HIGH ${loadDist.HIGH}%` : ''}
            </div>
          </div>

          {/* Automaticity summary */}
          <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.5rem' }}>
            Gesture Fluency
          </div>
          {auto.gestures && Object.entries(auto.gestures).length > 0 ? (
            Object.entries(auto.gestures).slice(0, 5).map(([gId, data]) => (
              <div key={gId} style={{
                display: 'flex', justifyContent: 'space-between',
                fontSize: '0.7rem', padding: '3px 0',
                borderBottom: '1px solid rgba(255,255,255,0.04)',
              }}>
                <span style={{ color: '#94a3b8' }}>{gId.replace(/_/g, ' ')}</span>
                <span style={{
                  color: data.trend === 'IMPROVING' ? '#4ade80' :
                    data.trend === 'DECLINING' ? '#f87171' : '#94a3b8',
                  fontWeight: 600,
                }}>
                  {data.score != null ? `${(data.score * 100).toFixed(0)}%` : '--'}
                  {data.trend === 'IMPROVING' ? ' ^' : data.trend === 'DECLINING' ? ' v' : ''}
                </span>
              </div>
            ))
          ) : (
            <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
              Not enough data yet. Complete more gestures to see fluency trends.
            </div>
          )}
        </div>
      </div>

      {/* Learning Insights — from ExplainabilityEngine */}
      {sessionNarrative && (
        <div style={{
          maxWidth: '640px',
          width: '100%',
          marginBottom: '2rem',
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '12px',
          padding: '1.5rem',
        }}>
          <h3 style={{ color: '#e2e8f0', fontSize: '0.9rem', fontWeight: 700, margin: '0 0 1rem' }}>
            Learning Insights
          </h3>
          {sessionNarrative.headline && (
            <p style={{ color: '#cbd5e1', fontSize: '0.8rem', lineHeight: 1.5, margin: '0 0 1rem' }}>
              {sessionNarrative.headline}
            </p>
          )}

          {/* Strengths & Challenges */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <div>
              <div style={{ color: '#4ade80', fontSize: '0.7rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                Strengths
              </div>
              {(sessionNarrative.strengths || []).slice(0, 4).map((s, i) => (
                <div key={i} style={{ color: '#94a3b8', fontSize: '0.65rem', padding: '2px 0', lineHeight: 1.4 }}>
                  + {s}
                </div>
              ))}
            </div>
            <div>
              <div style={{ color: '#fbbf24', fontSize: '0.7rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                Areas for Growth
              </div>
              {(sessionNarrative.challenges || []).slice(0, 4).map((c, i) => (
                <div key={i} style={{ color: '#94a3b8', fontSize: '0.65rem', padding: '2px 0', lineHeight: 1.4 }}>
                  - {c}
                </div>
              ))}
            </div>
          </div>

          {/* ISL Transfer Summary */}
          {sessionNarrative.islSummary && (
            <div style={{
              fontSize: '0.65rem', color: '#c084fc', background: 'rgba(168, 85, 247, 0.08)',
              borderRadius: 6, padding: '8px 12px', marginBottom: '0.75rem', lineHeight: 1.4,
            }}>
              {sessionNarrative.islSummary}
            </div>
          )}

          {/* Recommendations */}
          {(sessionNarrative.recommendations || []).length > 0 && (
            <>
              <div style={{ color: '#60a5fa', fontSize: '0.7rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                Recommendations
              </div>
              {sessionNarrative.recommendations.map((r, i) => (
                <div key={i} style={{
                  display: 'flex', gap: '0.5rem', fontSize: '0.65rem',
                  padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.04)',
                }}>
                  <span style={{
                    fontSize: '0.55rem', fontWeight: 700, padding: '1px 6px', borderRadius: 3,
                    background: r.priority === 'HIGH' ? 'rgba(248, 113, 113, 0.2)' : 'rgba(251, 191, 36, 0.15)',
                    color: r.priority === 'HIGH' ? '#f87171' : '#fbbf24',
                    flexShrink: 0,
                  }}>
                    {r.priority}
                  </span>
                  <div>
                    <div style={{ color: '#cbd5e1' }}>{r.action}</div>
                    <div style={{ color: '#64748b', marginTop: 2 }}>{r.rationale}</div>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      {/* Knowledge State — from BKT */}
      {knowledgeReport && (
        <div style={{
          maxWidth: '640px',
          width: '100%',
          marginBottom: '2rem',
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '12px',
          padding: '1.5rem',
        }}>
          <h3 style={{ color: '#e2e8f0', fontSize: '0.9rem', fontWeight: 700, margin: '0 0 0.75rem' }}>
            Knowledge State (BKT)
          </h3>
          <div style={{
            display: 'flex', gap: '0.75rem', marginBottom: '1rem', flexWrap: 'wrap',
          }}>
            <span style={{
              fontSize: '0.65rem', color: '#22d3ee',
              background: 'rgba(34, 211, 238, 0.1)', borderRadius: 4, padding: '3px 8px',
            }}>
              Avg mastery: {Math.round(knowledgeReport.averagePKnown * 100)}%
            </span>
            <span style={{
              fontSize: '0.65rem', color: '#4ade80',
              background: 'rgba(74, 222, 128, 0.1)', borderRadius: 4, padding: '3px 8px',
            }}>
              {knowledgeReport.masteryCount}/{knowledgeReport.totalConcepts} concepts mastered
            </span>
          </div>

          {/* Concept progress bars */}
          {knowledgeReport.weakestConcepts && knowledgeReport.weakestConcepts.length > 0 && (
            <div style={{ marginBottom: '0.75rem' }}>
              <div style={{ color: '#94a3b8', fontSize: '0.65rem', marginBottom: '0.5rem' }}>
                Developing Concepts
              </div>
              {knowledgeReport.weakestConcepts.map(c => (
                <ProgressBar
                  key={c.conceptId}
                  label={c.conceptId.replace(/_/g, ' ')}
                  percent={Math.round(c.pKnown * 100)}
                  color={c.pKnown > 0.6 ? '#60a5fa' : c.pKnown > 0.3 ? '#fbbf24' : '#f87171'}
                />
              ))}
            </div>
          )}

          {knowledgeReport.strongestConcepts && knowledgeReport.strongestConcepts.length > 0 && (
            <div style={{ marginBottom: '0.75rem' }}>
              <div style={{ color: '#94a3b8', fontSize: '0.65rem', marginBottom: '0.5rem' }}>
                Strongest Concepts
              </div>
              {knowledgeReport.strongestConcepts.slice(0, 5).map(c => (
                <ProgressBar
                  key={c.conceptId}
                  label={c.conceptId.replace(/_/g, ' ')}
                  percent={Math.round(c.pKnown * 100)}
                  color="#4ade80"
                />
              ))}
            </div>
          )}

          {/* Decay risk alerts */}
          {knowledgeReport.decayRisks && knowledgeReport.decayRisks.length > 0 && (
            <div style={{
              fontSize: '0.65rem', color: '#fb923c', background: 'rgba(251, 146, 60, 0.08)',
              borderRadius: 6, padding: '8px 12px', lineHeight: 1.4,
            }}>
              Review needed: {knowledgeReport.decayRisks.map(r => r.conceptId.replace(/_/g, ' ')).join(', ')}
            </div>
          )}
        </div>
      )}

      {/* Latest Pedagogy Explanations */}
      {allExplanations && allExplanations.length > 0 && (
        <div style={{
          maxWidth: '640px',
          width: '100%',
          marginBottom: '2rem',
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: '12px',
          padding: '1.25rem',
        }}>
          <h3 style={{ color: '#e2e8f0', fontSize: '0.9rem', fontWeight: 700, margin: '0 0 0.75rem' }}>
            System Explanations
          </h3>
          {allExplanations.slice(-3).map((exp, i) => (
            <div key={i} style={{
              fontSize: '0.65rem', color: '#94a3b8', padding: '6px 0',
              borderBottom: i < Math.min(allExplanations.length, 3) - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none',
              lineHeight: 1.5,
            }}>
              <span style={{ color: exp.severity === 'CRITICAL' ? '#f87171' : exp.severity === 'WARNING' ? '#fbbf24' : '#60a5fa', fontWeight: 600 }}>
                [{exp.severity}]
              </span>{' '}
              {exp.narrative}
            </div>
          ))}
        </div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: '1rem' }}>
        <button
          onClick={onNewSession}
          style={{
            padding: '0.85rem 2rem',
            fontSize: '0.95rem',
            fontWeight: 700,
            color: '#0f0f1a',
            background: 'linear-gradient(135deg, #4ade80, #22d3ee)',
            border: 'none',
            borderRadius: '10px',
            cursor: 'pointer',
            boxShadow: '0 4px 20px rgba(74, 222, 128, 0.2)',
          }}
        >
          New Session
        </button>
        <button
          onClick={onBackToMenu}
          style={{
            padding: '0.85rem 2rem',
            fontSize: '0.95rem',
            fontWeight: 600,
            color: '#94a3b8',
            background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: '10px',
            cursor: 'pointer',
          }}
        >
          Back to Menu
        </button>
      </div>
    </div>
  );
}
