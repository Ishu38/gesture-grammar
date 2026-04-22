/**
 * GuidedPracticePanel.jsx — Guided Practice Mode
 * Step-by-step lesson walkthrough using CurriculumEngine infrastructure.
 * Shows lesson selector, challenge prompts, hints, and feedback.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { LESSON_INDEX, getLessonsForStage } from '../data/lessonIndex';
import { LessonManager } from '../utils/CurriculumEngine';
import VisualSentenceSlots from './VisualSentenceSlots';

// =============================================================================
// LESSON SELECTOR
// =============================================================================

function LessonSelector({ onSelectLesson, masteryStage, completedLessons }) {
  const available = getLessonsForStage(masteryStage || 5);

  return (
    <div style={{ padding: 16 }}>
      <h3 style={{ color: '#e2e8f0', fontSize: 16, marginBottom: 12 }}>
        Choose a Lesson
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {available.map(lesson => {
          const completed = completedLessons?.includes(lesson.id);
          return (
            <button key={lesson.id} onClick={() => onSelectLesson(lesson)}
              style={{
                background: completed ? 'rgba(74, 222, 128, 0.1)' : '#1a1a2e',
                border: `1px solid ${completed ? '#4ade80' : '#4a4a6a'}`,
                borderRadius: 8, padding: '10px 14px',
                cursor: 'pointer', textAlign: 'left',
                color: '#e2e8f0', transition: 'border-color 0.2s',
              }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontWeight: 600, fontSize: 13 }}>{lesson.title}</span>
                <span style={{
                  fontSize: 10, padding: '2px 6px', borderRadius: 4,
                  background: completed ? '#4ade80' : '#4a4a6a',
                  color: completed ? '#0f0f1a' : '#94a3b8',
                  fontWeight: 600,
                }}>
                  {completed ? 'Done' : `Stage ${lesson.stage}`}
                </span>
              </div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
                Difficulty: {'★'.repeat(lesson.difficulty)}{'☆'.repeat(3 - lesson.difficulty)}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// =============================================================================
// CHALLENGE VIEW
// =============================================================================

function ChallengeView({ challenge, challengeIndex, totalChallenges, sentence, onSubmit, onClear, onGetHint, hintText, feedback }) {
  return (
    <div style={{ padding: 16 }}>
      {/* Progress bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <div style={{
          flex: 1, height: 4, background: '#2a2a3e', borderRadius: 2, overflow: 'hidden',
        }}>
          <div style={{
            width: `${((challengeIndex + 1) / totalChallenges) * 100}%`,
            height: '100%', background: '#60a5fa', borderRadius: 2,
            transition: 'width 0.3s ease',
          }} />
        </div>
        <span style={{ fontSize: 11, color: '#94a3b8' }}>
          {challengeIndex + 1}/{totalChallenges}
        </span>
      </div>

      {/* Challenge prompt */}
      <div style={{
        background: '#1a1a2e', borderRadius: 10, padding: 16,
        border: '1px solid #3a3a5e', marginBottom: 12, textAlign: 'center',
      }}>
        <div style={{ fontSize: 10, color: '#94a3b8', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
          {challenge.target_concept}
        </div>
        <div style={{ fontSize: 20, fontWeight: 700, color: '#e2e8f0' }}>
          {challenge.prompt_display}
        </div>
        {challenge.target_concept_explanation && (
          <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8, lineHeight: 1.4 }}>
            {challenge.target_concept_explanation}
          </div>
        )}
      </div>

      {/* Visual slots showing target */}
      <VisualSentenceSlots sentence={sentence} />

      {/* Hint */}
      {hintText && (
        <div style={{
          background: 'rgba(96, 165, 250, 0.1)', border: '1px solid #60a5fa',
          borderRadius: 8, padding: '8px 12px', marginTop: 8, marginBottom: 8,
          fontSize: 12, color: '#60a5fa',
        }}>
          Hint: {hintText}
        </div>
      )}

      {/* Feedback */}
      {feedback && (
        <div style={{
          background: feedback.isCorrect ? 'rgba(74, 222, 128, 0.1)' : 'rgba(248, 113, 113, 0.1)',
          border: `1px solid ${feedback.isCorrect ? '#4ade80' : '#f87171'}`,
          borderRadius: 8, padding: '10px 12px', marginTop: 8, marginBottom: 8,
          fontSize: 12, color: feedback.isCorrect ? '#4ade80' : '#f87171',
        }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>
            {feedback.isCorrect ? 'Correct!' : feedback.isPartial ? 'Keep going...' : 'Not quite.'}
          </div>
          <div style={{ color: '#94a3b8' }}>{feedback.feedback}</div>
          {feedback.gestureHint && (
            <div style={{ color: '#facc15', marginTop: 4 }}>
              Tip: {feedback.gestureHint}
            </div>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: 8, marginTop: 12, justifyContent: 'center' }}>
        <button onClick={onGetHint} style={{
          background: '#2a2a3e', border: '1px solid #4a4a6a', borderRadius: 6,
          padding: '8px 16px', color: '#60a5fa', cursor: 'pointer', fontSize: 12, fontWeight: 600,
        }}>
          Hint
        </button>
        <button onClick={onClear} style={{
          background: '#2a2a3e', border: '1px solid #4a4a6a', borderRadius: 6,
          padding: '8px 16px', color: '#f87171', cursor: 'pointer', fontSize: 12, fontWeight: 600,
        }}>
          Clear
        </button>
        <button onClick={onSubmit} style={{
          background: '#4ade80', border: 'none', borderRadius: 6,
          padding: '8px 20px', color: '#0f0f1a', cursor: 'pointer', fontSize: 12, fontWeight: 700,
        }}>
          Submit
        </button>
      </div>
    </div>
  );
}

// =============================================================================
// LESSON SUMMARY
// =============================================================================

function LessonSummary({ summary, onNextLesson, onBackToLessons }) {
  if (!summary) return null;

  return (
    <div style={{ padding: 16, textAlign: 'center' }}>
      <div style={{ fontSize: 28, marginBottom: 8 }}>
        {summary.sessionScore >= 150 ? '🌟' : summary.sessionScore >= 100 ? '⭐' : '📝'}
      </div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 4 }}>Lesson Complete!</h3>
      <div style={{ fontSize: 24, fontWeight: 700, color: '#fbbf24', marginBottom: 12 }}>
        {summary.sessionScore} points
      </div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 16 }}>
        {summary.completedChallenges}/{summary.totalChallenges} challenges completed
      </div>

      <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
        <button onClick={onBackToLessons} style={{
          background: '#2a2a3e', border: '1px solid #4a4a6a', borderRadius: 6,
          padding: '8px 16px', color: '#e2e8f0', cursor: 'pointer', fontSize: 12,
        }}>
          Back to Lessons
        </button>
        <button onClick={onNextLesson} style={{
          background: '#60a5fa', border: 'none', borderRadius: 6,
          padding: '8px 16px', color: '#0f0f1a', cursor: 'pointer', fontSize: 12, fontWeight: 700,
        }}>
          Next Lesson
        </button>
      </div>
    </div>
  );
}

// =============================================================================
// MAIN PANEL
// =============================================================================

export default function GuidedPracticePanel({ sentence, onClearSentence, onExitPractice, masteryReport }) {
  const [lessonManager] = useState(() => new LessonManager());
  const [activeLesson, setActiveLesson] = useState(null);
  const [currentChallenge, setCurrentChallenge] = useState(null);
  const [hintText, setHintText] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [lessonSummary, setLessonSummary] = useState(null);
  const [completedLessons, setCompletedLessons] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('mlaf_completed_lessons') || '[]');
    } catch { return []; }
  });

  const masteryStage = masteryReport?.currentStage || 5;

  // Refs to avoid stale closures inside setTimeout
  const activeLessonRef = useRef(activeLesson);
  const completedLessonsRef = useRef(completedLessons);
  const submitTimerRef = useRef(null);
  activeLessonRef.current = activeLesson;
  completedLessonsRef.current = completedLessons;

  // Clean up submit timer on unmount
  useEffect(() => {
    return () => { if (submitTimerRef.current) clearTimeout(submitTimerRef.current); };
  }, []);

  const handleSelectLesson = useCallback(async (lessonMeta) => {
    try {
      const module = await lessonMeta.file();
      const lessonData = module.default || module;
      lessonManager.loadLesson(lessonData);
      setActiveLesson(lessonData);
      setLessonSummary(null);
      setFeedback(null);
      setHintText(null);
      const challenge = lessonManager.getCurrentChallenge();
      setCurrentChallenge(challenge);
      onClearSentence?.();
    } catch (err) {
      console.error('Failed to load lesson:', err);
    }
  }, [lessonManager, onClearSentence]);

  const handleSubmit = useCallback(() => {
    if (!sentence || sentence.length === 0) {
      setFeedback({ isCorrect: false, feedback: 'Build a sentence first using gestures!' });
      return;
    }

    const userSequence = sentence.map(w => w.grammar_id);
    const result = lessonManager.submitAnswer(userSequence);
    setFeedback(result);

    if (result.isCorrect) {
      // Move to next challenge after a short delay
      // Uses refs to avoid stale closure over activeLesson/completedLessons
      submitTimerRef.current = setTimeout(() => {
        submitTimerRef.current = null;
        const next = lessonManager.getCurrentChallenge();
        if (next) {
          setCurrentChallenge(next);
          setFeedback(null);
          setHintText(null);
          onClearSentence?.();
        } else {
          // Lesson complete
          const summary = lessonManager.getLessonSummary();
          setLessonSummary(summary);
          setCurrentChallenge(null);

          // Mark lesson as completed (using refs for latest values)
          const currentLesson = activeLessonRef.current;
          const currentCompleted = completedLessonsRef.current;
          if (currentLesson && !currentCompleted.includes(currentLesson.level_id)) {
            const updated = [...currentCompleted, currentLesson.level_id];
            setCompletedLessons(updated);
            try { localStorage.setItem('mlaf_completed_lessons', JSON.stringify(updated)); } catch {}
          }
        }
      }, 1500);
    }
  }, [sentence, lessonManager, onClearSentence]);

  const handleGetHint = useCallback(() => {
    const hint = lessonManager.getHint();
    if (hint) setHintText(hint.hint);
  }, [lessonManager]);

  const handleClear = useCallback(() => {
    onClearSentence?.();
    setFeedback(null);
  }, [onClearSentence]);

  const handleBackToLessons = useCallback(() => {
    setActiveLesson(null);
    setCurrentChallenge(null);
    setLessonSummary(null);
    setFeedback(null);
    setHintText(null);
    onClearSentence?.();
  }, [onClearSentence]);

  return (
    <div style={{
      background: '#0f0f1a', borderRadius: 12,
      border: '1px solid #3a3a5e', overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '10px 16px', background: '#1a1a2e', borderBottom: '1px solid #3a3a5e',
      }}>
        <span style={{ color: '#60a5fa', fontWeight: 700, fontSize: 14 }}>
          {activeLesson ? activeLesson.title : 'Guided Practice'}
        </span>
        <button onClick={onExitPractice} style={{
          background: 'none', border: '1px solid #4a4a6a', borderRadius: 4,
          padding: '4px 10px', color: '#94a3b8', cursor: 'pointer', fontSize: 11,
        }}>
          Exit
        </button>
      </div>

      {/* Content */}
      {lessonSummary ? (
        <LessonSummary
          summary={lessonSummary}
          onNextLesson={handleBackToLessons}
          onBackToLessons={handleBackToLessons}
        />
      ) : currentChallenge ? (
        <ChallengeView
          challenge={currentChallenge.challenge}
          challengeIndex={currentChallenge.index}
          totalChallenges={currentChallenge.total}
          sentence={sentence}
          onSubmit={handleSubmit}
          onClear={handleClear}
          onGetHint={handleGetHint}
          hintText={hintText}
          feedback={feedback}
        />
      ) : (
        <LessonSelector
          onSelectLesson={handleSelectLesson}
          masteryStage={masteryStage}
          completedLessons={completedLessons}
        />
      )}
    </div>
  );
}
