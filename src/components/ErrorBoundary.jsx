import { Component } from 'react';

/**
 * ErrorBoundary — catches unhandled errors in the React tree and shows
 * a recovery UI instead of a white screen. Critical for MLAF reliability
 * since the pipeline has many moving parts (MediaPipe, AGGME, UMCE, Prolog).
 */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    console.error('[MLAF ErrorBoundary]', error, errorInfo);
  }

  handleRecover = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          padding: '2rem',
          maxWidth: '600px',
          margin: '4rem auto',
          fontFamily: 'system-ui, sans-serif',
          textAlign: 'center',
        }}>
          <h2 style={{ color: '#ef4444', marginBottom: '1rem' }}>
            MLAF Pipeline Error
          </h2>
          <p style={{ color: '#94a3b8', marginBottom: '1.5rem' }}>
            A component encountered an unexpected error. Your session data is preserved.
          </p>
          <details style={{
            textAlign: 'left',
            background: '#1e293b',
            padding: '1rem',
            borderRadius: '8px',
            marginBottom: '1.5rem',
            color: '#f87171',
            fontSize: '0.85rem',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}>
            <summary style={{ cursor: 'pointer', color: '#94a3b8' }}>Error details</summary>
            <p>{this.state.error?.toString()}</p>
            <p style={{ color: '#64748b', fontSize: '0.75rem' }}>
              {this.state.errorInfo?.componentStack}
            </p>
          </details>
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
            <button
              onClick={this.handleRecover}
              style={{
                padding: '0.75rem 1.5rem',
                background: '#3b82f6',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '1rem',
              }}
            >
              Try to Recover
            </button>
            <button
              onClick={this.handleReload}
              style={{
                padding: '0.75rem 1.5rem',
                background: '#475569',
                color: '#fff',
                border: 'none',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '1rem',
              }}
            >
              Reload App
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
