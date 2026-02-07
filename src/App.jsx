import SandboxMode from './components/SandboxMode'
import './App.css'

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Gesture Grammar</h1>
        <p className="subtitle">Build sentences with hand gestures</p>
      </header>
      <SandboxMode />
      <footer className="app-footer">
        Designed &amp; Created by Neil Shankar Ray
      </footer>
    </div>
  )
}

export default App
