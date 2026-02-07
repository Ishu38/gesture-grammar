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
    </div>
  )
}

export default App
