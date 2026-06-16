import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { loginUser } from '../api/client'
import './Login.css'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!email || !password) {
      setError('Please fill in all fields.')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const data = await loginUser({ email, password })
      localStorage.setItem('strataexec_token', data.access_token)
      navigate('/')
    } catch (err: any) {
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail)
      } else {
        setError('Connection failed. Please check your credentials or backend server status.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-container">
      <div className="login-backdrop">
        <div className="glow-sphere glow-sphere-1" />
        <div className="glow-sphere glow-sphere-2" />
      </div>

      <div className="login-card glass-card">
        <div className="login-header">
          <div className="brand-icon">✦</div>
          <h1 className="brand-title">StrataExec</h1>
          <p className="brand-subtitle">Quantitative Strategy Execution Platform</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {error && (
            <div className="error-alert">
              <span className="error-icon">⚠</span>
              <span className="error-message">{error}</span>
            </div>
          )}

          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="e.g. admin@strataexec.com"
              required
              disabled={loading}
              autoComplete="email"
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              disabled={loading}
              autoComplete="current-password"
            />
          </div>

          <button type="submit" className="login-btn" disabled={loading}>
            {loading ? (
              <span className="btn-loading-spinner">
                <span className="spinner-dot" />
                Signing In...
              </span>
            ) : (
              'Sign In ✦'
            )}
          </button>
        </form>

        <div className="login-footer">
          <p>Seeded credentials: <code>admin@strataexec.com</code> / <code>strataexec</code></p>
        </div>
      </div>
    </div>
  )
}
