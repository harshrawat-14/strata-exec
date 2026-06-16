/**
 * Typed API client — wraps fetch with base URL and error handling.
 */

import axios from 'axios'
import type {
  DashboardStats,
  EvaluationRequest,
  EvaluationResult,
  JobStatus,
  LobDepthPreview,
  SimulationRequest,
  SimulationResult,
  StrategyInfo,
  UploadedFileInfo,
  UploadedModelInfo,
  AvailableDate,
} from '../types'

const BASE = import.meta.env.VITE_API_URL || ''

export const api = axios.create({
  baseURL: BASE,
  headers: { 'Content-Type': 'application/json' },
})

// Request interceptor: attach JWT token if present
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('strataexec_token')
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor: redirect to login on 401 Unauthorized
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('strataexec_token')
      // Only redirect if we are not already on the login page
      if (!window.location.pathname.endsWith('/login')) {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

export const loginUser = (credentials: any): Promise<{ access_token: string; token_type: string }> =>
  api.post('/api/auth/login', credentials).then((r) => r.data)

export const fetchMe = (): Promise<{ id: string; email: string; is_active: boolean }> =>
  api.get('/api/auth/me').then((r) => r.data)

// ── Dashboard ─────────────────────────────────────────────────────────────────
export const fetchDashboard = (): Promise<DashboardStats> =>
  api.get('/api/dashboard').then((r) => r.data)

export const fetchStrategies = (): Promise<StrategyInfo[]> =>
  api.get('/api/strategies').then((r) => r.data)

export const fetchDates = (): Promise<AvailableDate[]> =>
  api.get('/api/dates').then((r) => r.data)

export const fetchRecentJobs = (): Promise<unknown[]> =>
  api.get('/api/jobs').then((r) => r.data)

// ── Simulation ────────────────────────────────────────────────────────────────
export const startSimulation = (req: SimulationRequest): Promise<JobStatus> =>
  api.post('/api/simulate', req).then((r) => r.data)

export const fetchSimulationResult = (jobId: string): Promise<SimulationResult> =>
  api.get(`/api/compare/${jobId}`).then((r) => r.data)

// ── RL Evaluation ─────────────────────────────────────────────────────────────
export const startEvaluation = (req: EvaluationRequest): Promise<JobStatus> =>
  api.post('/api/evaluate', req).then((r) => r.data)

export const fetchEvaluationResult = (jobId: string): Promise<EvaluationResult> =>
  api.get(`/api/evaluate/result/${jobId}`).then((r) => r.data)

// ── Sweep ─────────────────────────────────────────────────────────────────────
export const startSweep = (req: object): Promise<JobStatus> =>
  api.post('/api/sweep', req).then((r) => r.data)

export const fetchSweepResult = (jobId: string): Promise<unknown> =>
  api.get(`/api/sweep/${jobId}`).then((r) => r.data)

// ── Upload ────────────────────────────────────────────────────────────────────
export const uploadLobFile = (file: File): Promise<UploadedFileInfo> => {
  const fd = new FormData()
  fd.append('file', file)
  return api.post('/api/upload/lob', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data)
}

export const uploadAggFile = (file: File): Promise<UploadedFileInfo> => {
  const fd = new FormData()
  fd.append('file', file)
  return api.post('/api/upload/agg-trades', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data)
}

export const uploadModel = (file: File): Promise<UploadedModelInfo> => {
  const fd = new FormData()
  fd.append('file', file)
  return api.post('/api/upload/rl-model', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then((r) => r.data)
}

export const fetchUploadedFiles = (fileType?: string): Promise<UploadedFileInfo[]> =>
  api.get('/api/upload/files', { params: fileType ? { file_type: fileType } : {} }).then((r) => r.data)

export const fetchUploadedModels = (): Promise<UploadedModelInfo[]> =>
  api.get('/api/upload/models').then((r) => r.data)

export const fetchLobDepthPreview = (fileId: string): Promise<LobDepthPreview> =>
  api.get(`/api/upload/lob-preview/${fileId}`).then((r) => r.data)

export const cancelJob = (jobId: string): Promise<{ status: string; message: string }> =>
  api.post(`/api/jobs/${jobId}/cancel`).then((r) => r.data)
