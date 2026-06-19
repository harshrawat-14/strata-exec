/**
 * Data Upload page — drag-and-drop zones for LOB CSV, AggTrades CSV, RL model.
 * Shows order book preview after upload.
 */

import { useState, useCallback } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { FileText, Brain } from 'lucide-react'
import { clsx } from 'clsx'

import { uploadLobFile, uploadAggFile, uploadModel, fetchLobDepthPreview, fetchUploadedFiles, fetchUploadedModels } from '../api/client'
import { OrderBookViz, SpreadChart } from '../components/OrderBookViz'
import { Skeleton } from '../components/ui'
import type { UploadedFileInfo, UploadedModelInfo } from '../types'

type DropZoneType = 'lob' | 'agg' | 'model'

interface DropZoneProps {
  type: DropZoneType
  accept: string
  label: string
  description: string
  icon: React.ReactNode
  onDrop: (file: File) => void
  isLoading?: boolean
  result?: { name: string } | null
  error?: string | null
}

function DropZone({ type, accept, label, description, icon, onDrop, isLoading, result, error }: DropZoneProps) {
  const [dragOver, setDragOver] = useState(false)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) onDrop(file)
  }, [onDrop])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onDrop(file)
  }

  return (
    <label
      htmlFor={`upload-${type}`}
      className={clsx(
        'relative flex flex-col items-center justify-center gap-3 p-8 rounded-xl border-dashed cursor-pointer transition-all duration-300',
        dragOver ? 'scale-[1.01]' : ''
      )}
      style={{
        border: `1.5px dashed ${dragOver ? 'var(--card-border-hover)' : result ? 'var(--card-border)' : 'var(--divider)'}`,
        background: dragOver ? 'var(--card-hover)' : result ? 'var(--card)' : 'transparent',
      }}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <input
        id={`upload-${type}`}
        type="file"
        accept={accept}
        className="sr-only"
        onChange={handleChange}
      />

      {isLoading ? (
        <div className="flex flex-col items-center gap-2">
          <div className="w-8 h-8 rounded-full border-2 border-t-transparent animate-spin" style={{ borderColor: 'var(--card-border-hover)', borderTopColor: 'transparent' }} />
          <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>Processing…</span>
        </div>
      ) : result ? (
        <>
          <div className="w-8 h-8 rounded-full flex items-center justify-center text-lg" style={{ background: 'var(--card-hover)' }}>✓</div>
          <div className="text-center">
            <div className="text-sm font-semibold" style={{ color: 'var(--text)' }}>{result.name}</div>
            <div className="text-xs font-mono mt-0.5" style={{ color: 'var(--text-muted)' }}>Uploaded successfully</div>
          </div>
          <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)', opacity: 0.5 }}>Click to replace</span>
        </>
      ) : error ? (
        <>
          <div className="w-8 h-8 rounded-full flex items-center justify-center text-lg" style={{ background: 'var(--card-hover)' }}>✗</div>
          <div className="text-center">
            <div className="text-sm font-semibold" style={{ color: 'var(--text)' }}>Upload failed</div>
            <div className="text-xs font-mono mt-0.5 max-w-48" style={{ color: 'var(--text-muted)' }}>{error}</div>
          </div>
        </>
      ) : (
        <>
          <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ border: '1px solid var(--card-border)', color: 'var(--text-muted)' }}>
            {icon}
          </div>
          <div className="text-center">
            <div className="text-sm font-semibold" style={{ color: 'var(--text)' }}>{label}</div>
            <div className="text-xs font-mono mt-1" style={{ color: 'var(--text-muted)' }}>{description}</div>
          </div>
          <div className="text-[10px] font-mono px-3 py-1 rounded-full" style={{ color: 'var(--text-muted)', border: '1px solid var(--divider)' }}>
            Drop file or click to browse
          </div>
        </>
      )}
    </label>
  )
}

export default function DataUpload() {
  const [lobResult, setLobResult] = useState<UploadedFileInfo | null>(null)
  const [aggResult, setAggResult] = useState<UploadedFileInfo | null>(null)
  const [modelResult, setModelResult] = useState<UploadedModelInfo | null>(null)

  const [lobError, setLobError] = useState<string | null>(null)
  const [aggError, setAggError] = useState<string | null>(null)
  const [modelError, setModelError] = useState<string | null>(null)

  const [previewFileId, setPreviewFileId] = useState<string | null>(null)

  const lobMut = useMutation({
    mutationFn: uploadLobFile,
    onSuccess: (data) => {
      setLobResult(data)
      setLobError(null)
      setPreviewFileId(data.file_id)
    },
    onError: (e: any) => setLobError(e?.response?.data?.detail || 'Upload failed'),
  })

  const aggMut = useMutation({
    mutationFn: uploadAggFile,
    onSuccess: (data) => { setAggResult(data); setAggError(null) },
    onError: (e: any) => setAggError(e?.response?.data?.detail || 'Upload failed'),
  })

  const modelMut = useMutation({
    mutationFn: uploadModel,
    onSuccess: (data) => { setModelResult(data); setModelError(null) },
    onError: (e: any) => setModelError(e?.response?.data?.detail || 'Upload failed'),
  })

  const { data: preview } = useQuery({
    queryKey: ['lob-preview', previewFileId],
    queryFn: () => fetchLobDepthPreview(previewFileId!),
    enabled: !!previewFileId,
  })

  const { data: existingFiles } = useQuery({
    queryKey: ['uploaded-files'],
    queryFn: () => fetchUploadedFiles(),
  })

  const { data: existingModels } = useQuery({
    queryKey: ['uploaded-models'],
    queryFn: fetchUploadedModels,
  })

  return (
    <div className="space-y-8 animate-fade-in">
      <div>
        <h1 className="text-3xl font-bold text-black dark:text-white tracking-tight">
          Data Upload
        </h1>
        <p className="text-black/40 dark:text-white/40 mt-1 text-sm font-mono uppercase tracking-wider">
          Upload Binance LOB data, aggregate trades, or trained RL models
        </p>
      </div>

      {/* Drop zones */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <DropZone
          type="lob"
          accept=".csv"
          label="Book Depth CSV"
          description="BTCUSDT-bookDepth-YYYY-MM-DD.csv"
          icon={<FileText size={18} />}
          onDrop={(f) => lobMut.mutate(f)}
          isLoading={lobMut.isPending}
          result={lobResult ? { name: lobResult.original_name } : null}
          error={lobError}
        />
        <DropZone
          type="agg"
          accept=".csv"
          label="Agg Trades CSV"
          description="BTCUSDT-aggTrades-YYYY-MM-DD.csv"
          icon={<FileText size={18} />}
          onDrop={(f) => aggMut.mutate(f)}
          isLoading={aggMut.isPending}
          result={aggResult ? { name: aggResult.original_name } : null}
          error={aggError}
        />
        <DropZone
          type="model"
          accept=".zip"
          label="RL Model (.zip)"
          description="SB3 RecurrentPPO model zip"
          icon={<Brain size={18} />}
          onDrop={(f) => modelMut.mutate(f)}
          isLoading={modelMut.isPending}
          result={modelResult ? { name: modelResult.name } : null}
          error={modelError}
        />
      </div>

      {/* Preview panel */}
      {lobResult && (
        <div className="glass-card p-6 space-y-5">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <h2 className="text-xs font-semibold text-black/40 dark:text-white/40 uppercase tracking-widest font-mono">
              Order Book Preview
            </h2>
            <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs font-mono">
              <div>
                <span className="text-black/30 dark:text-white/30">Mid Price </span>
                <span className="font-mono text-black dark:text-white font-semibold">
                  ${lobResult.preview?.mid_price.toLocaleString('en', { maximumFractionDigits: 2 })}
                </span>
              </div>
              <div>
                <span className="text-black/30 dark:text-white/30">Spread </span>
                <span className="font-mono text-black dark:text-white font-semibold">
                  {lobResult.preview?.spread_bps.toFixed(2)} bps
                </span>
              </div>
              <div>
                <span className="text-black/30 dark:text-white/30">Snapshots </span>
                <span className="font-mono text-black dark:text-white">{lobResult.n_rows?.toLocaleString()}</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <p className="text-xs text-black/30 dark:text-white/30 mb-3 font-mono uppercase tracking-wider">Bid / Ask Depth (first snapshot)</p>
              {preview ? (
                <OrderBookViz snapshot={preview.depth_snapshots[0] || null} />
              ) : (
                <Skeleton className="h-48" />
              )}
            </div>
            <div>
              <p className="text-xs text-black/30 dark:text-white/30 mb-3 font-mono uppercase tracking-wider">Spread over time (bps)</p>
              {preview ? (
                <SpreadChart series={preview.spread_series} />
              ) : (
                <Skeleton className="h-48" />
              )}
            </div>
          </div>
        </div>
      )}

      {/* Existing files table */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Files */}
        <div className="glass-card p-5">
          <h2 className="text-xs font-semibold text-black/40 dark:text-white/40 uppercase tracking-widest font-mono mb-4">
            Uploaded Files
          </h2>
          {!existingFiles?.length ? (
            <p className="text-xs text-black/30 dark:text-white/30 text-center py-6 font-mono">No files uploaded yet</p>
          ) : (
            <div className="space-y-1">
              {existingFiles.map((f) => (
                <div key={f.file_id} className="flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-black/3 dark:hover:bg-white/3">
                  <div className="flex items-center gap-2">
                    <div className={clsx(
                      'w-1.5 h-1.5 rounded-full',
                      f.file_type === 'lob' ? 'bg-black dark:bg-white' : 'border border-black dark:border-white bg-transparent'
                    )} />
                    <div>
                      <div className="text-xs font-mono text-black dark:text-white">{f.date_str || f.original_name}</div>
                      <div className="text-[10px] text-black/40 dark:text-white/40 font-mono">{f.file_type} · {(f.file_size_bytes / 1024).toFixed(0)} KB</div>
                    </div>
                  </div>
                  <span className="badge-neutral text-[10px]">
                    {f.file_type}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Models */}
        <div className="glass-card p-5">
          <h2 className="text-xs font-semibold text-black/40 dark:text-white/40 uppercase tracking-widest font-mono mb-4">
            Available Models
          </h2>
          {!existingModels?.length ? (
            <p className="text-xs text-black/30 dark:text-white/30 text-center py-6 font-mono">No models available</p>
          ) : (
            <div className="space-y-1">
              {existingModels.map((m) => (
                <div key={m.model_id} className="flex items-center justify-between px-3 py-2.5 rounded-lg hover:bg-black/3 dark:hover:bg-white/3">
                  <div className="flex items-center gap-2">
                    <Brain size={12} className={m.is_builtin ? 'text-black dark:text-white' : 'text-black/40 dark:text-white/40'} />
                    <div>
                      <div className="text-xs font-medium text-black dark:text-white">{m.name}</div>
                      <div className="text-[10px] text-black/40 dark:text-white/40 font-mono">{(m.file_size_bytes / 1024).toFixed(0)} KB</div>
                    </div>
                  </div>
                  <span className={m.is_builtin ? 'badge-success text-[10px]' : 'badge-info text-[10px]'}>
                    {m.is_builtin ? 'built-in' : 'uploaded'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
