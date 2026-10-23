// Page 1 — Live Demo.
//
// Type or speak a command, then watch it travel through the pipeline:
// what was heard, how it split, each action in order, and the document
// before and after. The Old VDOC / VDOC 2.0 toggle runs the same command
// through the old single-action path so the difference is visible.
import { useCallback, useEffect, useState } from 'react'
import '../demo/demo.css'
import { createDoc, runActions, SAMPLE_LINES } from '../demo/document'
import {
  parseMulti, parseOld, fetchConfig, suggestCorrections, confirmCorrection,
} from '../demo/api'
import { useVoiceInput } from '../demo/useVoiceInput'
import {
  ActionCard, AutoCorrectedBadges, ChunkStrip, CorrectionPrompt,
  DocumentView, SnapBadges, WhisperDiagnostics,
} from '../demo/components'

const EXAMPLES = [
  'second line delete pannu, apram second line bold pannu',
  'first line bold pannu, apram third line delete pannu',
  'first line bold-um italic-um pannu',
  'bold the first line and delete the third line',
  'delete the second line, then bold the second line',
  'introduction la irunthu conclusion varikum bold panu',
]

export default function LiveDemoPage() {
  const [mode, setMode] = useState('new')         // 'old' | 'new'
  const [command, setCommand] = useState(EXAMPLES[0])
  const [doc, setDoc] = useState(() => createDoc())
  const [before, setBefore] = useState(null)
  const [result, setResult] = useState(null)      // { heard, chunks, actions, steps }
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [config, setConfig] = useState(null)
  const [diagnostics, setDiagnostics] = useState(null)
  const [pending, setPending] = useState(null)   // { command, suggestions }
  const [autoFixed, setAutoFixed] = useState([])
  const [snap, setSnap] = useState(null)         // { changes, original, snapped }

  useEffect(() => { fetchConfig().then(setConfig) }, [])

  const reset = useCallback(() => {
    setDoc(createDoc())
    setBefore(null)
    setResult(null)
    setError(null)
    setPending(null)
    setAutoFixed([])
    setSnap(null)
  }, [])

  // Parse and execute. `text` is already corrected by the time it gets here.
  const execute = useCallback(async (utterance, activeMode) => {
    setBusy(true)
    setError(null)

    // Always start from a clean document so the two modes are comparable.
    const startDoc = createDoc()
    const parsed = activeMode === 'old'
      ? await parseOld(utterance)
      : await parseMulti(utterance, startDoc.length)

    if (!parsed.ok) {
      setError(parsed.error)
      setBusy(false)
      return
    }

    const { doc: after, steps, touched } = runActions(startDoc, parsed.actions)

    setBefore(startDoc)
    setDoc(after)
    setResult({
      heard: utterance,
      mode: activeMode,
      chunks: parsed.chunks ?? [{ text: utterance, join_before: null }],
      actions: parsed.actions,
      steps,
      touched,
      split: parsed.split ?? false,
      splitter: parsed.splitter,
    })
    setBusy(false)
  }, [])

  /**
   * Run a command. Misheard command words are checked first: saved fixes are
   * applied straight away, anything new stops here and waits for a yes or no.
   */
  const run = useCallback(async (text, runMode) => {
    const utterance = (text ?? command).trim()
    const activeMode = runMode ?? mode
    if (!utterance) return

    setPending(null)
    setAutoFixed([])
    setSnap(null)

    // The document on screen joins the vocabulary, so its own words are
    // never snapped away and can be snapped *to*.
    const documentText = doc.map(line => line.text).join('\n')

    const review = await suggestCorrections(utterance, documentText)
    if (!review.ok) {           // corrector down — carry on with the raw text
      await execute(utterance, activeMode)
      return
    }

    if (review.snap_changes?.length) {
      setSnap({
        changes: review.snap_changes,
        original: review.original,
        snapped: review.snapped,
      })
      setCommand(review.corrected)
    }

    if (review.auto_applied?.length) {
      setAutoFixed(review.auto_applied)
      setCommand(review.corrected)
    }

    if (review.suggestions?.length) {
      setPending({ command: review.corrected, suggestions: review.suggestions, mode: activeMode })
      return
    }

    await execute(review.corrected, activeMode)
  }, [command, mode, execute, doc])

  // ── Correction prompt handlers ───────────────────────────────────────────
  const acceptSuggestion = useCallback(async item => {
    if (!pending) return
    setBusy(true)
    const saved = await confirmCorrection(item.word, item.suggestion, pending.command)
    setBusy(false)
    if (!saved.ok) { setError(saved.error); return }

    const corrected = saved.command || pending.command
    setCommand(corrected)
    setAutoFixed(prev => [...prev, saved.saved])
    const rest = pending.suggestions.filter(s => s.word !== item.word)
    if (rest.length) {
      setPending({ ...pending, command: corrected, suggestions: rest })
    } else {
      setPending(null)
      await execute(corrected, pending.mode)
    }
  }, [pending, execute])

  const rejectSuggestion = useCallback(async item => {
    if (!pending) return
    const rest = pending.suggestions.filter(s => s.word !== item.word)
    if (rest.length) {
      setPending({ ...pending, suggestions: rest })
    } else {
      setPending(null)
      await execute(pending.command, pending.mode)
    }
  }, [pending, execute])

  const runAnyway = useCallback(async () => {
    if (!pending) return
    const { command: text, mode: runMode } = pending
    setPending(null)
    await execute(text, runMode)
  }, [pending, execute])

  const onTranscribed = useCallback((text, diag) => {
    setDiagnostics(diag ?? null)
    setCommand(text)
    run(text)
  }, [run])

  const { recording, voiceStatus, toggleMic } = useVoiceInput(onTranscribed)

  const switchMode = next => {
    setMode(next)
    if (result) run(command, next)
  }

  const removed = before ? before.length - doc.length : 0

  return (
    <div className="demo-page">
      {/* ── Controls ── */}
      <div className="demo-bar">
        <div className="mode-toggle" role="group" aria-label="Pipeline version">
          <button
            className={`mode-btn${mode === 'old' ? ' is-active old' : ''}`}
            onClick={() => switchMode('old')}
          >
            Old VDOC
          </button>
          <button
            className={`mode-btn${mode === 'new' ? ' is-active new' : ''}`}
            onClick={() => switchMode('new')}
          >
            VDOC 2.0
          </button>
        </div>

        <input
          className="command-input"
          value={command}
          onChange={e => setCommand(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') run() }}
          placeholder="Type a command, or press the mic and speak it"
          aria-label="Command"
        />

        <button className="run-btn" onClick={() => run()} disabled={busy}>
          {busy ? 'Running…' : 'Run'}
        </button>
        <button
          className={`mic-btn${recording ? ' recording' : ''}`}
          onClick={toggleMic}
          title={recording ? 'Stop recording' : 'Speak a command'}
          aria-label={recording ? 'Stop recording' : 'Speak a command'}
        >
          {recording ? '⏹' : '🎙'}
        </button>
        <button className="reset-btn" onClick={reset}>Reset document</button>
      </div>

      <div className="example-row">
        <span className="example-label">Try:</span>
        {EXAMPLES.map(example => (
          <button key={example} className="example-chip" onClick={() => { setCommand(example); run(example) }}>
            {example}
          </button>
        ))}
      </div>

      {voiceStatus && <div className="notice">{voiceStatus}</div>}
      {error && <div className="notice is-error">{error}</div>}

      <SnapBadges
        changes={snap?.changes}
        original={snap?.original}
        snapped={snap?.snapped}
      />

      <AutoCorrectedBadges applied={autoFixed} />

      <CorrectionPrompt
        suggestions={pending?.suggestions}
        onYes={acceptSuggestion}
        onNo={rejectSuggestion}
        onRunAnyway={runAnyway}
        busy={busy}
      />

      <WhisperDiagnostics diagnostics={diagnostics} />

      {/* ── Steps ── */}
      <div className="demo-grid">
        <div className="demo-steps">
          <Step n={1} title="Heard">
            {result
              ? <div className="heard-text">“{result.heard}”</div>
              : <div className="muted">Run a command to see the steps.</div>}
          </Step>

          <Step
            n={2}
            title="Split into chunks"
            note={result
              ? (result.mode === 'old'
                  ? 'Old VDOC does not split — the whole sentence goes to the model as one command.'
                  : `Splitter ${result.splitter}${result.split ? '' : ' — no safe split, kept as one chunk'}`)
              : null}
          >
            {result
              ? <ChunkStrip chunks={result.chunks} />
              : <div className="muted">—</div>}
          </Step>

          <Step
            n={3}
            title="Actions"
            note={result && result.mode === 'old' && result.actions.length === 1
              ? 'Old VDOC returns at most one action, whatever was said.'
              : null}
          >
            {result
              ? (
                <div className="action-list">
                  {result.actions.map((entry, i) => (
                    <ActionCard
                      key={i}
                      entry={entry}
                      index={i}
                      total={result.actions.length}
                      step={result.steps[i]}
                    />
                  ))}
                </div>
              )
              : <div className="muted">—</div>}
          </Step>
        </div>

        {/* ── Document before / after ── */}
        <div className="demo-docs">
          <Step n={4} title="Document">
            <div className="doc-pair">
              <DocumentView
                title="Before"
                doc={before ?? createDoc(SAMPLE_LINES)}
              />
              <DocumentView
                title="After"
                doc={doc}
                touched={result?.touched ?? []}
                removedCount={removed > 0 ? removed : 0}
              />
            </div>
          </Step>
        </div>
      </div>

      {config?.ok && (
        <div className="config-foot">
          {config.model} · splitter {config.splitter} · line resolver {config.line_resolver}
          {' · '}LLM fallback: {config.groq_fallback}
        </div>
      )}
    </div>
  )
}

function Step({ n, title, note, children }) {
  return (
    <section className="step">
      <h3 className="step-title"><span className="step-no">{n}</span>{title}</h3>
      {note && <div className="step-note">{note}</div>}
      <div className="step-body">{children}</div>
    </section>
  )
}
