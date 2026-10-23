// Shared pieces for the VDOC 2.0 demo pages.
import { segmentsOf } from './document'

// ── Document view ───────────────────────────────────────────────────────────

function Segment({ segment }) {
  const style = {}
  if (segment.types.includes('bold')) style.fontWeight = 700
  if (segment.types.includes('italic')) style.fontStyle = 'italic'
  if (segment.types.includes('underline')) style.textDecoration = 'underline'
  if (segment.types.includes('highlight')) style.background = '#fff176'
  if (segment.types.includes('fontSize')) style.fontSize = '1.35em'
  return <span style={style}>{segment.text}</span>
}

export function DocumentView({ title, doc, touched = [], removedCount = 0 }) {
  const changed = new Set(touched)
  return (
    <div className="doc-panel">
      <div className="doc-panel-title">{title}</div>
      <div className="doc-body">
        {doc.map((line, i) => {
          const Tag = line.heading === 2 ? 'h2' : line.heading === 3 ? 'h3' : 'p'
          return (
            <div
              key={line.id}
              className={`doc-line${changed.has(line.id) ? ' is-changed' : ''}`}
            >
              <span className="doc-line-no">{i + 1}</span>
              <Tag className="doc-line-text">
                {segmentsOf(line).map((segment, k) => (
                  <Segment key={k} segment={segment} />
                ))}
              </Tag>
            </div>
          )
        })}
        {doc.length === 0 && <div className="doc-empty">(document is empty)</div>}
      </div>
      {removedCount > 0 && (
        <div className="doc-foot">{removedCount} line{removedCount > 1 ? 's' : ''} removed</div>
      )}
    </div>
  )
}

// ── Chunks after splitting ──────────────────────────────────────────────────

export function ChunkStrip({ chunks }) {
  if (!chunks?.length) return null
  return (
    <div className="chunk-strip">
      {chunks.map((chunk, i) => (
        <span key={i} className="chunk-group">
          {chunk.join_before && (
            <span className="join-word" title="joining word">{chunk.join_before}</span>
          )}
          <span className="chunk-pill">
            <span className="chunk-pill-no">{i + 1}</span>
            {chunk.text}
          </span>
        </span>
      ))}
    </div>
  )
}

// ── One action card ─────────────────────────────────────────────────────────

/** The target line the user asked for, phrased for a projector. */
export function targetLabel(entry) {
  const target = entry.target
  if (!target) return 'no target found'

  if (target.kind === 'anchor') {
    if (!target.from || !target.to) {
      return `incomplete anchor (from: ${target.from ?? '—'}, to: ${target.to ?? '—'})`
    }
    return `from "${target.from}" to "${target.to}"`
  }

  const unit = target.unit || 'line'
  const resolved = entry.resolved_line
  if (!resolved) {
    return target.n === -1 ? `last ${unit}` : `${unit} ${target.n}`
  }
  return resolved.moved
    ? `${unit} ${resolved.current} (moved from ${unit} ${resolved.original})`
    : `${unit} ${resolved.current}`
}

export function ActionCard({ entry, index, total, step }) {
  const target = entry.target
  const kind = target ? (target.kind === 'line' ? 'line target' : 'anchor target') : 'no target'
  const unsure = entry.low_confidence
  const failed = step && !step.ok

  return (
    <div className={`action-card${unsure ? ' is-unsure' : failed ? ' is-failed' : ''}`}>
      <div className="action-card-head">
        <span className="action-card-step">{index + 1} of {total}</span>
        <span className="action-card-verb">{entry.action ?? 'unknown'}</span>
        {entry.shared_target && (
          <span className="badge badge-shared" title="target borrowed from the previous chunk">
            shared target
          </span>
        )}
        {unsure
          ? <span className="badge badge-unsure">low confidence</span>
          : <span className={`badge badge-${target?.kind ?? 'none'}`}>{kind}</span>}
      </div>

      <div className="action-card-target">{targetLabel(entry)}</div>

      <div className="action-card-meta">
        <span className="chunk-quote">“{entry.chunk_text}”</span>
        {typeof entry.confidence === 'number' && (
          <span className={`conf${unsure ? ' is-unsure' : ''}`}>
            conf {entry.confidence.toFixed(3)}
          </span>
        )}
      </div>

      {step && (
        <div className={`action-card-result${step.ok ? '' : step.unsure ? ' is-unsure' : ' is-failed'}`}>
          {step.ok ? '✔ ' : step.unsure ? '🤔 ' : '✖ '}{step.message}
        </div>
      )}
    </div>
  )
}

// ── Misheard-word correction ────────────────────────────────────────────────

/** "Did you mean bold?" — nothing is rewritten until Yes is clicked. */
export function CorrectionPrompt({ suggestions, onYes, onNo, onRunAnyway, busy }) {
  if (!suggestions?.length) return null
  return (
    <div className="correction-panel">
      <div className="correction-head">
        That may be a misheard word. Nothing has been changed yet.
      </div>
      {suggestions.map(item => (
        <div key={item.word} className="correction-row">
          <span className="correction-q">
            Did you mean <strong>{item.suggestion}</strong> instead of
            {' '}<span className="heard-word">{item.word}</span>?
          </span>
          <span className="correction-distance">edit distance {item.distance}</span>
          <button className="yes-btn" disabled={busy} onClick={() => onYes(item)}>Yes</button>
          <button className="no-btn" disabled={busy} onClick={() => onNo(item)}>No</button>
        </div>
      ))}
      <button className="run-anyway-btn" disabled={busy} onClick={onRunAnyway}>
        Run the command as heard
      </button>
    </div>
  )
}

/** "auto-corrected: bowl → bold" for fixes confirmed in an earlier session. */
export function AutoCorrectedBadges({ applied }) {
  if (!applied?.length) return null
  return (
    <div className="auto-corrected-row">
      {applied.map((item, i) => (
        <span key={i} className="badge badge-autofix">
          auto-corrected: {item.from} → {item.to}
        </span>
      ))}
    </div>
  )
}

/** What Whisper did with the clip — the ASR half of the pipeline. */
export function WhisperDiagnostics({ diagnostics }) {
  if (!diagnostics) return null
  const d = diagnostics
  return (
    <div className="diag-panel">
      <div className="diag-title">Whisper (speech-to-text)</div>
      <div className="diag-grid">
        <Field label="model" value={d.model} />
        <Field label="language" value={d.language} />
        <Field label="clip length" value={`${d.clip_seconds}s`} />
        <Field label="sample rate" value={`${d.sample_rate} Hz`} />
        <Field label="source" value={d.source_codec
          ? `${d.source_codec} ${d.source_sample_rate ?? '?'} Hz, ${d.source_channels ?? '?'} ch`
          : 'unknown'} />
        <Field label="leading pad" value={`${d.leading_pad_ms} ms`} />
        <Field label="decode time" value={`${d.inference_seconds}s`} />
        <Field label="peak level" value={d.silent ? `${d.peak_amplitude} (silent?)` : d.peak_amplitude} />
        <Field label="saved to" value={d.debug_wav} />
      </div>
    </div>
  )
}

function Field({ label, value }) {
  return (
    <div className="diag-field">
      <span className="diag-label">{label}</span>
      <span className="diag-value">{String(value)}</span>
    </div>
  )
}
