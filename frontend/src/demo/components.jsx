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
  const failed = step && !step.ok

  return (
    <div className={`action-card${failed ? ' is-failed' : ''}`}>
      <div className="action-card-head">
        <span className="action-card-step">{index + 1} of {total}</span>
        <span className="action-card-verb">{entry.action ?? 'unknown'}</span>
        {entry.shared_target && (
          <span className="badge badge-shared" title="target borrowed from the previous chunk">
            shared target
          </span>
        )}
        <span className={`badge badge-${target?.kind ?? 'none'}`}>{kind}</span>
      </div>

      <div className="action-card-target">{targetLabel(entry)}</div>

      <div className="action-card-meta">
        <span className="chunk-quote">“{entry.chunk_text}”</span>
        {typeof entry.confidence === 'number' && (
          <span className="conf">conf {entry.confidence.toFixed(3)}</span>
        )}
      </div>

      {step && (
        <div className={`action-card-result${step.ok ? '' : ' is-failed'}`}>
          {step.ok ? '✔ ' : '✖ '}{step.message}
        </div>
      )}
    </div>
  )
}
