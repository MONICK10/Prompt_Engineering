// Pieces used only by the Prompt Engineering page.
//
// The styling vocabulary is demo.css's (.step, .badge, .doc-*); pe.css adds
// the few classes these need on top. Nothing here is imported by the other
// pages, so they are untouched.
import { Fragment } from 'react'
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

// The same colour per model the report charts use, so the printed figures
// and the web page agree.
export const MODEL_COLORS = {
  gemini: '#2a78d6',
  groq: '#eb6834',
  'ollama-local': '#1baf7a',
}
const SPARE = '#eda100'
export const colorOf = name => MODEL_COLORS[name] ?? SPARE

/** Same numbered card the Live Demo uses, redeclared so that page is untouched. */
export function Step({ n, title, note, children }) {
  return (
    <section className="step">
      <h3 className="step-title"><span className="step-no">{n}</span>{title}</h3>
      {note && <div className="step-note">{note}</div>}
      <div className="step-body">{children}</div>
    </section>
  )
}

const FLOW = [
  ['Voice', 'The student speaks, in English or Tanglish.'],
  ['Text', 'Whisper writes down what it heard.'],
  ['Prompt + LLM', 'The words go into a prompt. An LLM reads it.'],
  ['JSON', 'The LLM replies with one action object.'],
  ['Document edit', 'VDOC runs that action on the document.'],
]

export function FlowDiagram() {
  return (
    <div className="pe-flow">
      {FLOW.map(([title, caption], i) => (
        <Fragment key={title}>
          <div className="pe-flow-step">
            <div className="pe-flow-box">{title}</div>
            <div className="pe-flow-cap">{caption}</div>
          </div>
          {i < FLOW.length - 1 && <div className="pe-flow-arrow">→</div>}
        </Fragment>
      ))}
    </div>
  )
}

/** Prompt sent, raw reply, parsed JSON, time taken — in that order. */
export function ReplyBlock({ result }) {
  if (!result) return <div className="muted">Run a command to see the steps.</div>
  return (
    <>
      <details className="pe-prompt">
        <summary>Prompt sent to {result.model_label} ({result.prompt_sent?.length ?? 0} characters)</summary>
        <pre className="pe-pre is-light">{result.prompt_sent}</pre>
      </details>

      <div className="pe-field-label">Raw AI reply</div>
      <pre className="pe-pre">{result.raw_output || '(the model replied with nothing)'}</pre>

      <div className="pe-field-label">Parsed JSON</div>
      {result.parsed_json
        ? <pre className="pe-pre is-light">{JSON.stringify(result.parsed_json, null, 2)}</pre>
        : <div className="notice is-error">AI reply was not valid JSON</div>}

      {result.note && <div className="pe-card-note">Note: {result.note}</div>}
      <div className="pe-time">Took {result.latency_s?.toFixed(2)} s</div>
    </>
  )
}

/** One prompt's answer, as a card with a tick or a cross. */
export function CompareCard({ result }) {
  const ok = Boolean(result.ok)
  return (
    <div className="pe-card">
      <div className="pe-card-head">
        <span className="pe-card-title">{result.technique_label}</span>
        <span className={`pe-mark ${ok ? 'ok' : 'bad'}`}>{ok ? '✓' : '✗'}</span>
      </div>
      {result.parsed_json
        ? <pre className="pe-pre is-light">{JSON.stringify(result.parsed_json, null, 2)}</pre>
        : <div className="notice is-error">AI reply was not valid JSON</div>}
      {result.note && <div className="pe-card-note">{result.note}</div>}
      <div className="pe-time">{result.latency_s?.toFixed(2)} s</div>
    </div>
  )
}

/** The trained model's answer on the same text, for the contrast. */
export function BertCard({ bert }) {
  if (!bert) return null
  const entry = bert.actions?.[0]
  return (
    <div className="pe-card is-bert">
      <div className="pe-card-head">
        <span className="pe-card-title">DocEditBERT (trained model)</span>
        {entry && (
          <span className={`badge ${entry.low_confidence ? 'badge-unsure' : 'badge-line'}`}>
            {entry.low_confidence ? 'low confidence' : 'confident'}
          </span>
        )}
      </div>
      {entry
        ? (
          <>
            <div className="pe-field-label">Action</div>
            <div className="action-card-verb">{entry.action ?? 'nothing parsed'}</div>
            <div className="action-card-target">{entry.label}</div>
            <div className="pe-time">
              confidence {entry.confidence == null ? '—' : entry.confidence.toFixed(3)}
            </div>
          </>
        )
        : <div className="muted">No action returned.</div>}
      <div className="pe-card-note">
        This one was trained on our data. The cards beside it were only given a prompt.
      </div>
    </div>
  )
}

/** Accuracy %, models down the side, techniques across the top. */
export function AccuracyTable({ table, models, techniques }) {
  let best = -1
  models.forEach(m => techniques.forEach(t => {
    const v = table[m.name]?.[t.name]
    if (v != null && v > best) best = v
  }))

  return (
    <div className="pe-table-wrap">
      <table className="pe-table">
        <thead>
          <tr>
            <th>Model</th>
            {techniques.map(t => <th key={t.name}>{t.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {models.map(m => (
            <tr key={m.name}>
              <th>{m.label}</th>
              {techniques.map(t => {
                const v = table[m.name]?.[t.name]
                if (v == null) return <td key={t.name} className="none">not run</td>
                return (
                  <td key={t.name} className={`num${v === best ? ' best' : ''}`}>
                    {v.toFixed(1)}%
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function toSeries(table, models, techniques) {
  return techniques.map(t => {
    const row = { name: t.label }
    models.forEach(m => {
      const v = table[m.name]?.[t.name]
      if (v != null) row[m.name] = v
    })
    return row
  })
}

export function AccuracyChart({ table, models, techniques }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={toSeries(table, models, techniques)}
                margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-12} dy={8} height={52} />
        <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
        <Tooltip formatter={v => `${v.toFixed(1)}%`} />
        <Legend />
        {models.map(m => (
          <Bar key={m.name} dataKey={m.name} name={m.label}
               fill={colorOf(m.name)} radius={[4, 4, 0, 0]} maxBarSize={46} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

export function LatencyChart({ table, models, techniques, limit }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={toSeries(table, models, techniques)}
                margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-12} dy={8} height={52} />
        <YAxis tick={{ fontSize: 11 }} unit="s" />
        <Tooltip formatter={v => `${v.toFixed(2)} s`} />
        <Legend />
        <ReferenceLine y={limit} stroke="#c62828" strokeDasharray="5 4"
                       label={{ value: `voice limit (${limit} s)`, position: 'insideTopRight',
                                fill: '#c62828', fontSize: 11 }} />
        {models.map(m => (
          <Bar key={m.name} dataKey={m.name} name={m.label}
               fill={colorOf(m.name)} radius={[4, 4, 0, 0]} maxBarSize={46} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

/** v1 -> v2 -> v3 as a line per model, so the tuning shows as a climb. */
export function TuningChart({ tuning, models }) {
  const data = tuning.versions.map((v, i) => {
    const row = { name: v.label }
    models.forEach(m => {
      const value = tuning.accuracy[m.name]?.[i]
      if (value != null) row[m.name] = value
    })
    return row
  })
  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" tick={{ fontSize: 12 }} />
        <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} unit="%" />
        <Tooltip formatter={v => `${v.toFixed(1)}%`} />
        <Legend />
        {models.map(m => (
          <Line key={m.name} type="monotone" dataKey={m.name} name={m.label}
                stroke={colorOf(m.name)} strokeWidth={2.5}
                dot={{ r: 4 }} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

export function MistakesTable({ rows }) {
  if (!rows?.length) return <div className="muted">No mistakes recorded yet.</div>
  return (
    <div className="pe-table-wrap">
      <table className="pe-table">
        <thead>
          <tr>
            <th>Command</th><th>Expected</th><th>AI answer</th>
            <th>Model</th><th>Technique</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.command}</td>
              <td><code>{r.expected}</code></td>
              <td><code>{r.answer}</code></td>
              <td>{r.model}</td>
              <td>{r.technique}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
