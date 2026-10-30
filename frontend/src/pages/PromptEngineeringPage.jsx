// Page 5 — Prompt Engineering.
//
// The other pages answer a command with DocEditBERT, a model we trained. This
// page answers the same command with a general LLM and a written prompt, and
// shows what changing only the prompt does. It edits the same demo document
// through the same executor, so the two paths are genuinely comparable.
import { useCallback, useEffect, useState } from 'react'
import '../demo/demo.css'
import '../demo/pe.css'
import { createDoc, runActions, SAMPLE_LINES } from '../demo/document'
import { parseMulti } from '../demo/api'
import { fetchPeMeta, peCompare, peResults, peRun } from '../demo/peApi'
import { useVoiceInput } from '../demo/useVoiceInput'
import { DocumentView } from '../demo/components'
import {
  AccuracyChart, AccuracyTable, BertCard, CompareCard, FlowDiagram,
  LatencyChart, MistakesTable, ReplyBlock, Step, TuningChart,
} from '../demo/peComponents'

const PRESETS = [
  'third line reported a practice maathu',
  '3rd line-la reported-a car-nu maathu',
  'kadaisi paragraph-a eduthudu',
  'atha delete pannu',
  'change the heading to bold',
]

export default function PromptEngineeringPage() {
  const [meta, setMeta] = useState(null)
  const [command, setCommand] = useState(PRESETS[1])
  const [model, setModel] = useState('groq')
  const [technique, setTechnique] = useState('p4_few_shot_v3')

  const [doc, setDoc] = useState(() => createDoc())
  const [before, setBefore] = useState(null)
  const [touched, setTouched] = useState([])
  const [result, setResult] = useState(null)
  const [applyNote, setApplyNote] = useState(null)

  const [cards, setCards] = useState(null)      // { results, bert }
  const [data, setData] = useState(null)        // /pe/results
  const [busy, setBusy] = useState(false)
  const [comparing, setComparing] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchPeMeta().then(m => {
      setMeta(m)
      if (m.ok) { setModel(m.default_model); setTechnique(m.default_technique) }
    })
    peResults().then(setData)
  }, [])

  // ── Section 3: run one command and apply it to the document ──────────────
  const run = useCallback(async text => {
    const utterance = (text ?? command).trim()
    if (!utterance) return
    setBusy(true); setError(null); setApplyNote(null)

    const reply = await peRun(utterance, model, technique)
    setBusy(false)
    if (!reply.ok) { setError(reply.error); return }
    setResult(reply)

    const startDoc = createDoc()
    if (!reply.action) {
      // Still show the document, unchanged, so the before/after panel never
      // disappears just because the model gave something we cannot run.
      setBefore(startDoc); setDoc(startDoc); setTouched([])
      setApplyNote(reply.note ?? 'Nothing was applied to the document.')
      return
    }
    const { doc: after, steps, touched: hit } = runActions(startDoc, [reply.action])
    setBefore(startDoc); setDoc(after); setTouched(hit)
    setApplyNote(steps[0]?.message ?? null)
  }, [command, model, technique])

  const onTranscribed = useCallback(text => { setCommand(text); run(text) }, [run])
  const { recording, voiceStatus, toggleMic } = useVoiceInput(onTranscribed)

  const reset = () => {
    setDoc(createDoc()); setBefore(null); setTouched([])
    setResult(null); setApplyNote(null); setError(null)
  }

  // ── Section 4: three prompts, plus the trained model ─────────────────────
  const compare = useCallback(async text => {
    const utterance = (text ?? command).trim()
    if (!utterance) return
    setComparing(true); setError(null)
    const [prompts, bert] = await Promise.all([
      peCompare(utterance, model),
      parseMulti(utterance, createDoc().length),
    ])
    setComparing(false)
    if (!prompts.ok) { setError(prompts.error); return }
    setCards({ results: prompts.results, bert: bert.ok ? bert : null })
  }, [command, model])

  const removed = before ? before.length - doc.length : 0
  const models = data?.models ?? []
  const techniques = data?.techniques ?? []

  return (
    <div className="demo-page">
      {/* ── 1. What is this project? ── */}
      <Step n={1} title="What is this project?">
        <p className="pe-lead">
          VDOC understands Tanglish voice commands. Here, instead of a trained
          model, we use an <strong>LLM</strong> and only change the{' '}
          <strong>prompt</strong>. We compare prompts and pick the best one.
        </p>
        {meta?.ok && <span className="pe-sample">{meta.test_commands} test commands</span>}
        {meta && !meta.ok && <div className="notice is-error">{meta.error}</div>}
      </Step>

      {/* ── 2. How it works ── */}
      <Step n={2} title="How it works">
        <FlowDiagram />
      </Step>

      {/* ── 3. Try it live ── */}
      <Step n={3} title="Try it live">
        <div className="demo-bar" style={{ position: 'static', marginBottom: 12 }}>
          <input
            className="command-input"
            value={command}
            onChange={e => setCommand(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') run() }}
            placeholder="Type a command, or press the mic and speak it"
            aria-label="Command"
          />
          <button className="run-btn" onClick={() => run()} disabled={busy}>
            {busy ? 'Asking…' : 'Run'}
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

        <div className="demo-bar" style={{ position: 'static', marginBottom: 12 }}>
          <span className="pe-select-label">Model</span>
          <select className="pe-select" value={model} onChange={e => setModel(e.target.value)}>
            {(meta?.models ?? []).map(m => (
              <option key={m.name} value={m.name}>{m.label}</option>
            ))}
          </select>
          <span className="pe-select-label">Technique</span>
          <select className="pe-select" value={technique} onChange={e => setTechnique(e.target.value)}>
            {(meta?.techniques ?? []).map(t => (
              <option key={t.name} value={t.name}>{t.label}</option>
            ))}
          </select>
        </div>

        {voiceStatus && <div className="notice">{voiceStatus}</div>}
        {error && <div className="notice is-error">{error}</div>}

        <div className="pe-field-label">Heard</div>
        <div className="heard-text">{result ? `“${result.command}”` : '—'}</div>

        <ReplyBlock result={result} />

        {applyNote && <div className="step-note" style={{ marginTop: 10 }}>{applyNote}</div>}

        <div className="doc-pair" style={{ marginTop: 14 }}>
          <DocumentView title="Before" doc={before ?? createDoc(SAMPLE_LINES)} />
          <DocumentView title="After" doc={doc} touched={touched}
                        removedCount={removed > 0 ? removed : 0} />
        </div>
      </Step>

      {/* ── 4. Same command, different prompts ── */}
      <Step
        n={4}
        title="Same command, different prompts"
        note="The same words go to the same model three times. Only the prompt changes."
      >
        <div className="example-row">
          <span className="example-label">Try:</span>
          {PRESETS.map(preset => (
            <button key={preset} className="example-chip"
                    onClick={() => { setCommand(preset); compare(preset) }}>
              {preset}
            </button>
          ))}
        </div>
        <button className="run-btn" onClick={() => compare()} disabled={comparing}
                style={{ marginBottom: 12 }}>
          {comparing ? 'Comparing…' : 'Compare prompts'}
        </button>

        {cards && (
          <div className="pe-cards">
            {cards.results.map(r => <CompareCard key={r.technique} result={r} />)}
            <BertCard bert={cards.bert} />
          </div>
        )}
      </Step>

      {/* ── 5. Results ── */}
      <Step n={5} title="Results">
        {!data?.ok && <div className="notice is-error">{data?.error ?? 'Loading…'}</div>}
        {data?.ok && !data.ready && (
          <div className="notice">
            No saved results yet. Run the experiment first:{' '}
            <code>python -m src.evaluate --models groq</code> inside vdoc-prompt.
          </div>
        )}
        {data?.ok && data.ready && (
          <>
            <div className="pe-field-label">Accuracy — model × technique</div>
            <AccuracyTable table={data.accuracy} models={models} techniques={techniques} />

            <div className="pe-field-label">Accuracy by technique</div>
            <AccuracyChart table={data.accuracy} models={models} techniques={techniques} />

            <div className="pe-field-label">Speed by technique</div>
            <LatencyChart table={data.latency} models={models} techniques={techniques}
                          limit={data.voice_limit_s} />

            <div className="pe-field-label">Prompt tuning: v1 → v2 → v3</div>
            <TuningChart tuning={data.tuning} models={models} />
            <div className="pe-chart-note">
              {data.tuning.versions.map(v => (
                <div key={v.name}><strong>{v.label}:</strong> {v.note}</div>
              ))}
            </div>

            <div className="pe-field-label">Key findings</div>
            <ul className="pe-list">
              {data.findings.map((f, i) => <li key={i}>{f}</li>)}
            </ul>

            <div className="pe-field-label">Where it went wrong</div>
            <MistakesTable rows={data.mistakes} />
          </>
        )}
      </Step>

      {/* ── 6. Limits ── */}
      <Step n={6} title="Limits">
        <ul className="pe-list">
          <li>
            <strong>Gemini</strong> allows only 20 requests a day on the free
            tier, so its results are partial. Empty cells say “not run”, which
            is not the same as zero.
          </li>
          <li>
            <strong>Whisper</strong> can split Tanglish words. “reported-a”
            often comes back as “reported a”, and the missing marker changes
            what the model thinks the command is.
          </li>
          <li>
            <strong>Sample size</strong>: {data?.test_commands ?? meta?.test_commands ?? '—'}{' '}
            test commands. Small differences between two prompts may be noise.
          </li>
          <li>
            The tick on a card means the reply was usable JSON, not that the
            answer was right. Accuracy is measured in the Results section.
          </li>
        </ul>
      </Step>
    </div>
  )
}
