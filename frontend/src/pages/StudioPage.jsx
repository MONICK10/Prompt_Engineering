// Page 6 — VDOC Studio.
//
// Speak one Tanglish sentence, get a formatted document and a real .docx and
// .pdf. The LLM returns a JSON array of actions, Python applies them, and the
// preview paginates the result into A4 sheets by measuring what the browser
// drew — so what is on screen is what the exporters write.
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import '../demo/demo.css'
import '../demo/studio.css'
import PageView from '../demo/PageView'
import { flattenDoc } from '../demo/paginate'
import { useVoiceInput } from '../demo/useVoiceInput'
import {
  downloadUrl, studioCommand, studioPick, studioReset, studioState, studioUndo,
} from '../demo/studioApi'

const EXAMPLES = [
  'artificial intelligence pathi oru report create pannu',
  'page-ku border podu',
  'second paragraph highlight pannu',
  'title-a red color-la maathu',
  'atha bold pannu',
]

export default function StudioPage() {
  const [command, setCommand] = useState(EXAMPLES[0])
  const [doc, setDoc] = useState(null)
  const [log, setLog] = useState([])
  const [result, setResult] = useState(null)
  const [options, setOptions] = useState([])
  const [question, setQuestion] = useState(null)
  const [view, setView] = useState('page')      // 'page' | 'pdf'
  const [pdfStamp, setPdfStamp] = useState(0)
  const [touched, setTouched] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const previous = useRef(null)

  useEffect(() => {
    studioState().then(s => {
      if (!s.ok) return
      setDoc(s.doc); setLog(s.log || [])
      setOptions(s.options || []); setQuestion(s.question || null)
    })
  }, [])

  // Flash whatever changed, for two seconds.
  const flash = useCallback(nextDoc => {
    const before = previous.current
    previous.current = nextDoc
    if (!before || !nextDoc) return
    const old = new Map(flattenDoc(before).map(b => [b.key, JSON.stringify(b.run)]))
    const changed = flattenDoc(nextDoc)
      .filter(b => old.get(b.key) !== JSON.stringify(b.run))
      .map(b => b.key)
    setTouched(changed)
    if (changed.length) setTimeout(() => setTouched([]), 2000)
  }, [])

  const absorb = reply => {
    if (reply.doc) { flash(reply.doc); setDoc(reply.doc) }
    if (reply.log) setLog(reply.log)
    setOptions(reply.options ?? [])
    setQuestion(reply.question ?? null)
    setPdfStamp(Date.now())
  }

  const run = useCallback(async text => {
    const utterance = (text ?? command).trim()
    if (!utterance) return
    setBusy(true); setError(null)
    const reply = await studioCommand(utterance)
    setBusy(false)
    absorb(reply); setResult(reply)
    if (!reply.ok) setError(reply.error)
  }, [command])

  const choose = async index => {
    setBusy(true)
    const reply = await studioPick(index)
    setBusy(false)
    if (!reply.ok) { setError(reply.error); return }
    absorb(reply); setResult(reply)
  }

  const onTranscribed = useCallback(text => { setCommand(text); run(text) }, [run])
  const { recording, voiceStatus, toggleMic } = useVoiceInput(onTranscribed)

  const undo = async () => { const r = await studioUndo(); absorb(r); setResult(null) }
  const reset = async () => {
    const r = await studioReset()
    previous.current = null
    absorb(r); setResult(null); setError(null)
  }

  const page = doc?.page ?? {}
  const empty = !doc?.title?.text && !(doc?.sections ?? []).length
  const paragraphCount = useMemo(
    () => (doc?.sections ?? []).reduce((n, s) => n + (s.paragraphs?.length ?? 0), 0),
    [doc])

  return (
    <div className="studio-shell">
      <div className="studio-bar">
        <div className="studio-bar-row">
          <h2 className="studio-title">VDOC Studio</h2>
          <p className="studio-sub">Speak in Tanglish. Get a ready Word &amp; PDF document.</p>
        </div>

        <div className="studio-bar-row">
          <input
            className="command-input"
            value={command}
            onChange={e => setCommand(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') run() }}
            placeholder="Type or speak a command…"
            aria-label="Command"
          />
          <button className="run-btn" onClick={() => run()} disabled={busy}>
            {busy ? 'Working…' : 'Run'}
          </button>
          <button
            className={`mic-btn${recording ? ' recording' : ''}`}
            onClick={toggleMic}
            title={recording ? 'Stop recording' : 'Speak a command'}
            aria-label={recording ? 'Stop recording' : 'Speak a command'}
          >
            {recording ? '⏹' : '🎙'}
          </button>
          <button className="reset-btn" onClick={undo}>Undo</button>
          <button className="reset-btn" onClick={reset}>Reset</button>
          <div className="studio-toggle" role="group" aria-label="View">
            <button className={view === 'page' ? 'is-on' : ''}
                    onClick={() => setView('page')}>Page view</button>
            <button className={view === 'pdf' ? 'is-on' : ''}
                    onClick={() => { setPdfStamp(Date.now()); setView('pdf') }}>
              PDF view
            </button>
          </div>
        </div>

        <div className="studio-bar-row">
          <span className="example-label">Try:</span>
          {EXAMPLES.map(example => (
            <button key={example} className="example-chip"
                    onClick={() => { setCommand(example); run(example) }}>
              {example}
            </button>
          ))}
        </div>

        {voiceStatus && <div className="notice">{voiceStatus}</div>}
        {error && <div className="notice is-error">{error}</div>}

        {result && (
          <div className="studio-bar-row">
            <div className="studio-chips">
              {!result.valid_json && (
                <span className="studio-chip bad">AI reply was not valid JSON</span>
              )}
              {(result.messages ?? []).map((m, i) => (
                <span key={i} className={`studio-chip${m.ok ? '' : ' bad'}`}>
                  {m.ok ? '✓' : '✗'} {m.text}
                </span>
              ))}
              {result.latency_s > 0 && (
                <span className="studio-chip-time">({result.latency_s.toFixed(2)} s)</span>
              )}
            </div>
          </div>
        )}

        {question && (
          <div className="studio-ask">
            🤔 {question}
            <div className="studio-options">
              {options.map((option, i) => (
                <button key={i} className="studio-option" disabled={busy}
                        onClick={() => choose(i + 1)}>
                  <b>{i + 1}.</b>{option.label}
                </button>
              ))}
            </div>
            {options.length > 0 && (
              <div style={{ marginTop: 7, fontSize: 13 }}>
                Or say the number — “rendavathu”, “2” — or “vendam” to cancel.
              </div>
            )}
          </div>
        )}
      </div>

      <div className="studio-body">
        <div className="studio-docarea">
          {view === 'pdf'
            ? (
              <div className="studio-pdf-wrap">
                <iframe className="studio-pdf" title="Generated PDF"
                        src={`${downloadUrl('pdf')}&inline=1&t=${pdfStamp}#view=FitH`} />
              </div>
            )
            : empty
              ? <div className="studio-empty">Say “… pathi oru report create pannu” to begin.</div>
              : <PageView doc={doc} touched={touched} />}
        </div>

        <aside className="studio-side">
          <h4>Formatting</h4>
          <div className="studio-fact"><span>Font</span><span>{page.font}</span></div>
          <div className="studio-fact"><span>Size</span><span>{page.size} pt</span></div>
          <div className="studio-fact"><span>Border</span><span>{page.border}</span></div>
          <div className="studio-fact"><span>Line spacing</span><span>{page.line_spacing}</span></div>
          <div className="studio-fact"><span>Margins</span><span>{page.margins_cm} cm</span></div>
          <div className="studio-fact"><span>Sections</span><span>{(doc?.sections ?? []).length}</span></div>
          <div className="studio-fact"><span>Paragraphs</span><span>{paragraphCount}</span></div>

          <h4>Download</h4>
          <div className="studio-downloads">
            <a className={`studio-dl${empty ? ' is-off' : ''}`}
               href={downloadUrl('docx')}>Word</a>
            <a className={`studio-dl pdf${empty ? ' is-off' : ''}`}
               href={downloadUrl('pdf')}>PDF</a>
          </div>

          <h4>History</h4>
          <ol className="studio-log">
            {log.map((line, i) => <li key={i}>{line}</li>)}
          </ol>
        </aside>
      </div>
    </div>
  )
}
