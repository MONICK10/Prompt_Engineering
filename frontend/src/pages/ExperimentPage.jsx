import { useState, useCallback } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  Cell, ResponsiveContainer, LabelList,
} from 'recharts'

const API              = 'http://localhost:5050'
const BATCH_SIZE       = 50
const GROQ_BATCH_SIZE  = 5
const GROQ_INTER_DELAY = 2000

const SYSTEMS = [
  { key: 'rule',        label: 'Rule-based',  color: '#ef5350' },
  { key: 'bilstm',      label: 'BiLSTM',      color: '#ff9800' },
  { key: 'doceditbert', label: 'DocEditBERT', color: '#43a047' },
  { key: 'groq',        label: 'Groq LLM',    color: '#7c3aed' },
]

// Extra baselines shown in all panels (sent for every run)
const EXTRA_SYSTEMS = [
  { key: 'xlmr_zero',   label: 'XLM-R Zero-shot',        color: '#0097a7' },
  { key: 'mdistilbert', label: 'mDistilBERT (Tanglish-only)', color: '#e64a19' },
]

const RUNS = [
  {
    key:     'english',
    label:   'Run English 500',
    dataset: 'docEdit500',
    total:   500,
    title:   'English Commands — DocEdit-500',
    accent:  '#1565c0',
    note:    '500 standard English voice commands',
  },
  {
    key:     'tanglish',
    label:   'Run Tanglish 500',
    dataset: 'docEdit_tanglish_500',
    total:   500,
    title:   'Tanglish Commands — DocEdit-Tanglish-500',
    accent:  '#6a1b9a',
    note:    '500 Tamil-English code-switched commands ★ Novel dataset',
  },
  {
    key:     'combined',
    label:   'Run All 1000',
    dataset: 'docEdit1000',
    total:   1000,
    title:   'Combined — English + Tanglish (1000)',
    accent:  '#1b5e20',
    note:    '1000 commands across both languages',
  },
]

const initRunState = () => ({
  running:    false,
  progress:   0,
  total:      0,
  partial:    null,
  done:       null,
  groqStatus: '',
  error:      '',
})

// ── Helpers ───────────────────────────────────────────────────────────────────
function isCorrect(pred, gt) {
  if (!pred || !gt.action) return null
  return (
    (pred.action || '').toLowerCase() === gt.action.toLowerCase() &&
    (pred.from   || '').toLowerCase() === (gt.from || '').toLowerCase() &&
    (pred.to     || '').toLowerCase() === (gt.to   || '').toLowerCase()
  )
}

function pct(v) { return v == null ? '—' : `${(v * 100).toFixed(1)}%` }

function exportCSV(details, filename = 'vdoc_results.csv') {
  const header = [
    'id', 'text', 'gt_action', 'gt_from', 'gt_to',
    'rule_action', 'rule_from', 'rule_to', 'rule_correct',
    'bilstm_action', 'bilstm_from', 'bilstm_to', 'bilstm_correct',
    'bert_action', 'bert_from', 'bert_to', 'bert_correct',
  ].join(',')

  const rows = details.map((d, i) => {
    const gt = d.ground_truth
    const r  = d.rule        || {}
    const b  = d.bilstm      || {}
    const be = d.doceditbert || {}
    const ok = (p) => (
      (p.action||'').toLowerCase() === (gt.intent||'').toLowerCase() &&
      (p.from  ||'').toLowerCase() === (gt.from_anchor||'').toLowerCase() &&
      (p.to    ||'').toLowerCase() === (gt.to_anchor  ||'').toLowerCase()
    ) ? 1 : 0
    return [
      i + 1,
      `"${d.text.replace(/"/g, '""')}"`,
      gt.intent, gt.from_anchor, gt.to_anchor,
      r.action||'', r.from||'', r.to||'', ok(r),
      b.action||'', b.from||'', b.to||'', ok(b),
      be.action||'', be.from||'', be.to||'', ok(be),
    ].join(',')
  })

  const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

// ── Sub-components ────────────────────────────────────────────────────────────
function SystemCard({ label, color, result, gt }) {
  const ok = isCorrect(result, gt)
  return (
    <div style={{
      flex: 1, minWidth: 175,
      border: `1.5px solid ${color}33`,
      borderRadius: 10, padding: '14px 16px',
      background: '#fff', boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
    }}>
      <div style={{
        fontSize: 11, fontWeight: 800, color,
        textTransform: 'uppercase', letterSpacing: 1,
        borderBottom: `1px solid ${color}22`, paddingBottom: 8, marginBottom: 10,
      }}>
        {label}
      </div>
      {result ? (
        <>
          {[
            ['action', result.action],
            ['from',   result.from],
            ['to',     result.to],
            ['conf',   result.confidence != null ? result.confidence.toFixed(4) : '—'],
          ].map(([k, v]) => (
            <div key={k} style={{ display: 'flex', gap: 8, marginBottom: 5, alignItems: 'baseline' }}>
              <span style={{ fontSize: 11, color: '#aaa', width: 38, flexShrink: 0 }}>{k}</span>
              <span style={{ fontSize: 13, fontWeight: 600, color: v ? '#222' : '#ccc' }}>{v || '—'}</span>
            </div>
          ))}
          {ok !== null && (
            <div style={{
              marginTop: 10, padding: '4px 0', textAlign: 'center',
              borderRadius: 6, fontWeight: 700, fontSize: 12,
              background: ok ? '#e8f5e9' : '#ffebee',
              color:      ok ? '#2e7d32' : '#c62828',
            }}>
              {ok ? 'CORRECT' : 'WRONG'}
            </div>
          )}
        </>
      ) : (
        <div style={{ color: '#ccc', fontSize: 13 }}>—</div>
      )}
    </div>
  )
}

function AccBar({ label, color, value }) {
  const pctVal = value == null ? 0 : value * 100
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: '#333' }}>{label}</span>
        <span style={{ fontSize: 12, fontWeight: 700, color }}>{pct(value)}</span>
      </div>
      <div style={{ background: '#ebebeb', borderRadius: 6, height: 10, overflow: 'hidden' }}>
        <div style={{
          width: `${pctVal}%`, height: '100%',
          background: color, borderRadius: 6,
          transition: 'width 0.6s ease',
        }} />
      </div>
    </div>
  )
}

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div style={{
      background: '#fff', border: '1px solid #e0e0e0',
      borderRadius: 8, padding: '8px 14px', fontSize: 13,
      boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
    }}>
      <strong style={{ color: d.color }}>{d.name}</strong>
      <div>{pct(d.value / 100)}</div>
    </div>
  )
}

// ── Per-run result panel ──────────────────────────────────────────────────────
function RunResultPanel({ run, state, includeGroq }) {
  const displayAcc = state.done ?? state.partial
  if (!state.running && !displayAcc && !state.error) return null

  const isTanglish = run.key === 'tanglish'

  const extraForRun = isTanglish
    ? EXTRA_SYSTEMS.filter(s => s.key === 'xlmr_zero')   // Tanglish: XLM-R only, no mDistilBERT
    : EXTRA_SYSTEMS                                        // English/Combined: both
  const activeSystems = [
    ...SYSTEMS.filter(s => s.key !== 'groq' || (displayAcc && displayAcc[s.key] != null)),
    ...extraForRun.filter(s => displayAcc && displayAcc[s.key] != null),
  ]
  const chartData = displayAcc
    ? activeSystems.map(s => ({
        name:  s.label,
        value: parseFloat(((displayAcc[s.key] ?? 0) * 100).toFixed(1)),
        color: s.color,
      }))
    : []

  return (
    <div style={{
      marginTop: 20,
      borderLeft: `4px solid ${run.accent}`,
      paddingLeft: 18,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 14, fontWeight: 800, color: run.accent }}>
          {run.title}
        </span>
        <span style={{ fontSize: 11, color: '#999' }}>{run.note}</span>
        {state.done && (
          <span style={{
            marginLeft: 'auto', fontSize: 11, padding: '2px 10px',
            background: run.accent + '18', color: run.accent,
            borderRadius: 10, fontWeight: 700,
          }}>
            {state.done.total} commands
          </span>
        )}
      </div>

      {/* Error */}
      {state.error && (
        <div style={{
          padding: '8px 12px', background: '#ffebee', color: '#c62828',
          borderRadius: 6, fontSize: 12, marginBottom: 12,
        }}>
          {state.error}
        </div>
      )}

      {/* Progress bar */}
      {(state.running || state.done) && (
        <div style={{ marginBottom: 14 }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            fontSize: 11, color: '#888', marginBottom: 3,
          }}>
            <span>{state.done ? 'Complete' : 'Processing...'}</span>
            <span>{Math.min(state.progress, state.total)} / {state.total}</span>
          </div>
          <div style={{ background: '#e0e0e0', borderRadius: 8, height: 8, overflow: 'hidden' }}>
            <div style={{
              width: `${(Math.min(state.progress, state.total) / (state.total || 1)) * 100}%`,
              height: '100%', background: run.accent, borderRadius: 8,
              transition: 'width 0.4s ease',
            }} />
          </div>
          {state.groqStatus && (
            <div style={{ fontSize: 10, color: '#7c3aed', marginTop: 5 }}>
              {state.groqStatus}
            </div>
          )}
        </div>
      )}

      {/* Accuracy bars */}
      {displayAcc && (
        <div style={{
          background: '#fafafa', borderRadius: 8,
          padding: '12px 16px', marginBottom: 14,
          border: '1px solid #f0f0f0',
        }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#888', marginBottom: 10, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            {state.done ? 'Final Accuracy' : 'Running Accuracy'}
          </div>
          {activeSystems.map(s => (
            <AccBar key={s.key} label={s.label} color={s.color} value={displayAcc[s.key]} />
          ))}
        </div>
      )}

      {/* Bar chart */}
      {state.done && chartData.length > 0 && (
        <div style={{
          background: '#fafafa', borderRadius: 8,
          padding: '14px 16px 8px', marginBottom: 14,
          border: '1px solid #f0f0f0',
        }}>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={chartData} margin={{ top: 12, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eeeeee" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} tickFormatter={v => `${v}%`} />
              <Tooltip content={<ChartTooltip />} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={72}>
                {chartData.map((d, i) => <Cell key={i} fill={d.color} />)}
                <LabelList
                  dataKey="value"
                  position="top"
                  formatter={v => `${v}%`}
                  style={{ fontSize: 11, fontWeight: 700, fill: '#333' }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Summary table + export */}
      {state.done && (
        <div style={{
          background: '#fafafa', borderRadius: 8,
          padding: '12px 16px', marginBottom: 4,
          border: '1px solid #f0f0f0',
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <span style={{ fontSize: 11, fontWeight: 700, color: '#888', textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Results
            </span>
            <button
              onClick={() => exportCSV(state.done.details, `vdoc_${run.key}_results.csv`)}
              style={{
                fontSize: 11, padding: '4px 12px',
                border: `1px solid ${run.accent}`, borderRadius: 5,
                background: '#fff', color: run.accent,
                fontWeight: 700, cursor: 'pointer',
              }}
            >
              Export CSV
            </button>
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1.5px solid #e8e8e8' }}>
                {['System', 'Correct', 'Total', 'Accuracy'].map(h => (
                  <th key={h} style={{
                    padding: '5px 10px', textAlign: 'left',
                    fontSize: 10, color: '#aaa', fontWeight: 700,
                    textTransform: 'uppercase', letterSpacing: 0.5,
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {activeSystems.map(s => {
                const acc     = state.done[s.key] ?? 0
                const correct = Math.round(acc * state.done.total)
                return (
                  <tr key={s.key} style={{ borderBottom: '1px solid #f4f4f4' }}>
                    <td style={{ padding: '7px 10px' }}>
                      <span style={{
                        display: 'inline-block', width: 8, height: 8,
                        borderRadius: '50%', background: s.color, marginRight: 7,
                      }} />
                      <strong>{s.label}</strong>
                    </td>
                    <td style={{ padding: '7px 10px', color: '#333' }}>{correct}</td>
                    <td style={{ padding: '7px 10px', color: '#aaa' }}>{state.done.total}</td>
                    <td style={{ padding: '7px 10px' }}>
                      <strong style={{ color: s.color, fontSize: 13 }}>{pct(acc)}</strong>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ── Section wrapper ───────────────────────────────────────────────────────────
function Section({ title, subtitle, children }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12,
      padding: '22px 24px', marginBottom: 24,
      boxShadow: '0 2px 12px rgba(0,0,0,0.07)',
    }}>
      <div style={{ marginBottom: 18 }}>
        <h2 style={{ fontSize: 16, fontWeight: 800, color: '#1a1a1a', marginBottom: 3 }}>
          {title}
        </h2>
        <p style={{ fontSize: 13, color: '#888' }}>{subtitle}</p>
      </div>
      {children}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function ExperimentPage() {
  // Section A
  const [cmd,           setCmd]           = useState('')
  const [gt,            setGt]            = useState({ action: '', from: '', to: '' })
  const [singleResult,  setSingleResult]  = useState(null)
  const [singleLoading, setSingleLoading] = useState(false)
  const [singleError,   setSingleError]   = useState('')

  // Section B — independent state per dataset run
  const [engState,  setEngState]  = useState(initRunState())
  const [tangState, setTangState] = useState(initRunState())
  const [combState, setCombState] = useState(initRunState())
  const [includeGroq, setIncludeGroq] = useState(false)

  const stateMap  = { english: engState,  tanglish: tangState,  combined: combState }
  const setterMap = { english: setEngState, tanglish: setTangState, combined: setCombState }
  const anyRunning = engState.running || tangState.running || combState.running

  // ── Section A handler ─────────────────────────────────────────────────────
  const runSingle = useCallback(async () => {
    if (!cmd.trim()) return
    setSingleLoading(true)
    setSingleError('')
    setSingleResult(null)
    try {
      const { data } = await axios.post(`${API}/parse-all`, { command: cmd.trim() })
      setSingleResult(data)
    } catch (err) {
      setSingleError(
        err.code === 'ERR_NETWORK'
          ? 'Cannot reach backend at localhost:5050 — is the server running?'
          : err.response?.data?.error ?? err.message,
      )
    } finally {
      setSingleLoading(false)
    }
  }, [cmd])

  // ── Section B batch runner (parameterised by run key) ─────────────────────
  const runBatch = async (runKey) => {
    const run    = RUNS.find(r => r.key === runKey)
    const setter = setterMap[runKey]

    const isTanglish = runKey === 'tanglish'
    const useGroq    = includeGroq || isTanglish   // Groq always on for Tanglish

    setter(() => ({
      ...initRunState(),
      running:    true,
      total:      run.total,
      groqStatus: useGroq ? (isTanglish ? 'Groq auto-enabled for Tanglish — may take a few minutes.' : 'Groq enabled — this may take several minutes.') : '',
    }))

    let dataset
    try {
      const res = await fetch(`/${run.dataset}.json`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      dataset = await res.json()
    } catch (err) {
      setter(s => ({ ...s, running: false, error: `Could not load ${run.dataset}.json: ${err.message}` }))
      return
    }

    setter(s => ({ ...s, total: dataset.length }))

    const batchSize = useGroq ? GROQ_BATCH_SIZE : BATCH_SIZE
    let ruleSum = 0, bilstmSum = 0, bertSum = 0, groqSum = 0
    let xlmrZeroSum = 0, mdistilbertSum = 0
    const allDetails = []
    let processed = 0

    for (let start = 0; start < dataset.length; start += batchSize) {
      const batch = dataset.slice(start, start + batchSize).map(ex => ({
        text:        ex.text,
        intent:      ex.intent,
        from_anchor: ex.from_anchor,
        to_anchor:   ex.to_anchor,
      }))

      if (useGroq) {
        const p = Math.round((start / dataset.length) * 100)
        setter(s => ({
          ...s,
          groqStatus: `Groq: commands ${start + 1}–${start + batch.length} (${p}% done)…`,
        }))
      }

      try {
        const { data } = await axios.post(`${API}/batch-evaluate`, {
          commands:            batch,
          include_groq:        useGroq,
          include_xlmr_zero:   true,
          include_mdistilbert: !isTanglish,   // mDistilBERT only for English/Combined
        })
        ruleSum   += data.rule_accuracy        * batch.length
        bilstmSum += data.bilstm_accuracy      * batch.length
        bertSum   += data.doceditbert_accuracy * batch.length
        if (useGroq       && data.groq_accuracy        != null) groqSum       += data.groq_accuracy        * batch.length
        if (data.xlmr_zero_accuracy   != null) xlmrZeroSum    += data.xlmr_zero_accuracy   * batch.length
        if (!isTanglish   && data.mdistilbert_accuracy != null) mdistilbertSum += data.mdistilbert_accuracy * batch.length
        allDetails.push(...(data.details || []))
      } catch { /* batch failed — keeps previous sums, contributes 0 */ }

      processed += batch.length
      setter(s => ({
        ...s,
        progress: processed,
        partial: {
          rule:        ruleSum   / processed,
          bilstm:      bilstmSum / processed,
          doceditbert: bertSum   / processed,
          ...(useGroq      ? { groq:        groqSum        / processed } : {}),
          xlmr_zero:          xlmrZeroSum    / processed,
          ...(!isTanglish  ? { mdistilbert: mdistilbertSum / processed } : {}),
        },
      }))

      if (useGroq && processed < dataset.length)
        await new Promise(r => setTimeout(r, GROQ_INTER_DELAY))
    }

    setter(s => ({
      ...s,
      running:    false,
      groqStatus: useGroq ? 'Groq evaluation complete' : '',
      done: {
        rule:        ruleSum   / dataset.length,
        bilstm:      bilstmSum / dataset.length,
        doceditbert: bertSum   / dataset.length,
        ...(useGroq     ? { groq:        groqSum        / dataset.length } : {}),
        xlmr_zero:          xlmrZeroSum    / dataset.length,
        ...(!isTanglish ? { mdistilbert: mdistilbertSum / dataset.length } : {}),
        total:   dataset.length,
        details: allDetails,
      },
    }))
  }

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{
      height: '100%', overflowY: 'auto',
      background: '#f5f5f5', padding: '28px 20px 80px',
    }}>
      <div style={{ maxWidth: 860, margin: '0 auto' }}>

        {/* ══ Section A ════════════════════════════════════════════════════ */}
        <Section
          title="A — Single Command Tester"
          subtitle="Type a command and see how each system parses it."
        >
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            <input
              value={cmd}
              onChange={e => setCmd(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && runSingle()}
              placeholder='e.g. "bold from car to accident"'
              style={{
                flex: 1, padding: '9px 14px', borderRadius: 7,
                border: '1.5px solid #ddd', fontSize: 14,
                outline: 'none', fontFamily: 'inherit',
              }}
            />
            <button
              onClick={runSingle}
              disabled={singleLoading || !cmd.trim()}
              style={{
                padding: '9px 20px', borderRadius: 7, border: 'none',
                background: '#3a5fff', color: '#fff', fontWeight: 700,
                fontSize: 14, cursor: 'pointer', opacity: singleLoading ? 0.6 : 1,
                whiteSpace: 'nowrap',
              }}
            >
              {singleLoading ? 'Testing...' : 'Test All Systems'}
            </button>
          </div>

          <div style={{
            display: 'flex', gap: 8, alignItems: 'center',
            marginBottom: 18, flexWrap: 'wrap',
          }}>
            <span style={{ fontSize: 12, color: '#888', fontWeight: 600 }}>Ground truth:</span>
            {['action', 'from', 'to'].map(field => (
              <input
                key={field}
                value={gt[field]}
                onChange={e => setGt(p => ({ ...p, [field]: e.target.value }))}
                placeholder={field}
                style={{
                  width: field === 'action' ? 110 : 130,
                  padding: '6px 10px', borderRadius: 6,
                  border: '1px solid #ddd', fontSize: 13, fontFamily: 'inherit',
                }}
              />
            ))}
            <span style={{ fontSize: 11, color: '#bbb' }}>(optional — fill to see pass/fail)</span>
          </div>

          {singleError && (
            <div style={{
              padding: '8px 12px', background: '#ffebee', color: '#c62828',
              borderRadius: 6, fontSize: 13, marginBottom: 14,
            }}>
              {singleError}
            </div>
          )}

          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {SYSTEMS.map(s => (
              <SystemCard key={s.key} label={s.label} color={s.color}
                result={singleResult?.[s.key]} gt={gt} />
            ))}
          </div>

          <div style={{ marginTop: 16 }}>
            <span style={{ fontSize: 12, color: '#aaa', marginRight: 8 }}>Try:</span>
            {[
              'bold from car to accident',
              'italic from introduction to conclusion',
              'car la irunthu accident varikum bold panu',
              'car ah vehicle aa maathu',
            ].map(ex => (
              <button
                key={ex}
                onClick={() => setCmd(ex)}
                style={{
                  fontSize: 11, padding: '3px 8px', marginRight: 6, marginBottom: 4,
                  border: '1px solid #ddd', borderRadius: 4,
                  background: '#fafafa', cursor: 'pointer', color: '#555',
                }}
              >
                {ex}
              </button>
            ))}
          </div>
        </Section>

        {/* ══ Section B ════════════════════════════════════════════════════ */}
        <Section
          title="B — Batch Accuracy Evaluation"
          subtitle="Run a full dataset through all four systems. Each button is independent — run one, all three, or compare them side by side."
        >
          {/* ── Three run buttons + Groq toggle ── */}
          <div style={{
            display: 'flex', gap: 10, alignItems: 'center',
            flexWrap: 'wrap', marginBottom: 8,
          }}>
            {RUNS.map(run => {
              const state = stateMap[run.key]
              const busy  = state.running
              return (
                <button
                  key={run.key}
                  onClick={() => !anyRunning && runBatch(run.key)}
                  disabled={anyRunning}
                  style={{
                    padding: '10px 20px', borderRadius: 7, border: 'none',
                    background: anyRunning && !busy ? '#bdbdbd' : run.accent,
                    color: '#fff', fontWeight: 700, fontSize: 14,
                    cursor: anyRunning ? 'not-allowed' : 'pointer',
                    opacity: anyRunning && !busy ? 0.55 : 1,
                    minWidth: 150,
                  }}
                >
                  {busy
                    ? `Running… ${state.progress} / ${state.total}`
                    : run.label}
                </button>
              )
            })}

            <label style={{
              display: 'flex', alignItems: 'center', gap: 7, cursor: 'pointer',
              fontSize: 13, fontWeight: 600, color: '#7c3aed',
              padding: '8px 14px', border: '1.5px solid #7c3aed33',
              borderRadius: 7, background: includeGroq ? '#f3f0ff' : '#fff',
              marginLeft: 'auto',
            }}>
              <input
                type="checkbox"
                checked={includeGroq}
                onChange={e => setIncludeGroq(e.target.checked)}
                disabled={anyRunning}
                style={{ accentColor: '#7c3aed', width: 15, height: 15 }}
              />
              Include Groq LLM
            </label>
          </div>

          {/* hint when nothing has run yet */}
          {!anyRunning && !engState.done && !tangState.done && !combState.done && (
            <p style={{ fontSize: 12, color: '#bbb', marginTop: 4, marginBottom: 0 }}>
              Click a button above to start evaluation. Results appear below as they run.
            </p>
          )}

          {/* ── Per-run result panels ── */}
          {RUNS.map(run => (
            <RunResultPanel
              key={run.key}
              run={run}
              state={stateMap[run.key]}
              includeGroq={includeGroq}
            />
          ))}

          {/* ── Cross-run comparison table (visible once 2+ runs complete) ── */}
          {[engState, tangState, combState].filter(s => s.done).length >= 2 && (
            <div style={{
              marginTop: 28,
              background: '#f0f4ff', borderRadius: 10,
              padding: '16px 20px',
              border: '1px solid #c7d4f8',
            }}>
              <div style={{ fontSize: 13, fontWeight: 800, color: '#1a1a1a', marginBottom: 14 }}>
                Cross-Dataset Comparison
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #c7d4f8' }}>
                    <th style={{ padding: '6px 10px', textAlign: 'left', color: '#888', fontWeight: 700, fontSize: 10, textTransform: 'uppercase' }}>System</th>
                    {RUNS.filter(r => stateMap[r.key].done).map(r => (
                      <th key={r.key} style={{ padding: '6px 10px', textAlign: 'center', color: r.accent, fontWeight: 800, fontSize: 11 }}>
                        {r.key === 'english' ? 'English 500' : r.key === 'tanglish' ? 'Tanglish 500' : 'Combined 1000'}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...SYSTEMS, ...EXTRA_SYSTEMS].map(s => {
                    const doneRuns = RUNS.filter(r => stateMap[r.key].done && stateMap[r.key].done[s.key] != null)
                    if (doneRuns.length === 0) return null
                    return (
                      <tr key={s.key} style={{ borderBottom: '1px solid #dce6ff' }}>
                        <td style={{ padding: '8px 10px' }}>
                          <span style={{
                            display: 'inline-block', width: 8, height: 8,
                            borderRadius: '50%', background: s.color, marginRight: 7,
                          }} />
                          <strong>{s.label}</strong>
                          {s.key === 'xlmr_zero' && (
                            <span style={{ fontSize: 9, color: '#aaa', marginLeft: 5 }}>zero-shot</span>
                          )}
                          {s.key === 'mdistilbert' && (
                            <span style={{ fontSize: 9, color: '#aaa', marginLeft: 5 }}>Tanglish-trained</span>
                          )}
                        </td>
                        {RUNS.filter(r => stateMap[r.key].done).map(r => {
                          const val = stateMap[r.key].done[s.key]
                          return (
                            <td key={r.key} style={{ padding: '8px 10px', textAlign: 'center' }}>
                              <strong style={{ color: s.color, fontSize: 13 }}>
                                {val != null ? pct(val) : '—'}
                              </strong>
                            </td>
                          )
                        })}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Section>
      </div>
    </div>
  )
}
