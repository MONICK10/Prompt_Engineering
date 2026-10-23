// Page 2 — Hard Cases.
//
// Every row runs the real pipeline against the real backend and compares the
// result with what the case demands. Nothing here is a stored screenshot: if
// the backend is down, the rows say so rather than showing a stale pass.
import { useCallback, useState } from 'react'
import '../demo/demo.css'
import { createDoc, runActions } from '../demo/document'
import { parseMulti, parseOld } from '../demo/api'
import {
  HARD_CASES, checkCase, describeActual, describeExpected,
} from '../demo/hardCases'

export default function HardCasesPage() {
  const [results, setResults] = useState({})   // id -> result
  const [running, setRunning] = useState(null) // id | 'all' | null
  const [compareOld, setCompareOld] = useState(true)

  const runCase = useCallback(async testCase => {
    const startDoc = createDoc()

    const parsed = await parseMulti(testCase.command, startDoc.length)
    if (!parsed.ok) return { error: parsed.error }

    const verdict = checkCase(testCase, parsed.actions)
    const { steps } = runActions(startDoc, parsed.actions)

    let old = null
    if (compareOld) {
      const oldParsed = await parseOld(testCase.command)
      if (oldParsed.ok) {
        const oldRun = runActions(createDoc(), oldParsed.actions)
        const oldVerdict = checkCase(testCase, oldParsed.actions)
        old = {
          actions: oldParsed.actions,
          pass: oldVerdict.pass,
          applied: oldRun.steps.filter(s => s.ok).length,
        }
      }
    }

    return {
      ...verdict,
      actions: parsed.actions,
      chunks: parsed.chunks,
      steps,
      applied: steps.filter(s => s.ok).length,
      old,
      ranAt: new Date().toLocaleTimeString(),
    }
  }, [compareOld])

  const runOne = useCallback(async testCase => {
    setRunning(testCase.id)
    const result = await runCase(testCase)
    setResults(prev => ({ ...prev, [testCase.id]: result }))
    setRunning(null)
  }, [runCase])

  const runAll = useCallback(async () => {
    setRunning('all')
    setResults({})
    for (const testCase of HARD_CASES) {
      const result = await runCase(testCase)
      setResults(prev => ({ ...prev, [testCase.id]: result }))
    }
    setRunning(null)
  }, [runCase])

  const done = HARD_CASES.filter(c => results[c.id] && !results[c.id].error)
  const passed = done.filter(c => results[c.id].pass).length
  const oldPassed = done.filter(c => results[c.id].old?.pass).length

  // Keep the cases in their declared order, grouped by case name.
  const groups = []
  for (const testCase of HARD_CASES) {
    const last = groups[groups.length - 1]
    if (last && last.name === testCase.group) last.cases.push(testCase)
    else groups.push({ name: testCase.group, cases: [testCase] })
  }

  return (
    <div className="demo-page">
      <div className="demo-bar">
        <button className="run-btn" onClick={runAll} disabled={running !== null}>
          {running === 'all' ? 'Running…' : 'Run all'}
        </button>
        <label className="compare-toggle">
          <input
            type="checkbox"
            checked={compareOld}
            onChange={e => setCompareOld(e.target.checked)}
          />
          Also run Old VDOC
        </label>

        {done.length > 0 && (
          <div className="score-strip">
            <span className={`score${passed === done.length ? ' is-good' : ' is-bad'}`}>
              VDOC 2.0: {passed} / {done.length} pass
            </span>
            {compareOld && (
              <span className="score is-old">
                Old VDOC: {oldPassed} / {done.length} pass
              </span>
            )}
          </div>
        )}
      </div>

      {done.length === 0 && running === null && (
        <div className="notice">
          Nothing run yet. Press <strong>Run all</strong>, or run a single case.
          Every number on this page comes from a live run against the backend.
        </div>
      )}

      {groups.map(group => (
        <section key={group.name} className="case-group">
          <h3 className="case-group-title">{group.name}</h3>
          {group.cases.map(testCase => (
            <CaseRow
              key={testCase.id}
              testCase={testCase}
              result={results[testCase.id]}
              running={running === testCase.id || running === 'all'}
              onRun={() => runOne(testCase)}
              compareOld={compareOld}
            />
          ))}
        </section>
      ))}
    </div>
  )
}

function CaseRow({ testCase, result, running, onRun, compareOld }) {
  const state = !result ? 'idle'
    : result.error ? 'error'
    : result.pass ? 'pass' : 'fail'

  return (
    <div className={`case-row is-${state}`}>
      <div className="case-head">
        <span className={`case-verdict is-${state}`}>
          {state === 'pass' ? 'PASS' : state === 'fail' ? 'FAIL'
            : state === 'error' ? 'ERROR' : 'not run yet'}
        </span>
        <span className="case-lang">{testCase.lang}</span>
        <button className="case-run-btn" onClick={onRun} disabled={running}>
          {running ? '…' : 'Run'}
        </button>
      </div>

      <div className="case-command">“{testCase.command}”</div>
      {testCase.note && <div className="case-note">{testCase.note}</div>}

      <div className="case-compare">
        <div className="case-col">
          <div className="case-col-title">Expected</div>
          {testCase.expect.map((item, i) => (
            <div key={i} className="case-item">{i + 1}. {describeExpected(item)}</div>
          ))}
        </div>
        <div className="case-col">
          <div className="case-col-title">Actual — VDOC 2.0</div>
          {!result && <div className="case-item muted">not run yet</div>}
          {result?.error && <div className="case-item is-error">{result.error}</div>}
          {result?.actions?.map((entry, i) => (
            <div key={i} className="case-item">{i + 1}. {describeActual(entry)}</div>
          ))}
          {result?.actions?.length === 0 && <div className="case-item muted">no actions</div>}
        </div>
        {compareOld && (
          <div className="case-col">
            <div className="case-col-title">Actual — Old VDOC</div>
            {!result && <div className="case-item muted">not run yet</div>}
            {result?.old?.actions?.map((entry, i) => (
              <div key={i} className="case-item is-old">{i + 1}. {describeActual(entry)}</div>
            ))}
            {result && !result.old && !result.error && (
              <div className="case-item muted">not run</div>
            )}
          </div>
        )}
      </div>

      {result?.reasons?.length > 0 && (
        <ul className="case-reasons">
          {result.reasons.map((reason, i) => <li key={i}>{reason}</li>)}
        </ul>
      )}

      {result?.chunks && (
        <div className="case-foot">
          split into {result.chunks.length} chunk{result.chunks.length > 1 ? 's' : ''}
          {' · '}{result.applied} edit{result.applied === 1 ? '' : 's'} applied
          {' · '}run at {result.ranAt}
        </div>
      )}
    </div>
  )
}
