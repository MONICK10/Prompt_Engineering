// The hard cases, in English and Tanglish, with what each one must produce.
//
// These mirror multi_action/tests/test_hard_cases.py. The page runs them
// through the real backend and the real executor and compares -- nothing on
// screen is typed in by hand.

export const HARD_CASES = [
  // ── Case 1: two actions, two targets ──────────────────────────────────────
  {
    id: 'c1-en',
    group: 'Two actions, two targets',
    lang: 'English',
    command: 'bold the first line and delete the third line',
    expect: [
      { action: 'bold', line: 1 },
      { action: 'delete', line: 3 },
    ],
  },
  {
    id: 'c1-ta',
    group: 'Two actions, two targets',
    lang: 'Tanglish',
    command: 'first line bold pannu, apram third line delete pannu',
    expect: [
      { action: 'bold', line: 1 },
      { action: 'delete', line: 3 },
    ],
  },

  // ── Case 2: one target, two actions ───────────────────────────────────────
  {
    id: 'c2-en',
    group: 'One target, two actions',
    lang: 'English',
    command: 'make the first line bold and italic',
    note: 'The second chunk has no target and borrows the first one.',
    expect: [
      { action: 'bold', line: 1 },
      { action: 'italic', line: 1, shared: true },
    ],
  },
  {
    id: 'c2-ta',
    group: 'One target, two actions',
    lang: 'Tanglish',
    command: 'first line bold-um italic-um pannu',
    note: 'The second chunk has no target and borrows the first one.',
    expect: [
      { action: 'bold', line: 1 },
      { action: 'italic', line: 1, shared: true },
    ],
  },

  // ── Case 3: lines shift after a delete ────────────────────────────────────
  {
    id: 'c3-en',
    group: 'Lines shift after delete',
    lang: 'English',
    command: 'delete the second line, then bold the second line',
    note: 'Both say "second line". The second one must hit what was line 3.',
    expect: [
      { action: 'delete', line: 2 },
      { action: 'bold', line: 2, wasLine: 3 },
    ],
  },
  {
    id: 'c3-ta',
    group: 'Lines shift after delete',
    lang: 'Tanglish',
    command: 'second line delete pannu, apram second line bold pannu',
    note: 'Both say "second line". The second one must hit what was line 3.',
    expect: [
      { action: 'delete', line: 2 },
      { action: 'bold', line: 2, wasLine: 3 },
    ],
  },

  // ── Case 4: joining words ─────────────────────────────────────────────────
  {
    id: 'c4-en',
    group: 'Joining words',
    lang: 'English',
    command: 'bold the first line after that delete the third line',
    expect: [
      { action: 'bold', line: 1 },
      { action: 'delete', line: 3 },
    ],
  },
  {
    id: 'c4-ta',
    group: 'Joining words',
    lang: 'Tanglish',
    command: 'first line bold pannu pannitu third line delete pannu',
    expect: [
      { action: 'bold', line: 1 },
      { action: 'delete', line: 3 },
    ],
  },

  // ── Case 5: sentence breaks ───────────────────────────────────────────────
  {
    id: 'c5-ta',
    group: 'Sentence breaks',
    lang: 'Tanglish',
    command: 'first line bold pannu. second line font size 20 vei.',
    note: 'Whisper ends sentences with a full stop and no joining word.',
    expect: [
      { action: 'bold', line: 1 },
      { action: 'font-increase', line: 2 },
    ],
  },

  // ── Rule 1: VDOC 1.0 behaviour must be untouched ──────────────────────────
  {
    id: 'r1-anchor',
    group: 'Rule 1: old commands unchanged',
    lang: 'Tanglish',
    command: 'introduction la irunthu conclusion varikum bold panu',
    note: 'An anchor command must behave exactly as it did in VDOC 1.0.',
    expect: [
      { action: 'bold', from: 'introduction', to: 'conclusion' },
    ],
  },
  {
    id: 'r1-nosplit',
    group: 'Rule 1: old commands unchanged',
    lang: 'Tanglish',
    command: 'summary ku aprom conclusion insert panu',
    note: '"aprom" here is part of one insert-after command, not a join.',
    expect: [
      { action: 'insert-after', from: 'conclusion', to: 'summary' },
    ],
  },
  {
    id: 'r1-and',
    group: 'Rule 1: old commands unchanged',
    lang: 'English',
    command: 'select header to footer and make it bold',
    note: '"and" here joins an anchor pair, not two actions.',
    expect: [
      { action: 'bold', from: 'header', to: 'footer' },
    ],
  },
]

/** A short human-readable form of one expected action. */
export function describeExpected(item) {
  if (item.from || item.to) {
    return `${item.action} from "${item.from}" to "${item.to}"`
  }
  let label = `${item.action} line ${item.line}`
  if (item.wasLine) label += ` (was line ${item.wasLine})`
  if (item.shared) label += ' [shared target]'
  return label
}

/** The same form, built from what actually came back. */
export function describeActual(entry) {
  const target = entry.target
  if (!entry.action) return 'nothing parsed'
  if (!target) return `${entry.action} (no target)`
  if (target.kind === 'anchor') {
    return `${entry.action} from "${target.from}" to "${target.to}"`
  }
  const resolved = entry.resolved_line
  if (!resolved) return `${entry.action} line ${target.n}`
  let label = `${entry.action} line ${resolved.current}`
  if (resolved.moved) label += ` (was line ${resolved.original})`
  if (entry.shared_target) label += ' [shared target]'
  return label
}

/**
 * Compare what came back against what the case demands.
 * Returns { pass, reasons[] } — reasons list every mismatch, not just the
 * first, so a failure is diagnosable from the screen.
 */
export function checkCase(testCase, actions) {
  const reasons = []
  const expect = testCase.expect

  if (actions.length !== expect.length) {
    reasons.push(`expected ${expect.length} action${expect.length > 1 ? 's' : ''}, got ${actions.length}`)
  }

  expect.forEach((want, i) => {
    const got = actions[i]
    if (!got) { reasons.push(`action ${i + 1} is missing`); return }

    if (got.action !== want.action) {
      reasons.push(`action ${i + 1}: expected "${want.action}", got "${got.action}"`)
    }

    const target = got.target
    if (want.from || want.to) {
      if (!target || target.kind !== 'anchor') {
        reasons.push(`action ${i + 1}: expected an anchor target`)
      } else {
        if ((target.from || '').toLowerCase() !== (want.from || '').toLowerCase()) {
          reasons.push(`action ${i + 1}: expected from "${want.from}", got "${target.from}"`)
        }
        if ((target.to || '').toLowerCase() !== (want.to || '').toLowerCase()) {
          reasons.push(`action ${i + 1}: expected to "${want.to}", got "${target.to}"`)
        }
      }
      return
    }

    if (!target || target.kind !== 'line') {
      reasons.push(`action ${i + 1}: expected a line target`)
      return
    }
    const resolved = got.resolved_line
    if (!resolved) {
      reasons.push(`action ${i + 1}: line was not resolved to a position`)
    } else {
      if (resolved.current !== want.line) {
        reasons.push(`action ${i + 1}: expected line ${want.line}, got line ${resolved.current}`)
      }
      if (want.wasLine && resolved.original !== want.wasLine) {
        reasons.push(`action ${i + 1}: expected it to come from line ${want.wasLine}, got line ${resolved.original}`)
      }
    }
    if (want.shared && !got.shared_target) {
      reasons.push(`action ${i + 1}: expected the target to be shared from the previous chunk`)
    }
  })

  return { pass: reasons.length === 0, reasons }
}
