// A small line-based document model for the VDOC 2.0 demo pages.
//
// The main editor uses TipTap, where positions are ProseMirror offsets. The
// demo needs line numbers instead, because that is what the new pipeline
// targets, so it keeps its own model: an ordered list of lines, each with a
// stable id and a list of character-range marks.
//
// Marks are ranges rather than whole-line flags so an anchor command
// ("bold from car to accident") formats exactly the words it names, the same
// span the real editor would format.

export const SAMPLE_LINES = [
  'VDOC 2.0 — Multi-Action Voice Commands',
  'This is the introduction to the VDOC system.',
  'The car failed to brake and the accident was reported.',
  'Researchers compared a rule-based system against DocEditBERT.',
  'The conclusion summarises the key findings of the study.',
]

let nextLineId = 1

export function createDoc(texts = SAMPLE_LINES) {
  nextLineId = 1
  return texts.map(text => ({
    id: nextLineId++,
    original: nextLineId - 1,
    text,
    marks: [],
    heading: null,
  }))
}

function cloneDoc(doc) {
  return doc.map(line => ({ ...line, marks: line.marks.map(m => ({ ...m })) }))
}

// ── Finding anchor text ─────────────────────────────────────────────────────

function levenshtein(a, b) {
  const m = a.length, n = b.length
  const dp = Array.from({ length: m + 1 }, (_, i) => Array(n + 1).fill(0))
  for (let i = 0; i <= m; i++) dp[i][0] = i
  for (let j = 0; j <= n; j++) dp[0][j] = j
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
  return dp[m][n]
}

// Exact match first, then Levenshtein <= 2 on single words — the same
// fuzzy rule the existing editor uses.
function findInLine(line, needle, fromIndex = 0) {
  const hay = line.text.toLowerCase()
  const want = needle.toLowerCase().trim()
  if (!want) return null

  const exact = hay.indexOf(want, fromIndex)
  if (exact !== -1) return { start: exact, end: exact + want.length, fuzzy: false }

  const re = /\S+/g
  let match
  while ((match = re.exec(line.text)) !== null) {
    if (match.index < fromIndex) continue
    const word = match[0].replace(/[.,?!;:]+$/, '')
    if (levenshtein(want, word.toLowerCase()) <= 2) {
      return { start: match.index, end: match.index + word.length, fuzzy: true }
    }
  }
  return null
}

export function findAnchorRange(doc, fromText, toText) {
  for (let i = 0; i < doc.length; i++) {
    const start = findInLine(doc[i], fromText)
    if (!start) continue

    for (let j = i; j < doc.length; j++) {
      const searchFrom = j === i ? start.start : 0
      const end = findInLine(doc[j], toText, searchFrom)
      if (!end) continue
      if (j === i && end.end <= start.start) continue
      return {
        startLine: i, startOffset: start.start,
        endLine: j, endOffset: end.end,
        fuzzy: start.fuzzy || end.fuzzy,
      }
    }
  }
  return null
}

// ── Applying marks ──────────────────────────────────────────────────────────

function addMark(line, start, end, type) {
  if (end <= start) return
  line.marks.push({ start, end, type })
}

function markRange(doc, range, type) {
  const touched = []
  for (let i = range.startLine; i <= range.endLine; i++) {
    const line = doc[i]
    const start = i === range.startLine ? range.startOffset : 0
    const end = i === range.endLine ? range.endOffset : line.text.length
    addMark(line, start, end, type)
    touched.push(line.id)
  }
  return touched
}

function deleteRange(doc, range) {
  const touched = []
  if (range.startLine === range.endLine) {
    const line = doc[range.startLine]
    line.text = line.text.slice(0, range.startOffset) + line.text.slice(range.endOffset)
    line.marks = []
    touched.push(line.id)
    return { doc, touched }
  }
  const first = doc[range.startLine]
  const last = doc[range.endLine]
  first.text = first.text.slice(0, range.startOffset) + last.text.slice(range.endOffset)
  first.marks = []
  touched.push(first.id)
  const remaining = doc.filter((_, i) => i <= range.startLine || i > range.endLine)
  return { doc: remaining, touched }
}

// ── The executor ────────────────────────────────────────────────────────────

const MARK_ACTIONS = {
  bold: 'bold',
  italic: 'italic',
  underline: 'underline',
  highlight: 'highlight',
  'font-increase': 'fontSize',
}

/**
 * Apply one parsed action to the document.
 *
 * Returns { doc, touched, ok, message }. `doc` is a new array; the input is
 * never modified, so the Live Demo can keep a "before" copy to compare.
 */
export function applyAction(docIn, entry) {
  const doc = cloneDoc(docIn)
  const action = entry.action
  const target = entry.target

  // A low-confidence prediction is a guess. Acting on a guess edits the
  // wrong thing and the user cannot tell why, so ask for a repeat instead.
  // This check comes first: "not found" would blame the document for what is
  // really the model saying it does not know.
  if (entry.low_confidence) {
    const score = typeof entry.confidence === 'number' ? entry.confidence.toFixed(3) : '?'
    return {
      doc, touched: [], ok: false, unsure: true,
      message: `I'm not sure, please repeat (confidence ${score})`,
    }
  }

  if (!action) return { doc, touched: [], ok: false, message: 'no action detected' }
  if (!target) return { doc, touched: [], ok: false, message: 'no target found' }

  // ── Line target ───────────────────────────────────────────────────────────
  if (target.kind === 'line') {
    const position = entry.resolved_line
      ? entry.resolved_line.current
      : (target.n === -1 ? doc.length : target.n)
    const index = position - 1
    if (index < 0 || index >= doc.length) {
      return { doc, touched: [], ok: false, message: `line ${position} does not exist` }
    }
    const line = doc[index]

    if (action === 'delete') {
      return {
        doc: doc.filter((_, i) => i !== index),
        touched: [line.id], ok: true,
        message: `deleted line ${position}`,
      }
    }
    if (action === 'heading2' || action === 'heading3') {
      line.heading = action === 'heading2' ? 2 : 3
      return { doc, touched: [line.id], ok: true, message: `line ${position} → ${action}` }
    }
    if (action === 'insert-after') {
      const added = { id: 1000 + nextLineId++, text: '(new line)', marks: [], heading: null }
      doc.splice(index + 1, 0, added)
      return { doc, touched: [added.id], ok: true, message: `inserted after line ${position}` }
    }
    const markType = MARK_ACTIONS[action]
    if (!markType) return { doc, touched: [], ok: false, message: `cannot apply "${action}" to a line` }
    addMark(line, 0, line.text.length, markType)
    return { doc, touched: [line.id], ok: true, message: `${action} on line ${position}` }
  }

  // ── Anchor target ─────────────────────────────────────────────────────────
  if (!target.from || !target.to) {
    return { doc, touched: [], ok: false, message: 'could not extract all fields — no edit applied' }
  }

  const range = findAnchorRange(doc, target.from, target.to)
  if (!range) {
    return { doc, touched: [], ok: false, message: `"${target.from}" … "${target.to}" not found` }
  }

  if (action === 'delete') {
    const result = deleteRange(doc, range)
    return { ...result, ok: true, message: `deleted from "${target.from}" to "${target.to}"` }
  }
  if (action === 'heading2' || action === 'heading3') {
    const touched = []
    for (let i = range.startLine; i <= range.endLine; i++) {
      doc[i].heading = action === 'heading2' ? 2 : 3
      touched.push(doc[i].id)
    }
    return { doc, touched, ok: true, message: `${action} from "${target.from}" to "${target.to}"` }
  }
  const markType = MARK_ACTIONS[action]
  if (!markType) {
    return { doc, touched: [], ok: false, message: `"${action}" is not supported in the demo view` }
  }
  const touched = markRange(doc, range, markType)
  return {
    doc, touched, ok: true,
    message: `${action} from "${target.from}" to "${target.to}"${range.fuzzy ? ' (fuzzy match)' : ''}`,
  }
}

/** Run a whole action list in order, collecting a step-by-step trace. */
export function runActions(docIn, actions) {
  let doc = docIn
  const steps = []
  const touchedAll = new Set()
  for (const entry of actions) {
    const result = applyAction(doc, entry)
    doc = result.doc
    result.touched.forEach(id => touchedAll.add(id))
    steps.push({
      entry, ok: result.ok, message: result.message,
      touched: result.touched, unsure: result.unsure ?? false,
    })
  }
  return { doc, steps, touched: [...touchedAll] }
}

// ── Rendering ───────────────────────────────────────────────────────────────

/**
 * Split a line into styled segments at its mark boundaries, so overlapping
 * marks (bold over a highlight) render correctly.
 */
export function segmentsOf(line) {
  if (!line.marks.length) return [{ text: line.text, types: [] }]

  const points = new Set([0, line.text.length])
  line.marks.forEach(m => { points.add(m.start); points.add(m.end) })
  const cuts = [...points].filter(p => p >= 0 && p <= line.text.length).sort((a, b) => a - b)

  const segments = []
  for (let i = 0; i < cuts.length - 1; i++) {
    const [start, end] = [cuts[i], cuts[i + 1]]
    if (end <= start) continue
    const types = line.marks
      .filter(m => m.start <= start && m.end >= end)
      .map(m => m.type)
    segments.push({ text: line.text.slice(start, end), types: [...new Set(types)] })
  }
  return segments
}
