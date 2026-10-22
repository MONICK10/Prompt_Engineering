import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Underline from '@tiptap/extension-underline'
import Highlight from '@tiptap/extension-highlight'
import TextStyle from '@tiptap/extension-text-style'
import { Mark } from '@tiptap/core'
import { useState, useRef, useCallback } from 'react'
import axios from 'axios'

// Inline font-size mark (free alternative to Pro extension)
const FontSize = Mark.create({
  name: 'fontSize',
  addAttributes() {
    return { size: { default: null, renderHTML: a => a.size ? { style: `font-size: ${a.size}` } : {} } }
  },
  renderHTML({ HTMLAttributes }) { return ['span', HTMLAttributes, 0] },
  parseHTML() { return [{ tag: 'span[style*=font-size]' }] },
})

const API = 'http://localhost:5050'

const SAMPLE = `<h1>VDOC &mdash; Voice Document Editor</h1>
<p>This is the <strong>introduction</strong> to the VDOC system. It uses a locally-trained DistilBERT model to understand free-form voice commands. The <em>conclusion</em> of this demonstration will show how accurately it works compared to baselines.</p>
<p>In the accident report, investigators noted the car was traveling at high speed. The car failed to brake at the intersection, leading to a collision. The full accident sequence &mdash; from the car entering the road to the accident being reported &mdash; took less than thirty seconds.</p>
<p>Researchers compared three approaches: a rule-based system, a BiLSTM model trained from scratch, and DocEditBERT. The title of the paper is &ldquo;Voice-Controlled Document Editing with DocEditBERT&rdquo;, and the summary below presents the key findings.</p>
<p>Try a voice command: click the red microphone button and say something like <em>&ldquo;bold from car to accident&rdquo;</em> or <em>&ldquo;italic from introduction to conclusion&rdquo;</em>. The editor will find the text and apply the formatting automatically.</p>`

// ── Levenshtein distance (iterative DP) ─────────────────────────────────────
function levenshtein(a, b) {
  const m = a.length, n = b.length
  const dp = Array.from({ length: m + 1 }, (_, i) => Array(n + 1).fill(0))
  for (let i = 0; i <= m; i++) dp[i][0] = i
  for (let j = 0; j <= n; j++) dp[0][j] = j
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i-1] === b[j-1]
        ? dp[i-1][j-1]
        : 1 + Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
  return dp[m][n]
}

// ── ProseMirror text position finder ────────────────────────────────────────
function findOccurrences(doc, searchText) {
  const results = []
  const needle  = searchText.toLowerCase().trim()
  if (!needle) return results

  doc.descendants((node, pos) => {
    if (!node.isText) return
    const haystack = node.text.toLowerCase()
    let idx = haystack.indexOf(needle)
    while (idx !== -1) {
      const raw      = node.text
      const ctxStart = Math.max(0, idx - 28)
      const ctxEnd   = Math.min(raw.length, idx + needle.length + 28)
      results.push({
        from:    pos + idx,
        to:      pos + idx + needle.length,
        context: (ctxStart > 0 ? '...' : '') +
                 raw.slice(ctxStart, ctxEnd) +
                 (ctxEnd < raw.length ? '...' : ''),
      })
      idx = haystack.indexOf(needle, idx + 1)
    }
  })
  return results
}

// ── Fuzzy anchor finder (falls back to Levenshtein ≤ 2 when exact fails) ────
// Returns { matches: [...], fuzzy: bool }
function findAnchor(doc, searchText) {
  const exact = findOccurrences(doc, searchText)
  if (exact.length > 0) return { matches: exact, fuzzy: false }

  const needle = searchText.toLowerCase().trim()
  const fuzzyMatches = []

  doc.descendants((node, pos) => {
    if (!node.isText) return
    const text = node.text
    // Walk every whitespace-delimited token in the text node
    const re = /\S+/g
    let m
    while ((m = re.exec(text)) !== null) {
      const word  = m[0]
      const clean = word.toLowerCase().replace(/[.,?!;:]+$/, '')
      if (levenshtein(needle, clean) <= 2) {
        const ctxStart = Math.max(0, m.index - 28)
        const ctxEnd   = Math.min(text.length, m.index + word.length + 28)
        fuzzyMatches.push({
          from:    pos + m.index,
          to:      pos + m.index + word.length,
          context: (ctxStart > 0 ? '...' : '') +
                   text.slice(ctxStart, ctxEnd) +
                   (ctxEnd < text.length ? '...' : ''),
        })
      }
    }
  })

  return { matches: fuzzyMatches, fuzzy: fuzzyMatches.length > 0 }
}

// ── Toolbar button ────────────────────────────────────────────────────────────
function TBtn({ label, title, active, onClick }) {
  return (
    <button
      className={`toolbar-btn${active ? ' is-active' : ''}`}
      onClick={onClick}
      title={title}
    >
      {label}
    </button>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
export default function EditorPage() {
  const editor = useEditor({
    extensions: [StarterKit, Underline, Highlight, TextStyle, FontSize],
    content: SAMPLE,
  })

  const [recording,      setRecording]      = useState(false)
  const [status,         setStatus]         = useState('Ready — click the mic to give a voice command')
  const [overlay,        setOverlay]        = useState(null)   // parsed result card
  const [disambiguation, setDisambiguation] = useState(null)   // multi-occurrence panel

  const mediaRef  = useRef(null)
  const chunksRef = useRef([])
  const timerRef  = useRef(null)

  // ── Apply edit to document ─────────────────────────────────────────────────
  const applyEdit = useCallback((action, fromText, toText, fromIdx = null) => {
    if (!editor) return

    const { matches: fromOccs, fuzzy: fromFuzzy } = findAnchor(editor.state.doc, fromText)
    if (fromOccs.length === 0) {
      setStatus(`"${fromText}" not found in document`)
      return
    }

    // Need disambiguation
    if (fromIdx === null && fromOccs.length > 1) {
      setDisambiguation({ action, fromText, toText, occurrences: fromOccs })
      return
    }

    const fromOcc = fromOccs[fromIdx ?? 0]

    const { matches: toOccs, fuzzy: toFuzzy } = findAnchor(editor.state.doc, toText)
    if (toOccs.length === 0) {
      setStatus(`"${toText}" not found in document`)
      return
    }
    // Pick the to-occurrence that begins at or after the chosen from-position
    const toOcc = toOccs.find(o => o.from >= fromOcc.from) ?? toOccs[toOccs.length - 1]

    if (fromFuzzy || toFuzzy)
      setOverlay(prev => prev ? { ...prev, fuzzyMatch: true } : prev)

    const selFrom = fromOcc.from
    const selTo   = toOcc.to

    if (selFrom >= selTo) {
      setStatus(`"${fromText}" appears after "${toText}" at this location — pick a different occurrence`)
      return
    }

    let chain = editor.chain().focus().setTextSelection({ from: selFrom, to: selTo })

    switch (action) {
      case 'bold':          chain = chain.toggleBold();                               break
      case 'italic':        chain = chain.toggleItalic();                             break
      case 'underline':     chain = chain.toggleUnderline();                          break
      case 'delete':        chain = chain.deleteSelection();                          break
      case 'heading2':      chain = chain.setHeading({ level: 2 });                   break
      case 'heading3':      chain = chain.setHeading({ level: 3 });                   break
      case 'highlight':     chain = chain.toggleHighlight();                          break
      case 'font-increase': chain = chain.setMark('fontSize', { size: '1.4em' });    break
      default:
        setStatus(`Unknown action: "${action}"`)
        return
    }

    chain.run()
    setDisambiguation(null)
    setStatus(`Applied "${action}"  from "${fromText}"  to "${toText}"`)
  }, [editor])

  // ── Replace: find target word and replace with replacement ────────────────
  const applyReplace = useCallback((targetText, replacement) => {
    if (!editor) return
    const { matches: occs, fuzzy } = findAnchor(editor.state.doc, targetText)
    if (occs.length === 0) { setStatus(`"${targetText}" not found`); return }
    if (fuzzy) setOverlay(prev => prev ? { ...prev, fuzzyMatch: true } : prev)
    // Replace first occurrence (no disambiguation needed — single-word targets)
    const { from, to } = occs[0]
    editor.chain().focus().setTextSelection({ from, to }).insertContent(replacement).run()
    setDisambiguation(null)
    setStatus(`Replaced "${targetText}" with "${replacement}"`)
  }, [editor])

  // ── Insert-after: insert text immediately after anchor word ───────────────
  const applyInsertAfter = useCallback((textToInsert, anchor) => {
    if (!editor) return
    const { matches: occs, fuzzy } = findAnchor(editor.state.doc, anchor)
    if (occs.length === 0) { setStatus(`Anchor "${anchor}" not found`); return }
    if (fuzzy) setOverlay(prev => prev ? { ...prev, fuzzyMatch: true } : prev)
    const insertPos = occs[0].to   // position right after the anchor word
    editor.chain().focus()
      .setTextSelection({ from: insertPos, to: insertPos })
      .insertContent(' ' + textToInsert)
      .run()
    setDisambiguation(null)
    setStatus(`Inserted "${textToInsert}" after "${anchor}"`)
  }, [editor])

  // ── Process audio blob: transcribe → parse → apply ────────────────────────
  const processAudio = useCallback(async (blob) => {
    setStatus('Transcribing...')
    try {
      const form = new FormData()
      form.append('audio', blob, 'recording.webm')
      const { data: tData } = await axios.post(`${API}/transcribe`, form)
      const text = tData.text?.trim()

      if (!text) { setStatus('No speech detected'); return }
      setStatus(`Heard: "${text}"  —  parsing...`)

      const { data: p } = await axios.post(`${API}/parse`, { command: text })
      const fromAnchor = (p.from || '').replace(/[.,?!;:]+$/, '')
      const toAnchor   = (p.to   || '').replace(/[.,?!;:]+$/, '')
      setOverlay({
        transcription: text,
        action:        p.action,
        from:          fromAnchor,
        to:            toAnchor,
        confidence:    p.confidence,
        routedTo:      p.routed_to || 'doceditbert',
        method:        p.method    || 'doceditbert',
        fuzzyMatch:    false,
      })

      if (p.action && fromAnchor && toAnchor) {
        if (p.action === 'replace') {
          applyReplace(fromAnchor, toAnchor)
        } else if (p.action === 'insert-after') {
          applyInsertAfter(fromAnchor, toAnchor)
        } else {
          applyEdit(p.action, fromAnchor, toAnchor)
        }
      } else {
        setStatus('Could not extract all fields — no edit applied')
      }
    } catch (err) {
      const msg = err.code === 'ERR_NETWORK'
        ? 'Cannot reach backend at localhost:5050 — is the server running?'
        : err.response?.data?.error ?? err.message
      setStatus(`Error: ${msg}`)
    }
  }, [applyEdit])

  // ── Mic button handler ────────────────────────────────────────────────────
  const handleMic = useCallback(async () => {
    if (recording) {
      clearTimeout(timerRef.current)
      mediaRef.current?.stop()
      setRecording(false)
      return
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('This browser does not support microphone access (try Chrome or Firefox)')
      return
    }

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (err) {
      const name = err?.name ?? 'UnknownError'
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setStatus('Microphone access denied — click the lock icon in the address bar and allow mic')
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        setStatus('No microphone found — plug one in and try again')
      } else if (name === 'NotReadableError' || name === 'TrackStartError') {
        setStatus('Microphone is in use by another app — close it and try again')
      } else {
        setStatus(`Mic error (${name}): ${err?.message ?? 'unknown'}`)
      }
      return
    }

    chunksRef.current = []
    let mr
    try {
      mr = new MediaRecorder(stream)
    } catch (err) {
      stream.getTracks().forEach(t => t.stop())
      setStatus(`MediaRecorder failed: ${err?.message ?? err}`)
      return
    }
    mediaRef.current = mr

    mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
    mr.onstop = () => {
      stream.getTracks().forEach(t => t.stop())
      const blob = new Blob(chunksRef.current, { type: mr.mimeType || 'audio/webm' })
      processAudio(blob)
    }

    mr.start()
    setRecording(true)
    setOverlay(null)
    setStatus('Listening... (click mic again to stop, or wait 6 s)')

    // Auto-stop after 6 seconds
    timerRef.current = setTimeout(() => {
      if (mr.state === 'recording') { mr.stop(); setRecording(false) }
    }, 6000)
  }, [recording, processAudio])

  if (!editor) return null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>

      {/* ── Scrollable document area ── */}
      <div className="editor-wrap">

        {/* Formatting toolbar */}
        <div className="toolbar">
          <TBtn label="B"  title="Bold"      active={editor.isActive('bold')}            onClick={() => editor.chain().focus().toggleBold().run()} />
          <TBtn label="I"  title="Italic"    active={editor.isActive('italic')}          onClick={() => editor.chain().focus().toggleItalic().run()} />
          <TBtn label="U"  title="Underline" active={editor.isActive('underline')}       onClick={() => editor.chain().focus().toggleUnderline().run()} />
          <div className="toolbar-sep" />
          <TBtn label="H2" title="Heading 2" active={editor.isActive('heading',{level:2})} onClick={() => editor.chain().focus().toggleHeading({level:2}).run()} />
          <TBtn label="H3" title="Heading 3" active={editor.isActive('heading',{level:3})} onClick={() => editor.chain().focus().toggleHeading({level:3}).run()} />
          <div className="toolbar-sep" />
          <TBtn label="&#x2022;" title="Bullet list" active={editor.isActive('bulletList')}  onClick={() => editor.chain().focus().toggleBulletList().run()} />
          <TBtn label="1."       title="Ordered list" active={editor.isActive('orderedList')} onClick={() => editor.chain().focus().toggleOrderedList().run()} />
          <div className="toolbar-sep" />
          <TBtn label="&#x21A9;" title="Undo" onClick={() => editor.chain().focus().undo().run()} />
          <TBtn label="&#x21AA;" title="Redo" onClick={() => editor.chain().focus().redo().run()} />
        </div>

        {/* Document surface */}
        <div className="editor-surface">
          <EditorContent editor={editor} />
        </div>
      </div>

      {/* ── Status bar ── */}
      <div className="statusbar">{status}</div>

      {/* ── Parse result overlay ── */}
      {overlay && (
        <div className="parse-overlay">
          <button className="parse-overlay-close" onClick={() => setOverlay(null)}>&#x2715;</button>
          <div className="parse-overlay-label">Parsed command</div>
          <div className="parse-overlay-transcript">"{overlay.transcription}"</div>
          <div className="parse-tags">
            {overlay.action     && <span className="tag tag-action">action: {overlay.action}</span>}
            {overlay.from       && <span className="tag tag-from">from: {overlay.from}</span>}
            {overlay.to         && <span className="tag tag-to">to: {overlay.to}</span>}
            {overlay.confidence && <span className="tag tag-conf">conf: {overlay.confidence}</span>}
            {overlay.routedTo === 'groq' && (
              <span className="tag tag-fallback">LLM fallback</span>
            )}
            {overlay.fuzzyMatch && (
              <span className="tag tag-fuzzy">Fuzzy match found</span>
            )}
          </div>
        </div>
      )}

      {/* ── Disambiguation panel ── */}
      {disambiguation && (
        <div className="disambig-panel">
          <h4>Found "{disambiguation.fromText}" in {disambiguation.occurrences.length} places — which one?</h4>
          {disambiguation.occurrences.map((occ, i) => (
            <button
              key={i}
              className="disambig-option"
              onClick={() => applyEdit(disambiguation.action, disambiguation.fromText, disambiguation.toText, i)}
            >
              <span className="disambig-number">{i + 1}</span>
              {occ.context}
            </button>
          ))}
          <button className="disambig-cancel" onClick={() => setDisambiguation(null)}>Cancel</button>
        </div>
      )}

      {/* ── Voice button ── */}
      <button
        className={`voice-btn${recording ? ' recording' : ''}`}
        onClick={handleMic}
        title={recording ? 'Click to stop recording' : 'Click to speak a command'}
        aria-label={recording ? 'Stop recording' : 'Start voice command'}
      >
        {recording ? '⏹' : '🎙'}
      </button>
    </div>
  )
}
