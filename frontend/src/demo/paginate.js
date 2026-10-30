// Split a document into A4 pages by measuring what the browser actually drew.
//
// The exporters paginate with their own engines, so the preview can only agree
// with them by measuring: guessing from character counts drifts as soon as a
// font or size changes. Blocks are measured once off-screen at the real page
// width, then dealt into pages greedily. A block too tall for an empty page is
// split by line, which is why the measurement keeps a line height per block.

export const A4_RATIO = 297 / 210          // height / width
const SPLIT_SAFETY = 2                     // px, keeps a split line off the edge

/** One flat list of renderable blocks, in reading order. */
export function flattenDoc(doc) {
  if (!doc) return []
  const blocks = []
  if (doc.title?.text) blocks.push({ kind: 'title', run: doc.title, key: 'title' })
  ;(doc.sections ?? []).forEach((section, s) => {
    if (section.heading?.text) {
      blocks.push({ kind: 'heading', run: section.heading, key: `h${s}`, section: s })
    }
    ;(section.paragraphs ?? []).forEach((paragraph, p) => {
      blocks.push({ kind: 'paragraph', run: paragraph, key: `p${s}-${p}`, section: s })
    })
  })
  return blocks
}

/**
 * Deal measured blocks into pages.
 *
 * `sizes` maps a block key to { height, lineHeight }. A block that fits goes
 * on the current page; one that does not starts a new page; one taller than a
 * whole page is cut into line-sized pieces so text never leaves the sheet.
 */
export function paginate(blocks, sizes, contentHeight) {
  if (!contentHeight || contentHeight <= 0) return [blocks]
  const pages = []
  let page = []
  let used = 0

  const flush = () => { if (page.length) { pages.push(page); page = []; used = 0 } }

  for (const block of blocks) {
    const size = sizes[block.key]
    const height = size?.height ?? 0
    const line = Math.max(size?.lineHeight ?? 0, 1)

    if (height <= contentHeight - used) {
      page.push(block)
      used += height
      continue
    }
    if (height <= contentHeight) {        // fits a fresh page, just not this one
      flush()
      page.push(block)
      used = height
      continue
    }

    // Taller than any page: cut it by line across as many pages as it needs.
    const totalLines = Math.max(1, Math.round(height / line))
    const words = String(block.run.text ?? '').split(/\s+/).filter(Boolean)
    const perLine = Math.max(1, Math.ceil(words.length / totalLines))
    let at = 0
    let part = 0
    while (at < words.length) {
      const room = Math.max(1, Math.floor((contentHeight - used - SPLIT_SAFETY) / line))
      if (room < 1) { flush(); continue }
      const take = Math.max(1, room * perLine)
      const slice = words.slice(at, at + take)
      at += take
      page.push({
        ...block,
        key: `${block.key}#${part}`,
        run: { ...block.run, text: slice.join(' ') },
        continued: part > 0,
      })
      used += Math.min(room, Math.ceil(slice.length / perLine)) * line
      part += 1
      if (at < words.length) flush()
    }
  }
  flush()
  return pages.length ? pages : [[]]
}

/** Inline style for one run, shared by the measurer and the page. */
export function runStyle(run, page) {
  return {
    fontWeight: run.bold ? 700 : 400,
    fontStyle: run.italic ? 'italic' : 'normal',
    textDecoration: run.underline ? 'underline' : 'none',
    fontSize: `${run.size ?? page?.size ?? 12}pt`,
    fontFamily: run.font ?? page?.font ?? 'Times New Roman',
    textAlign: run.align ?? 'left',
    lineHeight: page?.line_spacing ?? 1.5,
    color: run.color ? colorOf(run.color) : undefined,
    backgroundColor: run.highlight ? colorOf(run.highlight) : undefined,
  }
}

// Mirrors COLORS in studio_tools.py, so preview, Word and PDF agree.
const NAMED = {
  yellow: '#ffff00', green: '#00ff00', cyan: '#00ffff', magenta: '#ff00ff',
  blue: '#0000ff', red: '#ff0000', darkblue: '#000080', darkred: '#800000',
  darkgreen: '#008000', gray: '#808080', grey: '#808080', black: '#000000',
  white: '#ffffff', orange: '#ffa500', pink: '#ffc0cb',
}
export function colorOf(name) {
  const key = String(name ?? '').trim().toLowerCase()
  return NAMED[key] ?? (key.startsWith('#') ? key : undefined)
}
