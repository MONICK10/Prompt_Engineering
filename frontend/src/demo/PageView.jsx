// The A4 page view: measure, paginate, then draw white sheets on grey.
//
// Two passes. The first renders every block into a hidden box that is exactly
// as wide as a page's text column and records its height; the second deals
// those heights into pages. Measuring beats estimating because the exporters
// lay out with real fonts too, and a guess drifts the moment a size changes.
import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { A4_RATIO, flattenDoc, paginate, runStyle } from './paginate'

const MARGIN_RATIO = 2.5 / 21          // 2.5 cm margins on a 21 cm page

export default function PageView({ doc, touched = [], width = 620 }) {
  const page = doc?.page ?? {}
  const blocks = useMemo(() => flattenDoc(doc), [doc])
  const [sizes, setSizes] = useState({})
  const measurer = useRef(null)

  const pageHeight = width * A4_RATIO
  const margin = width * MARGIN_RATIO
  const contentWidth = width - margin * 2
  const contentHeight = pageHeight - margin * 2 - 18   // 18px for the footer

  // Measure after every render that could change a height.
  useLayoutEffect(() => {
    if (!measurer.current) return
    const next = {}
    for (const node of measurer.current.children) {
      const key = node.dataset.key
      const line = parseFloat(getComputedStyle(node).lineHeight) || 16
      next[key] = { height: node.offsetHeight, lineHeight: line }
    }
    setSizes(prev => (shallowEqual(prev, next) ? prev : next))
  }, [doc, width])

  const pages = useMemo(
    () => paginate(blocks, sizes, contentHeight),
    [blocks, sizes, contentHeight])

  const ready = blocks.length === 0 || Object.keys(sizes).length > 0

  return (
    <>
      {/* Pass 1: off-screen, same column width as a real page. */}
      <div ref={measurer} className="pv-measure" style={{ width: contentWidth }}>
        {blocks.map(block => (
          <div key={block.key} data-key={block.key}
               className={`pv-block pv-${block.kind}`}
               style={runStyle(block.run, page)}>
            {block.run.text}
          </div>
        ))}
      </div>

      {/* Pass 2: the sheets. */}
      <div className="pv-stack">
        {(ready ? pages : [blocks]).map((items, i) => (
          <div key={i} className="pv-page"
               style={{ width, height: pageHeight, padding: margin,
                        fontFamily: page.font ?? 'Times New Roman' }}>
            {page.border && page.border !== 'none' && (
              <div className={`pv-border is-${page.border}`}
                   style={{ inset: margin * 0.45 }} />
            )}
            {items.map(block => (
              <div key={block.key}
                   className={`pv-block pv-${block.kind}`
                            + (touched.includes(block.key) ? ' pv-flash' : '')}
                   style={runStyle(block.run, page)}>
                {block.run.text}
              </div>
            ))}
            <div className="pv-foot" style={{ bottom: margin * 0.38 }}>
              Page {i + 1} of {ready ? pages.length : 1}
            </div>
          </div>
        ))}
      </div>
    </>
  )
}

function shallowEqual(a, b) {
  const ka = Object.keys(a), kb = Object.keys(b)
  if (ka.length !== kb.length) return false
  return ka.every(k => b[k] && a[k].height === b[k].height
                        && a[k].lineHeight === b[k].lineHeight)
}
