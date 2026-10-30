// Backend calls for the VDOC Studio page.
import axios from 'axios'
import { API } from './api'

function friendlyError(err) {
  if (err.code === 'ERR_NETWORK') {
    return 'Cannot reach the backend at localhost:5050 — start it with '
         + '.\\.venv\\Scripts\\python.exe backend\\app.py'
  }
  return err.response?.data?.error ?? err.message
}

export async function studioState() {
  try {
    const { data } = await axios.get(`${API}/studio/state`)
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

/** One spoken or typed sentence in; the changed document out. */
export async function studioCommand(text) {
  try {
    const { data } = await axios.post(`${API}/studio/command`, { text })
    return { ok: true, ...data }
  } catch (err) {
    // A 500 still carries the document, so the preview never goes blank.
    const body = err.response?.data
    if (body?.doc) return { ok: false, error: friendlyError(err), ...body }
    return { ok: false, error: friendlyError(err) }
  }
}

/** Apply one of the buttons offered after a clarify. Costs no LLM call. */
export async function studioPick(index) {
  try {
    const { data } = await axios.post(`${API}/studio/pick`, { index })
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

export async function studioUndo() {
  try {
    const { data } = await axios.post(`${API}/studio/undo`)
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

export async function studioReset() {
  try {
    const { data } = await axios.post(`${API}/studio/reset`)
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

/** The browser downloads straight from the backend, rebuilt on each request. */
export function downloadUrl(format) {
  return `${API}/studio/download?format=${format}`
}
