// Backend calls for the VDOC 2.0 demo pages.
import axios from 'axios'

export const API = 'http://localhost:5050'

function friendlyError(err) {
  if (err.code === 'ERR_NETWORK') {
    return 'Cannot reach the backend at localhost:5050 — start it with '
         + '.\\.venv\\Scripts\\python.exe backend\\app.py'
  }
  return err.response?.data?.error ?? err.message
}

/** VDOC 1.0: one command in, at most one action out. */
export async function parseOld(command) {
  try {
    const { data } = await axios.post(`${API}/parse`, { command })
    return {
      ok: true,
      actions: [{
        index: 0,
        chunk_text: command,
        join_before: null,
        action: data.action,
        confidence: data.confidence,
        target: (data.from || data.to)
          ? { kind: 'anchor', from: data.from, to: data.to,
              complete: Boolean(data.from && data.to) }
          : null,
        model_from: data.from,
        model_to: data.to,
        shared_target: false,
        label: data.action
          ? `${data.action}${data.from && data.to ? ` from ${data.from} to ${data.to}` : ''}`
          : 'nothing parsed',
      }],
      raw: data,
    }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

/** VDOC 2.0: one command in, an ordered list of actions out. */
export async function parseMulti(command, lineCount) {
  try {
    const { data } = await axios.post(`${API}/parse-multi`, {
      command, line_count: lineCount,
    })
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

/** Local Whisper transcription of a recorded blob. */
export async function transcribe(blob) {
  const form = new FormData()
  form.append('audio', blob, 'recording.webm')
  try {
    const { data } = await axios.post(`${API}/transcribe`, form)
    return { ok: true, text: (data.text || '').trim(), diagnostics: data.diagnostics }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

/**
 * Check an utterance for misheard command words.
 * Saved corrections come back already applied; the rest are suggestions
 * waiting for a yes or no.
 */
export async function suggestCorrections(command) {
  try {
    const { data } = await axios.post(`${API}/suggest-corrections`, { command })
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

/** Remember a correction the user accepted. */
export async function confirmCorrection(word, suggestion, command) {
  try {
    const { data } = await axios.post(`${API}/confirm-correction`, {
      word, suggestion, command,
    })
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

export async function fetchConfig() {
  try {
    const { data } = await axios.get(`${API}/multi-action/config`)
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}
