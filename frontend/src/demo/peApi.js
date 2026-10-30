// Backend calls for the Prompt Engineering page.
//
// These talk to the /pe/* routes, which wrap the vdoc-prompt project. They
// are kept apart from api.js so the other pages keep exactly the calls they
// had before.
import axios from 'axios'
import { API } from './api'

function friendlyError(err) {
  if (err.code === 'ERR_NETWORK') {
    return 'Cannot reach the backend at localhost:5050 — start it with '
         + '.\\.venv\\Scripts\\python.exe backend\\app.py'
  }
  return err.response?.data?.error ?? err.message
}

/** Which models and techniques the dropdowns should offer. */
export async function fetchPeMeta() {
  try {
    const { data } = await axios.get(`${API}/pe/meta`)
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

/** One command, one model, one technique. */
export async function peRun(command, model, technique) {
  try {
    const { data } = await axios.post(`${API}/pe/run`, { command, model, technique })
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

/** The same command under three prompts. */
export async function peCompare(command, model) {
  try {
    const { data } = await axios.post(`${API}/pe/compare`, { command, model })
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}

/** The saved experiment tables, read from runs.csv. */
export async function peResults() {
  try {
    const { data } = await axios.get(`${API}/pe/results`)
    return { ok: true, ...data }
  } catch (err) {
    return { ok: false, error: friendlyError(err) }
  }
}
