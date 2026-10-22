// Microphone capture for the demo pages — the same MediaRecorder flow the
// existing editor uses, lifted into a hook so more than one page can use it.
import { useCallback, useRef, useState } from 'react'
import { transcribe } from './api'

const AUTO_STOP_MS = 6000

export function useVoiceInput(onTranscribed) {
  const [recording, setRecording] = useState(false)
  const [voiceStatus, setVoiceStatus] = useState(null)

  const mediaRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)

  const toggleMic = useCallback(async () => {
    if (recording) {
      clearTimeout(timerRef.current)
      mediaRef.current?.stop()
      setRecording(false)
      return
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setVoiceStatus('This browser has no microphone support — try Chrome.')
      return
    }

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (err) {
      const name = err?.name ?? 'UnknownError'
      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        setVoiceStatus('Microphone blocked — allow it from the address bar.')
      } else if (name === 'NotFoundError') {
        setVoiceStatus('No microphone found.')
      } else {
        setVoiceStatus(`Mic error: ${err?.message ?? name}`)
      }
      return
    }

    chunksRef.current = []
    let recorder
    try {
      recorder = new MediaRecorder(stream)
    } catch (err) {
      stream.getTracks().forEach(t => t.stop())
      setVoiceStatus(`MediaRecorder failed: ${err?.message ?? err}`)
      return
    }
    mediaRef.current = recorder

    recorder.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
    recorder.onstop = async () => {
      stream.getTracks().forEach(t => t.stop())
      setVoiceStatus('Transcribing…')
      const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
      const result = await transcribe(blob)
      if (!result.ok) { setVoiceStatus(result.error); return }
      if (!result.text) { setVoiceStatus('No speech detected.'); return }
      setVoiceStatus(null)
      onTranscribed(result.text)
    }

    recorder.start()
    setRecording(true)
    setVoiceStatus('Listening… click again to stop, or wait 6 seconds.')
    timerRef.current = setTimeout(() => {
      if (recorder.state === 'recording') { recorder.stop(); setRecording(false) }
    }, AUTO_STOP_MS)
  }, [recording, onTranscribed])

  return { recording, voiceStatus, toggleMic }
}
