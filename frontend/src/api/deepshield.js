import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

const api = axios.create({ baseURL: BASE_URL })

export async function detectVideo(file, onProgress) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/detect/video', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress && onProgress(Math.round((e.loaded * 100) / e.total)),
  })
  return data
}

export async function detectImage(file, onProgress) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/detect/image', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress && onProgress(Math.round((e.loaded * 100) / e.total)),
  })
  return data
}

export async function detectAudio(file, onProgress) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/detect/audio', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: e => onProgress && onProgress(Math.round((e.loaded * 100) / e.total)),
  })
  return data
}

export async function getHealth() {
  const { data } = await api.get('/health')
  return data
}

export async function getHistory(limit = 50) {
  const { data } = await api.get(`/history?limit=${limit}`)
  return data
}

export async function getStats() {
  const { data } = await api.get('/stats')
  return data
}
