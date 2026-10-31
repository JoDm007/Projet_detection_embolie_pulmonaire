// frontend/js/api.js — SEUL module qui appelle le backend. Si l'API change, seul ce fichier bouge.

const CSRF_HEADER = { 'X-Requested-With': 'PESeg' };

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function request(method, path, { json, form } = {}) {
  const options = { method, credentials: 'same-origin', headers: { ...CSRF_HEADER } };
  if (json !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(json);
  } else if (form !== undefined) {
    options.body = form;
  }

  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError(0, 'Serveur injoignable. Est-il lancé ?');
  }

  if (response.status === 401 && path !== '/api/auth/login') {
    window.dispatchEvent(new CustomEvent('auth:expired'));
  }
  if (!response.ok) {
    let message = `Erreur ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') message = body.detail;
      else if (Array.isArray(body.detail)) message = 'Requête invalide.';
      if (body.error_id) message += ` (réf. ${body.error_id})`;
    } catch { /* corps non JSON */ }
    throw new ApiError(response.status, message);
  }
  if (response.status === 204) return null;
  return response.json();
}

export const api = {
  health: () => request('GET', '/api/health'),

  login: (username, password) => request('POST', '/api/auth/login', { json: { username, password } }),
  logout: () => request('POST', '/api/auth/logout'),
  me: () => request('GET', '/api/auth/me'),

  demoCases: () => request('GET', '/api/cases/demo'),

  listAnalyses: () => request('GET', '/api/analyses'),
  getAnalysis: (id) => request('GET', `/api/analyses/${encodeURIComponent(id)}`),
  findings: (id) => request('GET', `/api/analyses/${encodeURIComponent(id)}/findings`),
  deleteAnalysis: (id) => request('DELETE', `/api/analyses/${encodeURIComponent(id)}`),

  submitDemo: (caseId, engine) =>
    request('POST', `/api/analyses/demo/${encodeURIComponent(caseId)}`, { json: { engine } }),

  submitUpload: (file, engine) => {
    const form = new FormData();
    form.append('file', file);
    form.append('engine', engine);
    return request('POST', '/api/analyses', { form });
  },

  // URL des volumes lus directement par NiiVue (cookie de session envoyé, même origine)
  fileUrl: (id, name) => `/api/analyses/${encodeURIComponent(id)}/${name}`,
};
