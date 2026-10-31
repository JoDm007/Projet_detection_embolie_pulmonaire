// frontend/js/auth.js — écran de connexion et déconnexion
import { api } from './api.js';

export function initLogin(onLoggedIn) {
  const form = document.getElementById('login-form');
  const error = document.getElementById('login-error');

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    error.hidden = true;
    const button = form.querySelector('button[type=submit]');
    button.disabled = true;
    try {
      const user = await api.login(form.username.value.trim(), form.password.value);
      form.password.value = '';
      onLoggedIn(user);
    } catch (err) {
      error.textContent = err.message;
      error.hidden = false;
    } finally {
      button.disabled = false;
    }
  });
}

export function initLogout(onLoggedOut) {
  document.getElementById('logout-btn').addEventListener('click', async () => {
    try { await api.logout(); } catch { /* session déjà close */ }
    onLoggedOut();
  });
}
