// frontend/js/dom.js — petits utilitaires DOM partagés

// Crée un élément ; le texte passe par textContent : jamais d'HTML injecté
export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
