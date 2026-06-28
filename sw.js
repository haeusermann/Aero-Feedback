/* =============================================================================
 * Service worker - offline support
 * -----------------------------------------------------------------------------
 * The whole app is a handful of static files with NO runtime CDN dependencies,
 * so we can pre-cache everything and run fully offline (important on the
 * mountain without reception). Strategy: cache-first with a network fallback.
 *
 * Bump CACHE_VERSION whenever any cached file changes so clients pick up the
 * new build.
 * ===========================================================================*/

const CACHE_VERSION = 'frontflaeche-v1';

const ASSETS = [
  './',
  './index.html',
  './app.js',
  './style.css',
  './manifest.webmanifest',
  './icons/icon-192.png',
  './icons/icon-512.png',
  './icons/icon-512-maskable.png',
];

// Pre-cache the app shell on install.
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(ASSETS))
  );
  self.skipWaiting();
});

// Clean up old caches on activate.
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Cache-first: serve from cache, fall back to network, and quietly update the
// cache with any fresh network responses for next time.
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    caches.match(event.request).then((cached) => {
      const network = fetch(event.request)
        .then((resp) => {
          // Only cache same-origin, OK responses.
          if (resp && resp.ok && new URL(event.request.url).origin === self.location.origin) {
            const copy = resp.clone();
            caches.open(CACHE_VERSION).then((cache) => cache.put(event.request, copy));
          }
          return resp;
        })
        .catch(() => cached); // offline: fall back to whatever we have
      return cached || network;
    })
  );
});
