const CACHE_NAME = 'fitness-platform-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/offline.html',
  '/assets/css/main.css',
  '/assets/js/main.js',
  '/assets/images/logo.png',
  '/assets/images/offline-icon.png',
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    })
  );
});

// Fetch event - handle network requests
self.addEventListener('fetch', (event) => {
  // Skip cross-origin requests
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((response) => {
      // Return cached response if found
      if (response) {
        return response;
      }

      // Clone the request because it can only be used once
      const fetchRequest = event.request.clone();

      // Make network request
      return fetch(fetchRequest).then((response) => {
        // Check if response is valid
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }

        // Clone the response because it can only be used once
        const responseToCache = response.clone();

        // Cache the response
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });

        return response;
      }).catch(() => {
        // If offline and request is for HTML, return offline page
        if (event.request.mode === 'navigate') {
          return caches.match('/offline.html');
        }
      });
    })
  );
});

// Background sync for offline form submissions
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-workouts') {
    event.waitUntil(syncWorkouts());
  }
});

// Push notification handling
self.addEventListener('push', (event) => {
  const options = {
    body: event.data.text(),
    icon: '/assets/images/logo.png',
    badge: '/assets/images/badge.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Workout',
        icon: '/assets/images/checkmark.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/assets/images/xmark.png'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification('Fitness Platform', options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/workouts')
    );
  }
});

// Helper function to sync workouts
async function syncWorkouts() {
  try {
    const db = await openIndexedDB();
    const workouts = await db.getAll('offline-workouts');
    
    for (const workout of workouts) {
      await fetch('/api/workouts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(workout),
      });
      
      await db.delete('offline-workouts', workout.id);
    }
  } catch (error) {
    console.error('Error syncing workouts:', error);
  }
}

// Helper function to open IndexedDB
function openIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('fitness-platform-db', 1);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('offline-workouts')) {
        db.createObjectStore('offline-workouts', { keyPath: 'id' });
      }
    };
  });
} 