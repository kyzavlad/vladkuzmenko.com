module.exports = {
  // Project configuration
  project: {
    name: 'ai-video-platform',
    type: 'nextjs',
    rootDir: '.',
    buildCommand: 'npm run build',
    startCommand: 'npm start',
    outputDir: '.next',
    nodeVersion: '18.x',
  },

  // Route configuration
  routes: [
    {
      src: '/',
      dest: '/platform',
    },
    {
      src: '/platform',
      dest: '/platform/index.html',
    },
    {
      src: '/platform/(.*)',
      dest: '/platform/$1',
    },
    {
      src: '/api/(.*)',
      dest: '/api/$1',
    },
  ],

  // Middleware configuration
  middleware: {
    auth: {
      exclude: ['/api/auth/login', '/api/auth/register', '/api/auth/reset-password'],
    },
    rateLimit: {
      window: 60000,
      max: 100,
      skipSuccessfulRequests: true,
    },
  },

  // Build optimization
  build: {
    env: {
      NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
      NEXT_PUBLIC_WEBSOCKET_URL: process.env.NEXT_PUBLIC_WEBSOCKET_URL,
      NEXT_PUBLIC_STRIPE_PUBLIC_KEY: process.env.NEXT_PUBLIC_STRIPE_PUBLIC_KEY,
    },
    optimization: {
      minify: true,
      compress: true,
      splitChunks: true,
      treeshake: true,
    },
  },

  // Cache configuration
  cache: {
    patterns: [
      {
        source: '/platform/static/**/*',
        headers: {
          'Cache-Control': 'public, max-age=31536000, immutable',
        },
      },
      {
        source: '/api/**/*',
        headers: {
          'Cache-Control': 'no-cache, no-store, must-revalidate',
        },
      },
    ],
  },

  // Security headers
  headers: {
    '/*': {
      'X-Frame-Options': 'DENY',
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'strict-origin-when-cross-origin',
      'Content-Security-Policy': "default-src 'self' *.bolt.new; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https:; connect-src 'self' wss: https:;",
    },
  },

  // Environment configuration
  environment: {
    production: {
      // Production-specific settings
      NODE_ENV: 'production',
      NEXT_TELEMETRY_DISABLED: '1',
    },
    preview: {
      // Preview deployment settings
      NODE_ENV: 'production',
      NEXT_TELEMETRY_DISABLED: '1',
    },
  },

  // Monitoring and analytics
  monitoring: {
    errorTracking: true,
    performance: true,
    analytics: true,
  },

  // CI/CD configuration
  ci: {
    build: {
      steps: [
        {
          name: 'Install dependencies',
          command: 'npm ci',
        },
        {
          name: 'Run tests',
          command: 'npm test',
        },
        {
          name: 'Build project',
          command: 'npm run build',
        },
      ],
    },
    deploy: {
      steps: [
        {
          name: 'Verify deployment',
          command: 'node scripts/verify-all.js',
        },
        {
          name: 'Monitor performance',
          command: 'node scripts/monitor-performance.js',
        },
      ],
    },
  },

  // WebSocket configuration
  websocket: {
    enabled: true,
    path: '/ws',
    maxPayload: '1mb',
  },
}; 