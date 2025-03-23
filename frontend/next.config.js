/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  poweredByHeader: false,
  basePath: '/platform',

  // Configure image domains
  images: {
    domains: [
      'localhost',
      'api.bolt.new',
      'storage.googleapis.com',
      'cdn.bolt.new',
    ],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },

  // Handle routing
  async rewrites() {
    return [
      {
        source: '/platform',
        destination: '/platform',
      },
      {
        source: '/',
        destination: '/platform',
      },
    ];
  },

  // Redirect configuration
  async redirects() {
    return [
      {
        source: '/',
        destination: '/platform',
        permanent: true,
      },
    ];
  },

  // Headers configuration
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload',
          },
        ],
      },
    ];
  },

  // Webpack configuration
  webpack: (config, { dev, isServer }) => {
    // Optimize bundle size
    if (!dev && !isServer) {
      config.optimization = {
        ...config.optimization,
        splitChunks: {
          chunks: 'all',
          minSize: 20000,
          maxSize: 244000,
          cacheGroups: {
            vendor: {
              test: /[\\/]node_modules[\\/]/,
              name: 'vendor',
              chunks: 'all',
            },
          },
        },
      };
    }

    config.module.rules.push({
      test: /\.(mp4|webm|ogg|mp3|wav|flac|aac)(\?.*)?$/,
      type: 'asset/resource',
      generator: {
        filename: 'static/media/[name].[hash][ext]',
      },
    });

    return config;
  },

  // Environment variables configuration
  env: {
    NEXT_PUBLIC_APP_VERSION: process.env.npm_package_version,
  },

  // Experimental features
  experimental: {
    optimizeCss: true,
    scrollRestoration: true,
  },
};

module.exports = nextConfig; 