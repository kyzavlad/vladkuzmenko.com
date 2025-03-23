/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { 
    unoptimized: true,
    domains: ['images.unsplash.com']
  },
  assetPrefix: '/',
  basePath: '',
  webpack: (config) => {
    return config;
  }
};

module.exports = nextConfig;