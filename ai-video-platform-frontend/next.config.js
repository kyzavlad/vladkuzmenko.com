/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ['localhost', 'api.clip-generation-service.com'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**.amazonaws.com',
      },
    ],
  },
  async redirects() {
    return [
      {
        source: '/',
        destination: '/platform',
        permanent: true,
      },
    ];
  },
};

module.exports = nextConfig; 