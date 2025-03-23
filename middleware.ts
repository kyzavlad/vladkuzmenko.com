import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { securityMiddleware, corsConfig } from './lib/config/security';

// Rate limiting store
const rateLimitStore = new Map<string, { count: number; timestamp: number }>();

// Rate limiting middleware
const rateLimit = (req: NextRequest) => {
  const ip = req.ip ?? '127.0.0.1';
  const now = Date.now();
  const windowMs = 15 * 60 * 1000; // 15 minutes

  // Clean up old entries
  Array.from(rateLimitStore.entries()).forEach(([key, value]) => {
    if (now - value.timestamp > windowMs) {
      rateLimitStore.delete(key);
    }
  });

  // Check rate limit
  const current = rateLimitStore.get(ip);
  if (current) {
    if (now - current.timestamp > windowMs) {
      rateLimitStore.set(ip, { count: 1, timestamp: now });
    } else if (current.count >= 100) {
      return new NextResponse('Too many requests', { status: 429 });
    } else {
      current.count++;
    }
  } else {
    rateLimitStore.set(ip, { count: 1, timestamp: now });
  }

  return null;
};

// Helper function to verify JWT token
async function verifyToken(token: string) {
  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/auth/verify`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return response.ok;
  } catch (error) {
    console.error('Token verification failed:', error);
    return false;
  }
}

// Main middleware function
export async function middleware(req: NextRequest) {
  // Apply rate limiting
  const rateLimitResult = rateLimit(req);
  if (rateLimitResult) return rateLimitResult;

  // Apply security middleware
  const response = await securityMiddleware(req);

  // Add CORS headers for API routes
  if (req.nextUrl.pathname.startsWith('/api')) {
    Object.entries(corsConfig).forEach(([key, value]) => {
      response.headers.set(key, value);
    });
  }

  // Block access to sensitive files
  if (req.nextUrl.pathname.match(/\.(env|config|json|md|git)$/)) {
    return new NextResponse('Not found', { status: 404 });
  }

  // Add security headers to all responses
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('X-Frame-Options', 'SAMEORIGIN');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');

  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    response.headers.set('Access-Control-Allow-Origin', '*');
    response.headers.set('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    response.headers.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    response.headers.set('Access-Control-Max-Age', '86400');
  }

  // Check authentication for /platform routes
  if (req.nextUrl.pathname.startsWith('/platform')) {
    const token = req.cookies.get('auth_token')?.value;
    
    // Exclude public pages from authentication
    const publicPages = ['/platform/login', '/platform/register', '/platform/reset-password'];
    if (publicPages.some(page => req.nextUrl.pathname.startsWith(page))) {
      return response;
    }

    if (!token || !(await verifyToken(token))) {
      const loginUrl = new URL('/platform/login', req.url);
      loginUrl.searchParams.set('returnUrl', req.nextUrl.pathname);
      return NextResponse.redirect(loginUrl);
    }
  }

  // Handle WebSocket upgrade requests
  if (req.nextUrl.pathname.startsWith('/ws')) {
    response.headers.set('Upgrade', 'websocket');
    return response;
  }

  // If the request is for the login page, set a cookie to track that they've visited
  if (req.nextUrl.pathname === '/login') {
    response.cookies.set('visited_login', 'true', {
      httpOnly: true,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
    });
    return response;
  }

  return response;
}

// Configure which routes to run middleware on
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|public).*)',
  ],
}; 