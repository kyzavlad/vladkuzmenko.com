import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // Get allowed origins from env
  const allowedOrigins = process.env.ALLOWED_ORIGINS?.split(',') || [];
  
  // Check if origin is allowed
  const origin = request.headers.get('origin');
  if (origin && !allowedOrigins.includes(origin)) {
    return new NextResponse(null, {
      status: 403,
      statusText: 'Forbidden',
      headers: {
        'Content-Type': 'text/plain',
      },
    });
  }

  // Skip authentication for /platform
  if (request.nextUrl.pathname.startsWith('/platform')) {
    return NextResponse.next();
  }

  // Handle CORS preflight requests
  if (request.method === 'OPTIONS') {
    return new NextResponse(null, {
      headers: {
        'Access-Control-Allow-Origin': origin || '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    });
  }

  // Add CORS headers to all responses
  const response = NextResponse.next();
  response.headers.set('Access-Control-Allow-Origin', origin || '*');
  return response;
}

export const config = {
  matcher: '/api/:path*',
}