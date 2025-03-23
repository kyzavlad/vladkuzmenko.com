import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// Security headers configuration
export const securityHeaders = {
  'X-DNS-Prefetch-Control': 'on',
  'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
  'X-Frame-Options': 'SAMEORIGIN',
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
  'Content-Security-Policy': `
    default-src 'self';
    script-src 'self' 'unsafe-inline' 'unsafe-eval';
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: blob: https:;
    connect-src 'self' wss: https:;
    font-src 'self';
    object-src 'none';
    base-uri 'self';
    form-action 'self';
    frame-ancestors 'none';
    block-all-mixed-content;
  `.replace(/\s+/g, ' ').trim(),
};

// CORS configuration
export const corsConfig = {
  'Access-Control-Allow-Origin': process.env.NEXT_PUBLIC_APP_URL || '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Max-Age': '86400',
};

// Rate limiting configuration
export const rateLimitConfig = {
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
};

// Password requirements
export const passwordRequirements = {
  minLength: 8,
  requireUppercase: true,
  requireLowercase: true,
  requireNumbers: true,
  requireSpecialChars: true,
};

// Session configuration
export const sessionConfig = {
  maxAge: 24 * 60 * 60 * 1000, // 24 hours
  httpOnly: true,
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'lax' as const,
  path: '/',
};

// API key validation
export const validateApiKey = (apiKey: string): boolean => {
  // Implement API key validation logic
  const apiKeyPattern = /^[a-zA-Z0-9_-]{32,64}$/;
  return apiKeyPattern.test(apiKey);
};

// Input sanitization
export const sanitizeInput = (input: string): string => {
  // Implement input sanitization logic
  return input
    .replace(/[<>]/g, '')
    .trim();
};

// XSS protection
export const xssProtection = (req: NextRequest) => {
  const response = NextResponse.next();
  
  // Add security headers
  Object.entries(securityHeaders).forEach(([key, value]) => {
    response.headers.set(key, value);
  });

  // Add CORS headers
  Object.entries(corsConfig).forEach(([key, value]) => {
    response.headers.set(key, value);
  });

  return response;
};

// GDPR compliance helpers
export const gdprHelpers = {
  // Generate data export
  generateDataExport: async (userId: string) => {
    // Implement data export logic
    return {
      userData: {},
      activityLogs: [],
      preferences: {},
      timestamp: new Date().toISOString(),
    };
  },

  // Delete user data
  deleteUserData: async (userId: string) => {
    // Implement data deletion logic
    return true;
  },

  // Update user consent
  updateUserConsent: async (userId: string, consent: Record<string, boolean>) => {
    // Implement consent update logic
    return true;
  },
};

// CCPA compliance helpers
export const ccpaHelpers = {
  // Generate data report
  generateDataReport: async (userId: string) => {
    // Implement data report generation logic
    return {
      personalInfo: {},
      usageData: {},
      sharingInfo: {},
      timestamp: new Date().toISOString(),
    };
  },

  // Opt out of data sharing
  optOutOfDataSharing: async (userId: string) => {
    // Implement opt-out logic
    return true;
  },
};

// Security middleware
export const securityMiddleware = async (req: NextRequest) => {
  const response = NextResponse.next();

  // Add security headers
  Object.entries(securityHeaders).forEach(([key, value]) => {
    response.headers.set(key, value);
  });

  // Add CORS headers
  Object.entries(corsConfig).forEach(([key, value]) => {
    response.headers.set(key, value);
  });

  // Rate limiting check
  const ip = req.ip ?? '127.0.0.1';
  // Implement rate limiting logic here

  // API key validation for protected routes
  if (req.nextUrl.pathname.startsWith('/api/protected')) {
    const apiKey = req.headers.get('x-api-key');
    if (!apiKey || !validateApiKey(apiKey)) {
      return new NextResponse('Invalid API key', { status: 401 });
    }
  }

  return response;
};

// Camera permission handling
export const cameraPermission = {
  request: async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      stream.getTracks().forEach(track => track.stop());
      return true;
    } catch {
      return false;
    }
  },

  check: async (): Promise<boolean> => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.some(device => device.kind === 'videoinput');
    } catch {
      return false;
    }
  },
};

// Payment security
export const paymentSecurity = {
  // Validate payment token
  validatePaymentToken: (token: string): boolean => {
    // Implement payment token validation logic
    return true;
  },

  // Encrypt sensitive payment data
  encryptPaymentData: (data: any): string => {
    // Implement payment data encryption logic
    return '';
  },

  // Decrypt sensitive payment data
  decryptPaymentData: (encryptedData: string): any => {
    // Implement payment data decryption logic
    return {};
  },
}; 