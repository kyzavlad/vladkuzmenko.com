import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, password } = body;

    // Basic validation
    if (!email || !password) {
      return NextResponse.json(
        { error: "Email and password are required" },
        { status: 400 }
      );
    }

    // For demo purposes, let's check for a specific test account
    // In production, you would validate against your auth service
    if (email === "test@example.com" && password === "password") {
      const response = NextResponse.json({ success: true });
      
      response.cookies.set('auth-token', 'dummy-token', {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        maxAge: 60 * 60 * 24 * 7 // 1 week
      });
      return response;
    }

    return NextResponse.json(
      { error: "Invalid email or password" },
      { status: 401 }
    );
  } catch (error) {
    console.error('Login failed:', error);
    return NextResponse.json(
      { error: "An unexpected error occurred" },
      { status: 500 }
    );
  }
}