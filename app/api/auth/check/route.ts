import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function GET() {
  const cookieStore = cookies();
  const token = cookieStore.get('auth_token');

  if (!token) {
    return NextResponse.json(
      { error: 'Unauthorized' },
      { status: 401 }
    );
  }

  try {
    // Verify token with backend
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/auth/verify`, {
      headers: {
        Authorization: `Bearer ${token.value}`,
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: 'Invalid token' },
        { status: 401 }
      );
    }

    const userData = await response.json();

    return NextResponse.json({
      authenticated: true,
      user: userData,
    });
  } catch (error) {
    console.error('Auth check failed:', error);
    return NextResponse.json(
      { error: 'Authentication check failed' },
      { status: 500 }
    );
  }
}

// This is a simple example. In a real application, you would:
// 1. Hash passwords
// 2. Use a database
// 3. Implement proper session management
// 4. Add rate limiting
// 5. Add proper error handling

const VALID_EMAIL = 'admin@example.com';
const VALID_PASSWORD = 'password123';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { email, password } = body;

    if (email === VALID_EMAIL && password === VALID_PASSWORD) {
      // In a real application, you would:
      // 1. Generate a JWT or session token
      // 2. Set secure HTTP-only cookies
      // 3. Store session information
      return NextResponse.json({ success: true });
    }

    return NextResponse.json(
      { error: 'Invalid credentials' },
      { status: 401 }
    );
  } catch (error) {
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 