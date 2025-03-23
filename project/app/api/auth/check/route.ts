import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    // Get auth token from cookie
    const token = request.cookies.get('auth-token')?.value

    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    // Verify token with your auth service
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/verify`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })

    if (!response.ok) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 })
    }

    return NextResponse.json({ authenticated: true })
  } catch (error) {
    console.error('Auth check failed:', error)
    return NextResponse.json({ error: 'Auth check failed' }, { status: 500 })
  }
}