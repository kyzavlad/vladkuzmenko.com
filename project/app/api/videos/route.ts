import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const token = request.cookies.get('auth-token')?.value
    
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/videos`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Failed to fetch videos:', error)
    return NextResponse.json({ error: 'Failed to fetch videos' }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const token = request.cookies.get('auth-token')?.value
    const body = await request.json()
    
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/videos`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Failed to create video:', error)
    return NextResponse.json({ error: 'Failed to create video' }, { status: 500 })
  }
}