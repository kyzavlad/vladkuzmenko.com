import { NextRequest, NextResponse } from 'next/server';

// Helper function to forward headers
function getForwardHeaders(request: NextRequest) {
  return {
    'Authorization': request.headers.get('Authorization') || '',
    'Content-Type': request.headers.get('Content-Type') || 'application/json',
  };
}

export async function GET(request: NextRequest) {
  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos`, {
      headers: getForwardHeaders(request),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to fetch videos');
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching videos:', error);
    return NextResponse.json(
      { error: 'Failed to fetch videos' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Validate file type
    const allowedTypes = ['video/mp4', 'video/quicktime', 'video/webm'];
    if (!allowedTypes.includes(file.type)) {
      return NextResponse.json(
        { error: 'Invalid file type. Only MP4, MOV, and WebM files are allowed.' },
        { status: 400 }
      );
    }

    // Validate file size (e.g., 100MB limit)
    const maxSize = 100 * 1024 * 1024; // 100MB in bytes
    if (file.size > maxSize) {
      return NextResponse.json(
        { error: 'File size exceeds 100MB limit' },
        { status: 400 }
      );
    }

    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos`, {
      method: 'POST',
      headers: getForwardHeaders(request),
      body: formData,
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to upload video');
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error uploading video:', error);
    return NextResponse.json(
      { error: 'Failed to upload video' },
      { status: 500 }
    );
  }
}

export async function PUT(request: NextRequest) {
  try {
    const { videoId, updates } = await request.json();
    
    if (!videoId) {
      return NextResponse.json(
        { error: 'Video ID is required' },
        { status: 400 }
      );
    }

    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos/${videoId}`, {
      method: 'PUT',
      headers: getForwardHeaders(request),
      body: JSON.stringify(updates),
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'Failed to update video');
    }
    
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error updating video:', error);
    return NextResponse.json(
      { error: 'Failed to update video' },
      { status: 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { videoId } = await request.json();
    
    if (!videoId) {
      return NextResponse.json(
        { error: 'Video ID is required' },
        { status: 400 }
      );
    }

    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/videos/${videoId}`, {
      method: 'DELETE',
      headers: getForwardHeaders(request),
    });
    
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || 'Failed to delete video');
    }
    
    return new NextResponse(null, { status: 204 });
  } catch (error) {
    console.error('Error deleting video:', error);
    return NextResponse.json(
      { error: 'Failed to delete video' },
      { status: 500 }
    );
  }
} 