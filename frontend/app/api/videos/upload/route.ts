import { NextRequest, NextResponse } from 'next/server';
import { writeFile } from 'fs/promises';
import { join } from 'path';
import { mkdir } from 'fs/promises';

// In-memory storage for video metadata (replace with database in production)
const videos = new Map<string, {
  id: string;
  filename: string;
  size: number;
  mimeType: string;
  uploadedAt: Date;
  duration?: number;
  resolution?: {
    width: number;
    height: number;
  };
}>();

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json(
        { error: 'No file uploaded' },
        { status: 400 }
      );
    }

    // Validate file type
    if (!file.type.startsWith('video/')) {
      return NextResponse.json(
        { error: 'Invalid file type. Only video files are allowed.' },
        { status: 400 }
      );
    }

    // Generate unique ID and filename
    const videoId = `video-${Date.now()}`;
    const filename = `${videoId}-${file.name.replace(/[^a-zA-Z0-9.-]/g, '_')}`;

    // Create uploads directory if it doesn't exist
    const uploadsDir = join(process.cwd(), 'public', 'uploads');
    await mkdir(uploadsDir, { recursive: true });

    // Save file to disk
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    const filepath = join(uploadsDir, filename);
    await writeFile(filepath, buffer);

    // Store video metadata
    const videoData = {
      id: videoId,
      filename,
      size: file.size,
      mimeType: file.type,
      uploadedAt: new Date(),
    };
    videos.set(videoId, videoData);

    // Start processing video metadata in the background
    processVideoMetadata(videoId, filepath).catch(console.error);

    return NextResponse.json({
      id: videoId,
      url: `/uploads/${filename}`,
    }, { status: 201 });
  } catch (error) {
    console.error('Error uploading video:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    // Return list of uploaded videos
    const videoList = Array.from(videos.values()).map((video) => ({
      id: video.id,
      filename: video.filename,
      size: video.size,
      mimeType: video.mimeType,
      uploadedAt: video.uploadedAt,
      duration: video.duration,
      resolution: video.resolution,
      url: `/uploads/${video.filename}`,
    }));

    return NextResponse.json(videoList);
  } catch (error) {
    console.error('Error fetching videos:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

async function processVideoMetadata(videoId: string, filepath: string) {
  try {
    // Simulate video metadata extraction
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Update video metadata with mock duration and resolution
    const video = videos.get(videoId);
    if (video) {
      video.duration = 120; // Mock duration in seconds
      video.resolution = {
        width: 1920,
        height: 1080,
      };
      videos.set(videoId, video);
    }
  } catch (error) {
    console.error('Error processing video metadata:', error);
  }
} 