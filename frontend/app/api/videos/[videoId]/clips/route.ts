import { NextRequest, NextResponse } from 'next/server';
import { v4 as uuidv4 } from 'uuid';

interface ClipConfig {
  duration: number;
  aspectRatio: '9:16' | '16:9' | '1:1';
  addCaptions: boolean;
  addBackground: boolean;
  addIntro: boolean;
  addOutro: boolean;
  musicStyle: string;
  captionStyle: string;
}

interface ClipJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  error?: string;
  outputUrl?: string;
  config: ClipConfig;
  createdAt: string;
  updatedAt: string;
}

// In-memory storage for jobs (replace with database in production)
const clipJobs = new Map<string, ClipJob>();

export async function POST(
  req: NextRequest,
  { params }: { params: { videoId: string } }
) {
  try {
    const config: ClipConfig = await req.json();
    const jobId = uuidv4();

    const job: ClipJob = {
      id: jobId,
      status: 'pending',
      progress: 0,
      config,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    clipJobs.set(jobId, job);

    // Start processing in the background
    processClip(params.videoId, jobId);

    return NextResponse.json(job);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to create clip job' },
      { status: 500 }
    );
  }
}

export async function GET(
  _req: NextRequest,
  { params }: { params: { videoId: string } }
) {
  try {
    // Filter jobs for this video (in production, query database)
    const jobs = Array.from(clipJobs.values()).filter(
      (job) => job.id.startsWith(params.videoId)
    );

    return NextResponse.json(jobs);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to list clips' },
      { status: 500 }
    );
  }
}

async function processClip(videoId: string, jobId: string) {
  const job = clipJobs.get(jobId);
  if (!job) return;

  try {
    // Update status to processing
    job.status = 'processing';
    job.updatedAt = new Date().toISOString();
    clipJobs.set(jobId, job);

    // Simulate processing steps
    await simulateProcessing(job);

    // Update job with success
    job.status = 'completed';
    job.progress = 100;
    job.outputUrl = `/api/videos/${videoId}/clips/${jobId}/output.mp4`;
    job.updatedAt = new Date().toISOString();
    clipJobs.set(jobId, job);
  } catch (error) {
    // Update job with error
    job.status = 'failed';
    job.error = error instanceof Error ? error.message : 'Unknown error';
    job.updatedAt = new Date().toISOString();
    clipJobs.set(jobId, job);
  }
}

async function simulateProcessing(job: ClipJob) {
  const steps = [
    'Analyzing video content',
    'Extracting relevant segments',
    'Applying transitions',
    'Adding captions',
    'Adding background music',
    'Rendering final clip',
  ];

  for (const [index, step] of steps.entries()) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    job.progress = Math.round(((index + 1) / steps.length) * 100);
    job.updatedAt = new Date().toISOString();
    clipJobs.set(job.id, { ...job });
  }
} 