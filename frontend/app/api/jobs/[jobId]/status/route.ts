import { NextRequest, NextResponse } from 'next/server';
import { processingJobs } from '@/app/api/videos/[videoId]/process/route';

export async function GET(
  _req: NextRequest,
  { params }: { params: { jobId: string } }
) {
  try {
    const { jobId } = params;
    const job = processingJobs.get(jobId);

    if (!job) {
      return NextResponse.json(
        { error: 'Job not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      status: job.status,
      progress: job.progress,
      message: getStatusMessage(job),
      result: job.status === 'completed' ? job.result : undefined,
      error: job.status === 'error' ? job.error : undefined,
    });
  } catch (error) {
    console.error('Error fetching job status:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

function getStatusMessage(job: {
  status: string;
  progress: number;
  error?: string;
}): string {
  switch (job.status) {
    case 'queued':
      return 'Waiting to start processing...';
    case 'processing':
      return `Processing video... ${Math.round(job.progress)}%`;
    case 'completed':
      return 'Video processing completed successfully';
    case 'error':
      return `Error: ${job.error || 'Unknown error occurred'}`;
    default:
      return 'Unknown status';
  }
} 