import { NextRequest, NextResponse } from 'next/server';
import { Effect } from '@/components/video/editor/EffectsPanel';
import { processingJobs } from '../process/route';

// In-memory effect storage (replace with database in production)
const videoEffects = new Map<string, Effect[]>();

export async function GET(
  _req: NextRequest,
  { params }: { params: { videoId: string } }
) {
  try {
    const { videoId } = params;
    const effects = videoEffects.get(videoId) || [];
    return NextResponse.json(effects);
  } catch (error) {
    console.error('Error fetching effects:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(
  req: NextRequest,
  { params }: { params: { videoId: string } }
) {
  try {
    const { videoId } = params;
    const effect: Effect = await req.json();

    // Validate effect data
    if (!effect.type || !effect.name) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Generate effect ID
    effect.id = `effect-${Date.now()}`;

    // Store effect
    const effects = videoEffects.get(videoId) || [];
    effects.push(effect);
    videoEffects.set(videoId, effects);

    // Create a processing job for the effect
    const jobId = `job-${Date.now()}`;
    const job = {
      id: jobId,
      videoId,
      status: 'queued' as const,
      progress: 0,
      settings: {}, // No settings needed for single effect
      effects: [effect],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    processingJobs.set(jobId, job);

    // Start processing in the background
    processEffect(job).catch(console.error);

    return NextResponse.json({ jobId, effect }, { status: 201 });
  } catch (error) {
    console.error('Error adding effect:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: { videoId: string } }
) {
  try {
    const { videoId } = params;
    const { effectId, updates } = await req.json();

    // Validate request
    if (!effectId || !updates) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Find and update effect
    const effects = videoEffects.get(videoId) || [];
    const effectIndex = effects.findIndex((e) => e.id === effectId);

    if (effectIndex === -1) {
      return NextResponse.json(
        { error: 'Effect not found' },
        { status: 404 }
      );
    }

    const updatedEffect = {
      ...effects[effectIndex],
      ...updates,
    };
    effects[effectIndex] = updatedEffect;
    videoEffects.set(videoId, effects);

    // Create a processing job for the updated effect
    const jobId = `job-${Date.now()}`;
    const job = {
      id: jobId,
      videoId,
      status: 'queued' as const,
      progress: 0,
      settings: {}, // No settings needed for single effect
      effects: [updatedEffect],
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    processingJobs.set(jobId, job);

    // Start processing in the background
    processEffect(job).catch(console.error);

    return NextResponse.json({ jobId, effect: updatedEffect });
  } catch (error) {
    console.error('Error updating effect:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: { videoId: string } }
) {
  try {
    const { videoId } = params;
    const { effectId } = await req.json();

    if (!effectId) {
      return NextResponse.json(
        { error: 'Missing effect ID' },
        { status: 400 }
      );
    }

    // Remove effect
    const effects = videoEffects.get(videoId) || [];
    const updatedEffects = effects.filter((e) => e.id !== effectId);
    videoEffects.set(videoId, updatedEffects);

    // Create a processing job to remove the effect
    const jobId = `job-${Date.now()}`;
    const job = {
      id: jobId,
      videoId,
      status: 'queued' as const,
      progress: 0,
      settings: {}, // No settings needed for effect removal
      effects: [], // Empty effects list indicates removal
      createdAt: new Date(),
      updatedAt: new Date(),
    };

    processingJobs.set(jobId, job);

    // Start processing in the background
    processEffect(job).catch(console.error);

    return NextResponse.json({ jobId });
  } catch (error) {
    console.error('Error deleting effect:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

async function processEffect(job: {
  id: string;
  videoId: string;
  status: 'queued' | 'processing' | 'completed' | 'error';
  progress: number;
  effects: Effect[];
}) {
  try {
    // Update job status
    job.status = 'processing';
    job.progress = 0;
    processingJobs.set(job.id, job);

    // Simulate effect processing
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Update progress
    job.progress = 50;
    processingJobs.set(job.id, job);

    // Simulate more processing
    await new Promise((resolve) => setTimeout(resolve, 2000));

    // Complete the job
    job.status = 'completed';
    job.progress = 100;
    processingJobs.set(job.id, job);
  } catch (error) {
    // Update job with error status
    job.status = 'error';
    processingJobs.set(job.id, job);
    throw error;
  }
} 