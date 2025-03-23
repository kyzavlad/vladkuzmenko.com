import { NextRequest, NextResponse } from 'next/server';
import { v4 as uuidv4 } from 'uuid';

interface TranslationConfig {
  sourceLanguage: string;
  sourceDialect?: string;
  targetLanguages: Array<{
    code: string;
    dialect?: string;
  }>;
  preserveTone: boolean;
  preserveAccent: boolean;
  generateSubtitles: boolean;
}

interface TranslationJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  error?: string;
  outputs: Array<{
    language: string;
    dialect?: string;
    url: string;
    subtitlesUrl?: string;
  }>;
  config: TranslationConfig;
  createdAt: string;
  updatedAt: string;
}

// In-memory storage for jobs (replace with database in production)
const translationJobs = new Map<string, TranslationJob>();

export async function POST(
  req: NextRequest,
  { params }: { params: { videoId: string } }
) {
  try {
    const config: TranslationConfig = await req.json();
    const jobId = uuidv4();

    const job: TranslationJob = {
      id: jobId,
      status: 'pending',
      progress: 0,
      outputs: [],
      config,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    translationJobs.set(jobId, job);

    // Start processing in the background
    processTranslation(params.videoId, jobId);

    return NextResponse.json(job);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to create translation job' },
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
    const jobs = Array.from(translationJobs.values()).filter(
      (job) => job.id.startsWith(params.videoId)
    );

    return NextResponse.json(jobs);
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to list translations' },
      { status: 500 }
    );
  }
}

async function processTranslation(videoId: string, jobId: string) {
  const job = translationJobs.get(jobId);
  if (!job) return;

  try {
    // Update status to processing
    job.status = 'processing';
    job.updatedAt = new Date().toISOString();
    translationJobs.set(jobId, job);

    // Simulate processing steps
    await simulateProcessing(job);

    // Generate mock outputs
    job.outputs = job.config.targetLanguages.map((target) => ({
      language: target.code,
      dialect: target.dialect,
      url: `/api/videos/${videoId}/translations/${jobId}/${target.code}/output.mp4`,
      subtitlesUrl: job.config.generateSubtitles
        ? `/api/videos/${videoId}/translations/${jobId}/${target.code}/subtitles.vtt`
        : undefined,
    }));

    // Update job with success
    job.status = 'completed';
    job.progress = 100;
    job.updatedAt = new Date().toISOString();
    translationJobs.set(jobId, job);
  } catch (error) {
    // Update job with error
    job.status = 'failed';
    job.error = error instanceof Error ? error.message : 'Unknown error';
    job.updatedAt = new Date().toISOString();
    translationJobs.set(jobId, job);
  }
}

async function simulateProcessing(job: TranslationJob) {
  const steps = [
    'Analyzing audio content',
    'Extracting speech segments',
    'Transcribing source language',
    'Translating content',
    'Synthesizing voice',
    'Generating subtitles',
    'Rendering final video',
  ];

  for (const [index, step] of steps.entries()) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    job.progress = Math.round(((index + 1) / steps.length) * 100);
    job.updatedAt = new Date().toISOString();
    translationJobs.set(job.id, { ...job });
  }
} 