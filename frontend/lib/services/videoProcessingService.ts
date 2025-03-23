import { VideoEditSettings } from '@/components/video/editor/SettingsPanel';
import { Effect } from '@/components/video/editor/EffectsPanel';

interface ProcessingProgress {
  status: 'processing' | 'completed' | 'error';
  progress: number;
  message: string;
}

interface ProcessingResult {
  url: string;
  duration: number;
  resolution: {
    width: number;
    height: number;
  };
}

class VideoProcessingService {
  private apiBaseUrl: string;
  private progressCallbacks: Map<string, (progress: ProcessingProgress) => void>;

  constructor(apiBaseUrl: string = process.env.NEXT_PUBLIC_API_URL || '') {
    this.apiBaseUrl = apiBaseUrl;
    this.progressCallbacks = new Map();
  }

  async processVideo(
    videoId: string,
    settings: VideoEditSettings,
    effects: Effect[],
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<ProcessingResult> {
    if (onProgress) {
      this.progressCallbacks.set(videoId, onProgress);
    }

    try {
      // Start processing request
      const response = await fetch(`${this.apiBaseUrl}/api/videos/${videoId}/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ settings, effects }),
      });

      if (!response.ok) {
        throw new Error('Failed to start video processing');
      }

      const { jobId } = await response.json();

      // Poll for processing status
      return await this.pollProcessingStatus(videoId, jobId);
    } catch (error) {
      this.updateProgress(videoId, {
        status: 'error',
        progress: 0,
        message: error instanceof Error ? error.message : 'Unknown error occurred',
      });
      throw error;
    } finally {
      this.progressCallbacks.delete(videoId);
    }
  }

  async applyEffect(
    videoId: string,
    effect: Effect,
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<ProcessingResult> {
    if (onProgress) {
      this.progressCallbacks.set(videoId, onProgress);
    }

    try {
      const response = await fetch(
        `${this.apiBaseUrl}/api/videos/${videoId}/effects`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(effect),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to apply effect');
      }

      const { jobId } = await response.json();

      // Poll for processing status
      return await this.pollProcessingStatus(videoId, jobId);
    } catch (error) {
      this.updateProgress(videoId, {
        status: 'error',
        progress: 0,
        message: error instanceof Error ? error.message : 'Unknown error occurred',
      });
      throw error;
    } finally {
      this.progressCallbacks.delete(videoId);
    }
  }

  async updateEffect(
    videoId: string,
    effectId: string,
    updates: Partial<Effect>,
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<ProcessingResult> {
    if (onProgress) {
      this.progressCallbacks.set(videoId, onProgress);
    }

    try {
      const response = await fetch(
        `${this.apiBaseUrl}/api/videos/${videoId}/effects/${effectId}`,
        {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(updates),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to update effect');
      }

      const { jobId } = await response.json();

      // Poll for processing status
      return await this.pollProcessingStatus(videoId, jobId);
    } catch (error) {
      this.updateProgress(videoId, {
        status: 'error',
        progress: 0,
        message: error instanceof Error ? error.message : 'Unknown error occurred',
      });
      throw error;
    } finally {
      this.progressCallbacks.delete(videoId);
    }
  }

  async removeEffect(
    videoId: string,
    effectId: string,
    onProgress?: (progress: ProcessingProgress) => void
  ): Promise<ProcessingResult> {
    if (onProgress) {
      this.progressCallbacks.set(videoId, onProgress);
    }

    try {
      const response = await fetch(
        `${this.apiBaseUrl}/api/videos/${videoId}/effects/${effectId}`,
        {
          method: 'DELETE',
        }
      );

      if (!response.ok) {
        throw new Error('Failed to remove effect');
      }

      const { jobId } = await response.json();

      // Poll for processing status
      return await this.pollProcessingStatus(videoId, jobId);
    } catch (error) {
      this.updateProgress(videoId, {
        status: 'error',
        progress: 0,
        message: error instanceof Error ? error.message : 'Unknown error occurred',
      });
      throw error;
    } finally {
      this.progressCallbacks.delete(videoId);
    }
  }

  private async pollProcessingStatus(
    videoId: string,
    jobId: string
  ): Promise<ProcessingResult> {
    const pollInterval = 1000; // 1 second
    const maxAttempts = 300; // 5 minutes
    let attempts = 0;

    while (attempts < maxAttempts) {
      try {
        const response = await fetch(
          `${this.apiBaseUrl}/api/jobs/${jobId}/status`
        );

        if (!response.ok) {
          throw new Error('Failed to fetch processing status');
        }

        const data = await response.json();

        this.updateProgress(videoId, {
          status: data.status,
          progress: data.progress,
          message: data.message,
        });

        if (data.status === 'completed') {
          return {
            url: data.result.url,
            duration: data.result.duration,
            resolution: data.result.resolution,
          };
        }

        if (data.status === 'error') {
          throw new Error(data.message || 'Processing failed');
        }

        await new Promise((resolve) => setTimeout(resolve, pollInterval));
        attempts++;
      } catch (error) {
        this.updateProgress(videoId, {
          status: 'error',
          progress: 0,
          message: error instanceof Error ? error.message : 'Unknown error occurred',
        });
        throw error;
      }
    }

    throw new Error('Processing timed out');
  }

  private updateProgress(videoId: string, progress: ProcessingProgress) {
    const callback = this.progressCallbacks.get(videoId);
    if (callback) {
      callback(progress);
    }
  }
}

export const videoProcessingService = new VideoProcessingService(); 