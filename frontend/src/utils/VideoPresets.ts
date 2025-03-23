interface PlatformPreset {
  name: string;
  aspectRatio: string;
  maxDuration: number;
  recommendedDuration: number;
  maxFileSize: number; // in MB
  videoCodec: string;
  audioCodec: string;
  resolution: {
    width: number;
    height: number;
  };
  frameRate: number;
  videoBitrate: number; // in Mbps
  audioBitrate: number; // in kbps
}

export const platformPresets: Record<string, PlatformPreset> = {
  youtube_shorts: {
    name: 'YouTube Shorts',
    aspectRatio: '9:16',
    maxDuration: 60,
    recommendedDuration: 30,
    maxFileSize: 256,
    videoCodec: 'h264',
    audioCodec: 'aac',
    resolution: {
      width: 1080,
      height: 1920,
    },
    frameRate: 60,
    videoBitrate: 8,
    audioBitrate: 128,
  },
  tiktok: {
    name: 'TikTok',
    aspectRatio: '9:16',
    maxDuration: 180,
    recommendedDuration: 15,
    maxFileSize: 287,
    videoCodec: 'h264',
    audioCodec: 'aac',
    resolution: {
      width: 1080,
      height: 1920,
    },
    frameRate: 60,
    videoBitrate: 6,
    audioBitrate: 128,
  },
  instagram_reels: {
    name: 'Instagram Reels',
    aspectRatio: '9:16',
    maxDuration: 90,
    recommendedDuration: 30,
    maxFileSize: 250,
    videoCodec: 'h264',
    audioCodec: 'aac',
    resolution: {
      width: 1080,
      height: 1920,
    },
    frameRate: 30,
    videoBitrate: 5,
    audioBitrate: 128,
  },
  twitch_clips: {
    name: 'Twitch Clips',
    aspectRatio: '16:9',
    maxDuration: 60,
    recommendedDuration: 30,
    maxFileSize: 500,
    videoCodec: 'h264',
    audioCodec: 'aac',
    resolution: {
      width: 1920,
      height: 1080,
    },
    frameRate: 60,
    videoBitrate: 6,
    audioBitrate: 160,
  },
};

export const getPresetByPlatform = (platform: string): PlatformPreset | undefined => {
  return platformPresets[platform];
};

export const validateVideoForPlatform = (
  platform: string,
  duration: number,
  fileSize: number,
): { isValid: boolean; errors: string[] } => {
  const preset = platformPresets[platform];
  const errors: string[] = [];

  if (!preset) {
    return { isValid: false, errors: ['Invalid platform selected'] };
  }

  if (duration > preset.maxDuration) {
    errors.push(
      `Video duration (${duration}s) exceeds platform maximum (${preset.maxDuration}s)`
    );
  }

  if (fileSize > preset.maxFileSize * 1024 * 1024) {
    errors.push(
      `File size (${Math.round(fileSize / (1024 * 1024))}MB) exceeds platform maximum (${
        preset.maxFileSize
      }MB)`
    );
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
};

export const getOptimalEncodingSettings = (
  platform: string,
  originalWidth: number,
  originalHeight: number,
): {
  width: number;
  height: number;
  videoBitrate: number;
  audioBitrate: number;
  frameRate: number;
} => {
  const preset = platformPresets[platform];
  
  if (!preset) {
    throw new Error('Invalid platform selected');
  }

  // Calculate aspect ratio
  const targetAspectRatio = preset.resolution.width / preset.resolution.height;
  const originalAspectRatio = originalWidth / originalHeight;

  let width = preset.resolution.width;
  let height = preset.resolution.height;

  // Adjust dimensions while maintaining aspect ratio
  if (originalAspectRatio > targetAspectRatio) {
    width = Math.round(height * originalAspectRatio);
  } else {
    height = Math.round(width / originalAspectRatio);
  }

  return {
    width,
    height,
    videoBitrate: preset.videoBitrate,
    audioBitrate: preset.audioBitrate,
    frameRate: preset.frameRate,
  };
}; 