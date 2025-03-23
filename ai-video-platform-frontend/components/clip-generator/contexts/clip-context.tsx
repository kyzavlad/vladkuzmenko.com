'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

// Types
export interface VideoSource {
  id: string;
  title: string;
  url: string;
  thumbnail: string;
  duration: number;
  platform?: 'youtube' | 'twitch' | 'vimeo' | 'custom';
  category?: string;
  createdAt: string;
}

export interface Format {
  id: string;
  name: string;
  description: string;
  aspectRatio: string;
  maxDuration: number;
  platforms: string[];
}

export interface ClipFormat {
  id: string;
  name: string;
  aspectRatio: '9:16' | '16:9' | '1:1' | '4:5';
  targetPlatform: 'tiktok' | 'instagram' | 'youtube' | 'facebook' | 'twitter' | 'general';
  platform: string;
  icon: string;
  description: string;
  durationPresets: number[]; // In seconds
}

export interface ClipSettings {
  duration: {
    min: number;
    max: number;
    target: number;
  };
  count: number;
  features: {
    faceTracking: boolean;
    silenceRemoval: boolean;
    momentDetection: boolean;
    autoCaption: boolean;
  };
  targetPlatform: ClipFormat['targetPlatform'];
  quality: 'draft' | 'standard' | 'high';
  format: ClipFormat['id'];
}

export interface ClipEnhancement {
  music?: {
    trackId: string;
    volume: number;
    fadeIn: boolean;
    fadeOut: boolean;
  };
  captions: {
    enabled: boolean;
    style: 'minimal' | 'subtitles' | 'kinetic' | 'emphasized';
    position: 'top' | 'middle' | 'bottom';
    autoTranslate: boolean;
  };
  visual: {
    filter: string | null;
    zoom: number;
    watermark: {
      enabled: boolean;
      position: 'topLeft' | 'topRight' | 'bottomLeft' | 'bottomRight';
      image: string | null;
    };
    endCard: {
      enabled: boolean;
      template: string | null;
    };
  };
}

export interface TimelineSegment {
  id: string;
  startTime: number;
  endTime: number;
  thumbnail: string;
  interestScore: number; // 0-100
  hasFaces: boolean;
  waveformData: number[]; // Audio amplitude data
  keyframes: {
    time: number;
    thumbnail: string;
  }[];
}

export interface GeneratedClip {
  id: string;
  sourceId: string;
  title: string;
  description: string;
  duration: number;
  startTime: number;
  endTime: number;
  thumbnailUrl: string;
  previewUrl: string;
  videoUrl: string;
  format: ClipFormat;
  targetPlatform: ClipFormat['targetPlatform'];
  createdAt: string;
  engagementScore: number; // 0-100 AI prediction
  timeline: TimelineSegment;
  enhancements: ClipEnhancement;
  status: 'processing' | 'ready' | 'failed';
  tags: string[];
  exportHistory: {
    platform: string;
    date: string;
    format: string;
    url: string;
  }[];
  sourceVideoTitle: string;
}

export interface ClipGenerationJob {
  id: string;
  sourceId: string;
  settings: ClipSettings;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: number; // 0-100
  stage: 'analyzing' | 'detecting-moments' | 'generating-clips' | 'enhancing' | 'finalizing';
  estimatedTimeRemaining: number; // in seconds
  createdAt: string;
  updatedAt: string;
  clips: GeneratedClip[];
  error?: string;
}

interface ClipContextType {
  availableFormats: Format[];
  selectedFormat: Format | null;
  setSelectedFormat: (format: Format | null) => void;
  uploadedVideo: File | null;
  setUploadedVideo: (video: File | null) => void;
  generatedClips: any[];
  setGeneratedClips: (clips: any[]) => void;
  isProcessing: boolean;
  setIsProcessing: (processing: boolean) => void;
  error: string | null;
  setError: (error: string | null) => void;
}

const CLIP_FORMATS: Format[] = [
  {
    id: 'tiktok-vertical',
    name: 'TikTok Vertical',
    description: '9:16 aspect ratio, perfect for TikTok',
    aspectRatio: '9:16',
    maxDuration: 60,
    platforms: ['TikTok']
  },
  {
    id: 'instagram-reels',
    name: 'Instagram Reels',
    description: '9:16 aspect ratio, optimized for Instagram Reels',
    aspectRatio: '9:16',
    maxDuration: 90,
    platforms: ['Instagram']
  },
  {
    id: 'youtube-shorts',
    name: 'YouTube Shorts',
    description: '9:16 aspect ratio, ideal for YouTube Shorts',
    aspectRatio: '9:16',
    maxDuration: 60,
    platforms: ['YouTube']
  }
];

// Sample video sources for development
const SAMPLE_SOURCES: VideoSource[] = [
  {
    id: 'video-1',
    title: 'Product Keynote 2023',
    url: '/videos/keynote-2023.mp4',
    thumbnail: '/videos/thumbnails/keynote-2023.jpg',
    duration: 3600, // 1 hour
    platform: 'custom',
    category: 'tech',
    createdAt: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toISOString()
  },
  {
    id: 'video-2',
    title: 'Interview with AI Expert',
    url: '/videos/ai-interview.mp4',
    thumbnail: '/videos/thumbnails/ai-interview.jpg',
    duration: 1800, // 30 minutes
    platform: 'youtube',
    category: 'education',
    createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()
  }
];

// Sample jobs for development
const SAMPLE_JOBS: ClipGenerationJob[] = [
  {
    id: 'job-1',
    sourceId: 'video-1',
    settings: {
      duration: { min: 15, max: 60, target: 30 },
      count: 5,
      features: {
        faceTracking: true,
        silenceRemoval: true,
        momentDetection: true,
        autoCaption: true
      },
      targetPlatform: 'tiktok',
      quality: 'standard',
      format: 'tiktok-vertical'
    },
    status: 'completed',
    progress: 100,
    stage: 'finalizing',
    estimatedTimeRemaining: 0,
    createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
    clips: [] // Would be populated with generated clips
  }
];

// Create Context
const ClipContext = createContext<ClipContextType | undefined>(undefined);

export const useClipContext = () => {
  const context = useContext(ClipContext);
  if (context === undefined) {
    throw new Error('useClipContext must be used within a ClipProvider');
  }
  return context;
};

export const ClipProvider = ({ children }: { children: ReactNode }) => {
  const [availableFormats] = useState<Format[]>(CLIP_FORMATS);
  const [selectedFormat, setSelectedFormat] = useState<Format | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<File | null>(null);
  const [generatedClips, setGeneratedClips] = useState<any[]>([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const value = {
    availableFormats,
    selectedFormat,
    setSelectedFormat,
    uploadedVideo,
    setUploadedVideo,
    generatedClips,
    setGeneratedClips,
    isProcessing,
    setIsProcessing,
    error,
    setError
  };

  return (
    <ClipContext.Provider value={value}>
      {children}
    </ClipContext.Provider>
  );
};

export default ClipProvider; 