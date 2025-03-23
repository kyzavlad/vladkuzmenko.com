'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  FiEdit, FiCheck, FiX, FiPlay, FiPause, 
  FiVolume2, FiVolumeX, FiAlertCircle, 
  FiChevronRight, FiChevronLeft, FiRefreshCw,
  FiThumbsUp, FiThumbsDown, FiMaximize, FiMinimize
} from 'react-icons/fi';
import { useTranslationContext, TranslationSegment } from '../contexts/translation-context';

interface TranslationPreviewProps {
  jobId: string;
  languageId: string;
}

export const TranslationPreview: React.FC<TranslationPreviewProps> = ({ jobId, languageId }) => {
  const { 
    currentJob,
    fetchTranslationJob,
    updateSegment,
    approveSegment,
    markSegmentForReview,
    generatePreview,
    currentEditingSegment,
    setCurrentEditingSegment,
    isProcessing,
    error
  } = useTranslationContext();

  const [isLoading, setIsLoading] = useState(true);
  const [segments, setSegments] = useState<TranslationSegment[]>([]);
  const [activeSegment, setActiveSegment] = useState<string | null>(null);
  const [editingText, setEditingText] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [segmentPreviewUrl, setSegmentPreviewUrl] = useState<string | null>(null);
  const [isFullPreview, setIsFullPreview] = useState(true);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  const originalVideoRef = useRef<HTMLVideoElement>(null);
  const translatedVideoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const handlePlay = () => {
    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;
    
    if (!originalVideo || !translatedVideo) return;
    
    setIsPlaying(true);
    if (!originalVideo.paused) {
      translatedVideo.play();
    }
  };
  
  const handlePause = () => {
    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;
    
    if (!originalVideo || !translatedVideo) return;
    
    setIsPlaying(false);
    if (originalVideo.paused) {
      translatedVideo.pause();
    }
  };
  
  // Load job data
  useEffect(() => {
    const loadJob = async () => {
      setIsLoading(true);
      try {
        const job = await fetchTranslationJob(jobId);
        if (job.results && job.results[languageId]) {
          setSegments(job.results[languageId].segments);
          // Generate full preview by default
          const url = await generatePreview(jobId, languageId);
          setPreviewUrl(url);
          setIsFullPreview(true);
        }
      } catch (err) {
        console.error('Error loading translation job:', err);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadJob();
  }, [jobId, languageId, fetchTranslationJob, generatePreview]);
  
  // Update segments when currentJob changes
  useEffect(() => {
    if (currentJob?.results && currentJob.results[languageId]) {
      setSegments(currentJob.results[languageId].segments);
    }
  }, [currentJob, languageId]);
  
  // Handle video playback sync
  useEffect(() => {
    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;
    
    if (!originalVideo || !translatedVideo) return;
    
    const handleTimeUpdate = () => {
      if (originalVideo) {
        setProgress(originalVideo.currentTime);
      }
    };
    
    const handleLoadedMetadata = () => {
      if (originalVideo) {
        setDuration(originalVideo.duration);
      }
    };
    
    originalVideo.addEventListener('timeupdate', handleTimeUpdate);
    originalVideo.addEventListener('loadedmetadata', handleLoadedMetadata);
    originalVideo.addEventListener('play', handlePlay);
    originalVideo.addEventListener('pause', handlePause);
    
    return () => {
      originalVideo.removeEventListener('timeupdate', handleTimeUpdate);
      originalVideo.removeEventListener('loadedmetadata', handleLoadedMetadata);
      originalVideo.removeEventListener('play', handlePlay);
      originalVideo.removeEventListener('pause', handlePause);
    };
  }, []);
  
  // Handle muting
  useEffect(() => {
    if (originalVideoRef.current) {
      originalVideoRef.current.muted = isMuted;
    }
    if (translatedVideoRef.current) {
      translatedVideoRef.current.muted = !isMuted;
    }
  }, [isMuted]);
  
  const togglePlay = () => {
    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;
    
    if (!originalVideo || !translatedVideo) return;
    
    if (isPlaying) {
      originalVideo.pause();
      translatedVideo.pause();
      setIsPlaying(false);
    } else {
      originalVideo.play();
      translatedVideo.play();
      setIsPlaying(true);
    }
  };
  
  const toggleMute = () => {
    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;
    
    if (!originalVideo || !translatedVideo) return;
    
    originalVideo.muted = !isMuted;
    translatedVideo.muted = !isMuted;
    setIsMuted(!isMuted);
  };
  
  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    const originalVideo = originalVideoRef.current;
    const translatedVideo = translatedVideoRef.current;
    
    if (!originalVideo || !translatedVideo) return;
    
    originalVideo.currentTime = time;
    translatedVideo.currentTime = time;
    setProgress(time);
  };
  
  const handleSegmentClick = async (segmentId: string) => {
    setActiveSegment(segmentId);
    const segment = segments.find(seg => seg.id === segmentId);
    
    if (segment && originalVideoRef.current) {
      originalVideoRef.current.currentTime = segment.startTime;
      if (translatedVideoRef.current) {
        translatedVideoRef.current.currentTime = segment.startTime;
      }
    }
  };
  
  const startSegmentPreview = async (segmentId: string) => {
    try {
      const url = await generatePreview(jobId, languageId, segmentId);
      setSegmentPreviewUrl(url);
      setIsFullPreview(false);
      
      // Pause main videos
      if (originalVideoRef.current) originalVideoRef.current.pause();
      if (translatedVideoRef.current) translatedVideoRef.current.pause();
      setIsPlaying(false);
    } catch (err) {
      console.error('Error generating segment preview:', err);
    }
  };
  
  const startFullPreview = async () => {
    if (previewUrl) {
      setIsFullPreview(true);
      return;
    }
    
    try {
      const url = await generatePreview(jobId, languageId);
      setPreviewUrl(url);
      setIsFullPreview(true);
    } catch (err) {
      console.error('Error generating full preview:', err);
    }
  };
  
  const startEditing = (segmentId: string) => {
    const segment = segments.find(seg => seg.id === segmentId);
    if (segment) {
      setEditingText(segment.translatedText);
      setCurrentEditingSegment(segmentId);
    }
  };
  
  const cancelEditing = () => {
    setEditingText('');
    setCurrentEditingSegment(null);
  };
  
  const saveEditing = () => {
    if (currentEditingSegment && editingText.trim()) {
      updateSegment(languageId, currentEditingSegment, editingText);
      setCurrentEditingSegment(null);
    }
  };
  
  const toggleSegmentApproval = (segmentId: string, isApproved: boolean) => {
    approveSegment(languageId, segmentId, isApproved);
  };
  
  const toggleSegmentReview = (segmentId: string, needsReview: boolean) => {
    markSegmentForReview(languageId, segmentId, needsReview);
  };
  
  const formatTime = (time: number): string => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };
  
  const getLipSyncQualityColor = (score: number) => {
    if (score >= 85) return 'text-green-500';
    if (score >= 70) return 'text-yellow-500';
    return 'text-red-500';
  };
  
  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };
  
  if (isLoading) {
    return (
      <div className="w-full bg-white rounded-lg shadow-md p-6 mb-6 flex justify-center items-center h-96">
        <div className="flex flex-col items-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4"></div>
          <p className="text-gray-700">Loading translation preview...</p>
        </div>
      </div>
    );
  }

  if (!currentJob || !segments.length) {
    return (
      <div className="w-full bg-white rounded-lg shadow-md p-6 mb-6">
        <p className="text-gray-700 text-center">No translation data available for this language.</p>
      </div>
    );
  }

  return (
    <div 
      ref={containerRef}
      className="relative bg-black rounded-lg overflow-hidden"
    >
      <div className="grid grid-cols-2 gap-4 p-4">
        {/* Original Video */}
        <div className="relative">
          <video
            ref={originalVideoRef}
            src={currentJob.videoUrl}
            className="w-full rounded"
            onPlay={handlePlay}
            onPause={handlePause}
          />
          <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
            Original
          </div>
        </div>
        
        {/* Translated Video */}
        <div className="relative">
          <video
            ref={translatedVideoRef}
            src={previewUrl || ''}
            className="w-full rounded"
            onPlay={handlePlay}
            onPause={handlePause}
          />
          <div className="absolute top-2 left-2 bg-black bg-opacity-50 text-white px-2 py-1 rounded text-sm">
            Translated
          </div>
        </div>
      </div>
      
      {/* Controls */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black to-transparent p-4">
        {/* Progress bar */}
        <input
          type="range"
          min={0}
          max={duration}
          value={progress}
          onChange={handleSeek}
          className="w-full"
        />
        
        <div className="flex items-center justify-between mt-2">
          <div className="flex items-center space-x-4">
            <button
              onClick={togglePlay}
              className="text-white hover:text-gray-300 transition-colors"
            >
              {isPlaying ? <FiPause size={24} /> : <FiPlay size={24} />}
            </button>
            
            <button
              onClick={toggleMute}
              className="text-white hover:text-gray-300 transition-colors"
            >
              {isMuted ? <FiVolumeX size={24} /> : <FiVolume2 size={24} />}
            </button>
            
            <div className="text-white text-sm">
              {formatTime(progress)} / {formatTime(duration)}
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={toggleFullscreen}
              className="text-white hover:text-gray-300 transition-colors"
            >
              {isFullscreen ? <FiMinimize size={24} /> : <FiMaximize size={24} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TranslationPreview; 