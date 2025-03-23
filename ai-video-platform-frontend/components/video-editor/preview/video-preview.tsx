'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  FiPlay, 
  FiPause, 
  FiVolume2, 
  FiVolumeX, 
  FiMaximize, 
  FiMinimize,
  FiSkipForward,
  FiSkipBack,
  FiSettings
} from 'react-icons/fi';
import { useEditorContext } from '../contexts/editor-context';
import SubtitleOverlay from './subtitle-overlay';
import SplitPreview from './split-preview';

export default function VideoPreview() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [volume, setVolume] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isVolumeSliderOpen, setIsVolumeSliderOpen] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [isMouseMoving, setIsMouseMoving] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const controlsTimeout = useRef<NodeJS.Timeout>();
  
  const { 
    activeFile, 
    settings, 
    beforeAfterMode
  } = useEditorContext();
  
  // Listen for video events
  useEffect(() => {
    const videoElement = videoRef.current;
    if (!videoElement) return;
    
    const handleLoadedMetadata = () => {
      setDuration(videoElement.duration);
    };
    
    const handleTimeUpdate = () => {
      setCurrentTime(videoElement.currentTime);
    };
    
    const handlePlay = () => {
      setIsPlaying(true);
    };
    
    const handlePause = () => {
      setIsPlaying(false);
    };
    
    const handleVolumeChange = () => {
      setVolume(videoElement.volume);
      setIsMuted(videoElement.muted);
    };
    
    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
      videoElement.currentTime = 0;
    };
    
    // Add event listeners
    videoElement.addEventListener('loadedmetadata', handleLoadedMetadata);
    videoElement.addEventListener('timeupdate', handleTimeUpdate);
    videoElement.addEventListener('play', handlePlay);
    videoElement.addEventListener('pause', handlePause);
    videoElement.addEventListener('volumechange', handleVolumeChange);
    videoElement.addEventListener('ended', handleEnded);
    
    return () => {
      // Remove event listeners
      videoElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
      videoElement.removeEventListener('timeupdate', handleTimeUpdate);
      videoElement.removeEventListener('play', handlePlay);
      videoElement.removeEventListener('pause', handlePause);
      videoElement.removeEventListener('volumechange', handleVolumeChange);
      videoElement.removeEventListener('ended', handleEnded);
    };
  }, []);
  
  // Handle auto-hide controls
  useEffect(() => {
    const handleMouseMove = () => {
      setIsMouseMoving(true);
      setShowControls(true);
      
      // Clear the existing timeout
      if (controlsTimeout.current) {
        clearTimeout(controlsTimeout.current);
      }
      
      // Set a new timeout
      controlsTimeout.current = setTimeout(() => {
        if (isPlaying) {
          setShowControls(false);
          setIsMouseMoving(false);
        }
      }, 3000);
    };
    
    const container = videoContainerRef.current;
    if (container) {
      container.addEventListener('mousemove', handleMouseMove);
      
      return () => {
        container.removeEventListener('mousemove', handleMouseMove);
        if (controlsTimeout.current) {
          clearTimeout(controlsTimeout.current);
        }
      };
    }
  }, [isPlaying]);
  
  // Toggle play/pause
  const togglePlay = () => {
    if (!videoRef.current) return;
    
    if (isPlaying) {
      videoRef.current.pause();
    } else {
      videoRef.current.play();
    }
  };
  
  // Toggle mute
  const toggleMute = () => {
    if (!videoRef.current) return;
    
    videoRef.current.muted = !isMuted;
  };
  
  // Set volume
  const handleVolumeChange = (newVolume: number) => {
    if (!videoRef.current) return;
    
    videoRef.current.volume = newVolume;
    
    if (newVolume === 0) {
      videoRef.current.muted = true;
    } else if (isMuted) {
      videoRef.current.muted = false;
    }
  };
  
  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (!videoContainerRef.current) return;
    
    if (!isFullscreen) {
      if (videoContainerRef.current.requestFullscreen) {
        videoContainerRef.current.requestFullscreen();
      } else if ((videoContainerRef.current as any).webkitRequestFullscreen) {
        (videoContainerRef.current as any).webkitRequestFullscreen();
      } else if ((videoContainerRef.current as any).mozRequestFullScreen) {
        (videoContainerRef.current as any).mozRequestFullScreen();
      } else if ((videoContainerRef.current as any).msRequestFullscreen) {
        (videoContainerRef.current as any).msRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if ((document as any).webkitExitFullscreen) {
        (document as any).webkitExitFullscreen();
      } else if ((document as any).mozCancelFullScreen) {
        (document as any).mozCancelFullScreen();
      } else if ((document as any).msExitFullscreen) {
        (document as any).msExitFullscreen();
      }
    }
  };
  
  // Update fullscreen state on change
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);
    
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, []);
  
  // Handle progress bar interactions
  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!progressRef.current || !videoRef.current) return;
    
    const rect = progressRef.current.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    const newTime = pos * duration;
    
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };
  
  // Skip forward/backward
  const skipTime = (seconds: number) => {
    if (!videoRef.current) return;
    
    const newTime = Math.min(Math.max(0, currentTime + seconds), duration);
    videoRef.current.currentTime = newTime;
    setCurrentTime(newTime);
  };
  
  // Format time (seconds to MM:SS)
  const formatTime = (timeInSeconds: number) => {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };
  
  // Progress percentage
  const progressPercentage = duration ? (currentTime / duration) * 100 : 0;
  
  if (!activeFile) {
    return (
      <div className="relative aspect-video bg-neutral-900 rounded-lg flex items-center justify-center">
        <div className="text-center text-neutral-400">
          <p className="mb-2">No media selected</p>
          <p className="text-sm">Select a file from your library to preview and edit</p>
        </div>
      </div>
    );
  }
  
  return (
    <div 
      ref={videoContainerRef} 
      className="relative aspect-video bg-black rounded-lg overflow-hidden"
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => isPlaying && setShowControls(false)}
    >
      {activeFile ? (
        <>
          <SplitPreview>
            <video
              ref={videoRef}
              src={activeFile.url}
              className="w-full h-full"
            />
          </SplitPreview>
          
          <SubtitleOverlay currentTime={currentTime} />
          
          {/* Video Controls */}
          <motion.div 
            className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black to-transparent"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: showControls ? 1 : 0, y: showControls ? 0 : 20 }}
            transition={{ duration: 0.2 }}
          >
            {/* Progress Bar */}
            <div 
              ref={progressRef}
              className="w-full h-1 bg-neutral-600 rounded-full mb-4 cursor-pointer"
              onClick={handleProgressClick}
            >
              <div 
                className="h-full bg-primary rounded-full relative"
                style={{ width: `${progressPercentage}%` }}
              >
                <div className="absolute right-0 top-1/2 transform translate-x-1/2 -translate-y-1/2 w-3 h-3 bg-primary rounded-full"></div>
              </div>
            </div>
            
            {/* Controls */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                {/* Play/Pause */}
                <button 
                  className="text-white hover:text-primary transition-colors"
                  onClick={togglePlay}
                >
                  {isPlaying ? <FiPause size={24} /> : <FiPlay size={24} />}
                </button>
                
                {/* Skip Backward/Forward */}
                <button 
                  className="text-white hover:text-primary transition-colors"
                  onClick={() => skipTime(-10)}
                >
                  <FiSkipBack size={20} />
                </button>
                <button 
                  className="text-white hover:text-primary transition-colors"
                  onClick={() => skipTime(10)}
                >
                  <FiSkipForward size={20} />
                </button>
                
                {/* Volume */}
                <div className="relative">
                  <button 
                    className="text-white hover:text-primary transition-colors"
                    onClick={toggleMute}
                    onMouseEnter={() => setIsVolumeSliderOpen(true)}
                  >
                    {isMuted || volume === 0 ? <FiVolumeX size={24} /> : <FiVolume2 size={24} />}
                  </button>
                  
                  {/* Volume Slider */}
                  {isVolumeSliderOpen && (
                    <div 
                      className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 p-2 bg-neutral-800 rounded-lg"
                      onMouseLeave={() => setIsVolumeSliderOpen(false)}
                    >
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={volume}
                        onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
                        className="w-24 accent-primary"
                      />
                    </div>
                  )}
                </div>
                
                {/* Time */}
                <div className="text-white text-sm">
                  <span>{formatTime(currentTime)}</span>
                  <span className="mx-1">/</span>
                  <span>{formatTime(duration)}</span>
                </div>
              </div>
              
              <div className="flex items-center space-x-4">
                {/* Settings */}
                <button 
                  className="text-white hover:text-primary transition-colors"
                  onClick={() => setIsSettingsOpen(!isSettingsOpen)}
                >
                  <FiSettings size={20} />
                </button>
                
                {/* Fullscreen */}
                <button 
                  className="text-white hover:text-primary transition-colors"
                  onClick={toggleFullscreen}
                >
                  {isFullscreen ? <FiMinimize size={20} /> : <FiMaximize size={20} />}
                </button>
              </div>
            </div>
          </motion.div>
        </>
      ) : (
        <div className="flex items-center justify-center h-full text-neutral-200">
          <p>No media file selected</p>
        </div>
      )}
    </div>
  );
} 