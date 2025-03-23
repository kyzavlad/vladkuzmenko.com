'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiPlus, FiMinus, FiScissors, FiCopy, FiTrash2 } from 'react-icons/fi';
import { useEditorContext } from '../contexts/editor-context';
import TimelineTrack, { TimelineClip } from './timeline-track';
import TimelineScrubber from './timeline-scrubber';

export default function TimelineEditor() {
  const [zoom, setZoom] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [timelineWidth, setTimelineWidth] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  
  const timelineRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  
  const { activeFile, settings } = useEditorContext();
  
  const duration = activeFile?.duration || 0;
  
  // Mock data for demonstration
  const [clips, setClips] = useState<TimelineClip[]>([
    {
      id: '1',
      type: 'video',
      start: 0,
      end: duration,
      track: 0,
      name: activeFile?.name || 'Video Clip',
      color: '#4CAF50'
    }
  ]);
  
  const [subtitles, setSubtitles] = useState<TimelineClip[]>([
    {
      id: '1',
      type: 'subtitle',
      text: 'First subtitle',
      start: 2,
      end: 5,
      track: 0,
      name: 'Subtitle 1',
      color: '#2196F3'
    },
    {
      id: '2',
      type: 'subtitle',
      text: 'Second subtitle',
      start: 7,
      end: 11,
      track: 0,
      name: 'Subtitle 2',
      color: '#2196F3'
    }
  ]);
  
  const [music, setMusic] = useState<TimelineClip[]>([
    {
      id: '1',
      type: 'music',
      name: 'Background music',
      start: 0,
      end: duration,
      track: 0,
      color: '#9C27B0'
    }
  ]);
  
  // Update timeline with when active file changes
  useEffect(() => {
    if (activeFile) {
      // Reset current time
      setCurrentTime(0);
      setPlaying(false);
      
      // Update main clip
      setClips([
        {
          id: '1',
          type: 'video',
          start: 0,
          end: activeFile.duration || 0,
          track: 0,
          name: activeFile.name || 'Video Clip',
          color: '#4CAF50'
        }
      ]);
    }
  }, [activeFile]);
  
  // Update timeline width on resize
  useEffect(() => {
    if (!timelineRef.current) return;
    
    const updateTimelineWidth = () => {
      if (timelineRef.current) {
        setTimelineWidth(timelineRef.current.offsetWidth);
      }
    };
    
    updateTimelineWidth();
    window.addEventListener('resize', updateTimelineWidth);
    
    return () => {
      window.removeEventListener('resize', updateTimelineWidth);
    };
  }, []);
  
  // Calculate pixel to time ratio
  const pixelsPerSecond = (timelineWidth / (activeFile?.duration || 60)) * zoom;
  
  // Scroll timeline to keep playhead visible
  useEffect(() => {
    if (!containerRef.current) return;
    
    const playheadPosition = currentTime * pixelsPerSecond;
    const container = containerRef.current;
    const containerWidth = container.offsetWidth;
    
    // Only scroll if the playhead is outside the visible area
    if (playheadPosition < container.scrollLeft || 
        playheadPosition > container.scrollLeft + containerWidth - 100) {
      container.scrollLeft = playheadPosition - containerWidth / 2;
    }
  }, [currentTime, pixelsPerSecond]);
  
  // Handle zoom
  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 0.25, 5));
  };
  
  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 0.25, 0.5));
  };
  
  // Handle scrubber drag
  const handleScrubberDrag = (newTime: number) => {
    if (activeFile) {
      const clampedTime = Math.max(0, Math.min(newTime, duration));
      setCurrentTime(clampedTime);
      
      // In a real app, we would update the video preview time
      // videoRef.current.currentTime = clampedTime;
    }
  };
  
  // Convert seconds to mm:ss format
  const formatTime = (timeInSeconds: number) => {
    const minutes = Math.floor(timeInSeconds / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };
  
  if (!activeFile) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-neutral-200">Select a media file to edit</p>
      </div>
    );
  }
  
  return (
    <div className="bg-neutral-400 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-neutral-300">
        <h2 className="text-lg font-bold text-neutral-100">Timeline</h2>
      </div>
      
      <div 
        ref={containerRef}
        className="p-4 space-y-4 max-h-[400px] overflow-y-auto"
      >
        {activeFile && (
          <>
            <TimelineTrack
              name="Video"
              clips={clips}
              pixelsPerSecond={100}
              onClipUpdate={(updatedClip: TimelineClip) => {
                setClips(clips.map(clip => 
                  clip.id === updatedClip.id ? updatedClip : clip
                ));
              }}
            />
            
            {settings.subtitles.enabled && (
              <TimelineTrack
                name="Subtitles"
                clips={subtitles}
                pixelsPerSecond={100}
                onClipUpdate={(updatedClip: TimelineClip) => {
                  setSubtitles(subtitles.map(clip => 
                    clip.id === updatedClip.id ? updatedClip : clip
                  ));
                }}
              />
            )}
            
            {settings.music.enabled && (
              <TimelineTrack
                name="Music"
                clips={music}
                pixelsPerSecond={100}
                onClipUpdate={(updatedClip: TimelineClip) => {
                  setMusic(music.map(clip => 
                    clip.id === updatedClip.id ? updatedClip : clip
                  ));
                }}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
} 