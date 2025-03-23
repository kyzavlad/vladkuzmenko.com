'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';

interface TimelineScrubberProps {
  currentTime: number;
  duration: number;
  pixelsPerSecond: number;
  onScrub: (time: number) => void;
}

export default function TimelineScrubber({
  currentTime,
  duration,
  pixelsPerSecond,
  onScrub
}: TimelineScrubberProps) {
  const [isDragging, setIsDragging] = useState(false);
  const scrubberRef = useRef<HTMLDivElement>(null);
  
  // Calculate scrubber position based on current time
  const scrubberPosition = currentTime * pixelsPerSecond;
  
  // Handle mouse down on scrubber
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  // Handle global mouse move when dragging
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !scrubberRef.current) return;
      
      // Get parent timeline element
      const timeline = scrubberRef.current.parentElement;
      if (!timeline) return;
      
      // Calculate timeline bounds
      const rect = timeline.getBoundingClientRect();
      const timelineLeft = rect.left;
      const timelineWidth = rect.width;
      
      // Calculate position within timeline (clamped between 0 and timeline width)
      const position = Math.max(0, Math.min(e.clientX - timelineLeft, timelineWidth));
      
      // Convert position to time
      const time = position / pixelsPerSecond;
      
      // Call onScrub with the new time (clamped between 0 and duration)
      onScrub(Math.max(0, Math.min(time, duration)));
    };
    
    const handleMouseUp = () => {
      setIsDragging(false);
    };
    
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, pixelsPerSecond, duration, onScrub]);
  
  return (
    <div
      ref={scrubberRef}
      className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10 pointer-events-none"
      style={{ left: `${scrubberPosition}px` }}
    >
      {/* Scrubber Head */}
      <motion.div
        className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-5 h-5 bg-red-500 rounded-full cursor-grab pointer-events-auto"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95, cursor: 'grabbing' }}
        animate={{ scale: isDragging ? 1.1 : 1 }}
        onMouseDown={handleMouseDown}
      />
      
      {/* Current Time Display */}
      <div className="absolute -top-7 left-1/2 transform -translate-x-1/2 bg-neutral-800 text-white text-xs py-0.5 px-1.5 rounded whitespace-nowrap">
        {formatTime(currentTime)}
      </div>
    </div>
  );
}

// Format time as MM:SS
function formatTime(timeInSeconds: number) {
  const minutes = Math.floor(timeInSeconds / 60);
  const seconds = Math.floor(timeInSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
} 