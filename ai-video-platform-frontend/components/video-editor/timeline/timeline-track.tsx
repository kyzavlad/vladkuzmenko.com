'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { FiFileText, FiMusic, FiVideo } from 'react-icons/fi';

export interface TimelineClip {
  id: string;
  start: number;
  end: number;
  type: string;
  track: number;
  name: string;
  thumbnail?: string;
  color: string;
  text?: string;
}

interface TimelineMarker {
  id: string;
  time: number;
  text: string;
}

interface TimelineTrackProps {
  name: string;
  clips?: TimelineClip[];
  markers?: TimelineMarker[];
  isSubtitleTrack?: boolean;
  pixelsPerSecond: number;
  onClipUpdate?: (clip: TimelineClip) => void;
}

export default function TimelineTrack({
  name,
  clips = [],
  markers = [],
  isSubtitleTrack = false,
  pixelsPerSecond,
  onClipUpdate
}: TimelineTrackProps) {
  // Get icon for track
  const getTrackIcon = () => {
    switch (name.toLowerCase()) {
      case 'video':
        return <FiVideo size={14} />;
      case 'subtitles':
        return <FiFileText size={14} />;
      case 'music':
      case 'audio':
        return <FiMusic size={14} />;
      default:
        return <FiVideo size={14} />;
    }
  };
  
  return (
    <div className="mb-3 flex">
      {/* Track Label */}
      <div className="w-24 flex-shrink-0 pr-3 flex items-center">
        <div className="flex items-center space-x-1.5 text-neutral-100">
          {getTrackIcon()}
          <span className="text-xs font-medium truncate">{name}</span>
        </div>
      </div>
      
      {/* Track Content */}
      <div className="flex-1 relative">
        {/* Track Background */}
        <div className="h-10 bg-neutral-400 rounded-md overflow-hidden relative">
          {/* Clips */}
          {clips.map((clip) => (
            <motion.div 
              key={clip.id}
              className="absolute top-0 h-full rounded-md cursor-move"
              style={{ 
                left: `${clip.start * pixelsPerSecond}px`,
                width: `${(clip.end - clip.start) * pixelsPerSecond}px`,
                backgroundColor: clip.color,
              }}
              whileHover={{ y: -1, boxShadow: '0 2px 5px rgba(0,0,0,0.2)' }}
              drag="x"
              dragConstraints={{ left: 0, right: 0 }}
              dragElastic={0.1}
              title={clip.name}
            >
              <div className="h-full flex items-center px-2 overflow-hidden">
                <span className="text-xs text-white truncate">{clip.name}</span>
              </div>
              
              {/* Resize handles */}
              <div className="absolute left-0 top-0 bottom-0 w-1.5 cursor-w-resize" />
              <div className="absolute right-0 top-0 bottom-0 w-1.5 cursor-e-resize" />
            </motion.div>
          ))}
          
          {/* Subtitle Markers */}
          {isSubtitleTrack && markers.map((marker) => (
            <div 
              key={marker.id}
              className="absolute top-0 h-full"
              style={{ 
                left: `${marker.time * pixelsPerSecond}px`,
                width: "2px",
              }}
              title={marker.text}
            >
              <div className="h-full w-0.5 bg-white opacity-70" />
              <div 
                className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-1 bg-neutral-800 text-white text-xs py-0.5 px-1.5 rounded whitespace-nowrap"
                style={{ maxWidth: "150px" }}
              >
                <div className="truncate">{marker.text}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
} 