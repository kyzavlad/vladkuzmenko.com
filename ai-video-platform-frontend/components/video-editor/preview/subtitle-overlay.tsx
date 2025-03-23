'use client';

import React from 'react';
import { useEditorContext } from '../contexts/editor-context';

interface SubtitleOverlayProps {
  currentTime: number;
}

export default function SubtitleOverlay({ currentTime }: SubtitleOverlayProps) {
  const { settings } = useEditorContext();
  
  if (!settings.subtitles.enabled) return null;
  
  // In a real app, this would get the current subtitle based on the time
  const subtitle = "Sample subtitle text";
  
  const style: React.CSSProperties = {
    fontFamily: settings.subtitles.style.font,
    fontSize: settings.subtitles.style.size,
    color: settings.subtitles.style.color,
    textAlign: 'center',
    padding: '0.5rem',
    backgroundColor: settings.subtitles.style.backgroundColor,
    position: 'absolute',
    left: '50%',
    transform: 'translateX(-50%)',
    maxWidth: '80%',
    borderRadius: '0.25rem',
    ...(settings.subtitles.style.outline && {
      textShadow: `2px 2px 2px ${settings.subtitles.style.outlineColor}`
    }),
    ...(settings.subtitles.style.position === 'top' && { top: '2rem' }),
    ...(settings.subtitles.style.position === 'bottom' && { bottom: '2rem' }),
    ...(settings.subtitles.style.position === 'middle' && { top: '50%', transform: 'translate(-50%, -50%)' })
  };
  
  return (
    <div style={style}>
      {subtitle}
    </div>
  );
} 