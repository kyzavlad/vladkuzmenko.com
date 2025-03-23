'use client';

import React from 'react';
import { useEditorContext } from '../contexts/editor-context';

interface SplitPreviewProps {
  children: React.ReactNode;
}

export default function SplitPreview({ children }: SplitPreviewProps) {
  const { beforeAfterMode } = useEditorContext();
  
  if (!beforeAfterMode) return <>{children}</>;
  
  return (
    <div className="relative w-full h-full">
      {/* Original video (left side) */}
      <div className="absolute inset-0 overflow-hidden" style={{ clipPath: 'inset(0 50% 0 0)' }}>
        {children}
        <div className="absolute top-4 left-4 bg-black bg-opacity-50 text-white px-3 py-1 rounded-md text-sm">
          Before
        </div>
      </div>
      
      {/* Processed video (right side) */}
      <div className="absolute inset-0 overflow-hidden" style={{ clipPath: 'inset(0 0 0 50%)' }}>
        {children}
        <div className="absolute top-4 right-4 bg-black bg-opacity-50 text-white px-3 py-1 rounded-md text-sm">
          After
        </div>
      </div>
      
      {/* Divider */}
      <div className="absolute top-0 bottom-0 left-1/2 w-0.5 bg-white cursor-col-resize">
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-lg">
          <div className="w-4 h-0.5 bg-black rounded-full"></div>
        </div>
      </div>
    </div>
  );
} 