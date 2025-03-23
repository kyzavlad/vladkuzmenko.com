'use client';

import React, { useState } from 'react';
import { FiHelpCircle } from 'react-icons/fi';

interface InfoTooltipProps {
  text: string;
}

export default function InfoTooltip({ text }: InfoTooltipProps) {
  const [isVisible, setIsVisible] = useState(false);
  
  return (
    <div className="relative ml-1.5">
      <button
        className="text-neutral-200 hover:text-neutral-100 transition-colors"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
      >
        <FiHelpCircle size={14} />
      </button>
      
      {isVisible && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 bg-neutral-500 rounded-lg shadow-elevation-3 p-2 text-xs text-neutral-100 z-50">
          {text}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 rotate-45 w-2 h-2 bg-neutral-500"></div>
        </div>
      )}
    </div>
  );
} 