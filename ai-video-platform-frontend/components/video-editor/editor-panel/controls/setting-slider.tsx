'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';

interface SettingSliderProps {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  showValue?: boolean;
  marks?: Record<number, string>;
}

export default function SettingSlider({
  value,
  min,
  max,
  step = 1,
  onChange,
  disabled = false,
  showValue = false,
  marks = {}
}: SettingSliderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const sliderRef = useRef<HTMLDivElement>(null);
  
  // Normalize value to percentage for styling
  const percentage = ((value - min) / (max - min)) * 100;
  
  const handleSliderClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (disabled) return;
    
    if (sliderRef.current) {
      const rect = sliderRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = Math.min(Math.max(0, x / rect.width), 1);
      
      // Convert percentage to value and snap to step
      let newValue = min + percentage * (max - min);
      newValue = Math.round(newValue / step) * step;
      
      // Ensure value is within bounds
      newValue = Math.min(Math.max(min, newValue), max);
      
      onChange(newValue);
    }
  };
  
  const handleDragStart = () => {
    if (!disabled) {
      setIsDragging(true);
    }
  };
  
  const handleDragEnd = () => {
    setIsDragging(false);
  };
  
  const handleDrag = (e: MouseEvent) => {
    if (isDragging && sliderRef.current) {
      const rect = sliderRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = Math.min(Math.max(0, x / rect.width), 1);
      
      // Convert percentage to value and snap to step
      let newValue = min + percentage * (max - min);
      newValue = Math.round(newValue / step) * step;
      
      // Ensure value is within bounds
      newValue = Math.min(Math.max(min, newValue), max);
      
      onChange(newValue);
    }
  };
  
  // Handle drag events on document
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleDrag);
      document.addEventListener('mouseup', handleDragEnd);
      
      return () => {
        document.removeEventListener('mousemove', handleDrag);
        document.removeEventListener('mouseup', handleDragEnd);
      };
    }
  }, [isDragging]);
  
  // Value indicator position
  const thumbPosition = `${percentage}%`;
  
  return (
    <div className={`relative ${disabled ? 'opacity-50' : ''}`}>
      {/* Slider Track */}
      <div
        ref={sliderRef}
        className="h-1 bg-neutral-300 rounded-full cursor-pointer"
        onClick={handleSliderClick}
      >
        {/* Active Track */}
        <div 
          className="absolute h-1 bg-primary rounded-full"
          style={{ width: thumbPosition }}
        />
      </div>
      
      {/* Thumb */}
      <div
        className="absolute top-0 w-0 h-0"
        style={{ left: thumbPosition }}
      >
        <motion.div
          className={`w-3.5 h-3.5 bg-primary rounded-full shadow-elevation-1 -translate-x-1/2 -translate-y-1/3 cursor-grab ${
            isDragging ? 'cursor-grabbing scale-110' : ''
          }`}
          whileTap={{ scale: 1.1 }}
          onMouseDown={handleDragStart}
          animate={{ scale: isDragging ? 1.1 : 1 }}
          transition={{ duration: 0.2 }}
        />
      </div>
      
      {/* Marks */}
      {Object.keys(marks).length > 0 && (
        <div className="flex justify-between mt-2">
          {Object.entries(marks).map(([markValue, label]) => {
            const markPercentage = ((Number(markValue) - min) / (max - min)) * 100;
            return (
              <div 
                key={markValue} 
                className="relative"
                style={{ left: `calc(${markPercentage}% - 12px)` }}
              >
                <div 
                  className={`w-1 h-1 rounded-full mx-auto mb-1 ${
                    Number(markValue) <= value ? 'bg-primary' : 'bg-neutral-300'
                  }`} 
                />
                <span className="text-xs text-neutral-200 whitespace-nowrap">
                  {label}
                </span>
              </div>
            );
          })}
        </div>
      )}
      
      {/* Value display */}
      {showValue && (
        <div className="text-xs text-neutral-200 text-right mt-1">
          {value}
        </div>
      )}
    </div>
  );
} 