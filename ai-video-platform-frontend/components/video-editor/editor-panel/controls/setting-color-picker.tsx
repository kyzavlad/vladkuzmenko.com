'use client';

import React, { useState, useRef, useEffect } from 'react';
import { HexColorPicker } from 'react-colorful';
import { FiChevronDown } from 'react-icons/fi';

interface SettingColorPickerProps {
  value: string;
  onChange: (value: string) => void;
}

export default function SettingColorPicker({
  value,
  onChange
}: SettingColorPickerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const popover = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popover.current && !popover.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  
  return (
    <div className="relative">
      <button
        className="w-full bg-neutral-300 rounded-md py-2 px-3 flex items-center justify-between hover:bg-neutral-350 transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center">
          <div 
            className="w-4 h-4 rounded-sm mr-2" 
            style={{ backgroundColor: value }}
          />
          <span className="text-neutral-100">{value}</span>
        </div>
        <FiChevronDown 
          size={16} 
          className={`text-neutral-200 transition-transform ${isOpen ? 'rotate-180' : ''}`} 
        />
      </button>
      
      {isOpen && (
        <div 
          ref={popover}
          className="absolute top-full left-0 mt-2 z-50 bg-neutral-500 p-3 rounded-lg shadow-elevation-3"
        >
          <HexColorPicker color={value} onChange={onChange} />
        </div>
      )}
    </div>
  );
} 