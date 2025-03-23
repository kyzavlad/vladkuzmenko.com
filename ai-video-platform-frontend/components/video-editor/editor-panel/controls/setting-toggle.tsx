'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { FiCheck } from 'react-icons/fi';

interface SettingToggleProps {
  value: boolean;
  onChange: (value: boolean) => void;
  size?: 'sm' | 'md';
  disabled?: boolean;
}

export default function SettingToggle({ 
  value, 
  onChange, 
  size = 'md', 
  disabled = false 
}: SettingToggleProps) {
  const handleClick = () => {
    if (!disabled) {
      onChange(!value);
    }
  };

  const toggleSize = {
    sm: {
      track: 'w-8 h-4',
      knob: 'w-3 h-3',
      translateX: value ? 16 : 2,
    },
    md: {
      track: 'w-10 h-5',
      knob: 'w-4 h-4',
      translateX: value ? 20 : 2,
    },
  };

  return (
    <button
      type="button"
      className={`relative inline-flex items-center justify-center ${
        disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
      }`}
      onClick={handleClick}
      disabled={disabled}
      aria-pressed={value}
    >
      <span 
        className={`block ${toggleSize[size].track} rounded-full transition-colors ${
          value 
            ? 'bg-primary' 
            : 'bg-neutral-300'
        }`}
      />
      <motion.span
        className={`absolute block ${toggleSize[size].knob} rounded-full bg-white shadow-sm transition-colors`}
        initial={false}
        animate={{ 
          x: toggleSize[size].translateX,
          backgroundColor: value ? 'var(--color-primary)' : 'white'
        }}
        transition={{ 
          type: 'spring', 
          stiffness: 500, 
          damping: 30 
        }}
      >
        {value && size === 'md' && (
          <FiCheck 
            className="absolute inset-0 m-auto text-white" 
            size={10} 
          />
        )}
      </motion.span>
    </button>
  );
} 