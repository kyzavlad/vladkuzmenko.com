'use client';

import React from 'react';
import { FiChevronDown } from 'react-icons/fi';

interface SettingSelectProps {
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}

export default function SettingSelect({ 
  value, 
  options, 
  onChange
}: SettingSelectProps) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-neutral-300 text-neutral-100 rounded-md py-2 pl-3 pr-8 appearance-none cursor-pointer hover:bg-neutral-350 transition-colors"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      
      <div className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-neutral-200">
        <FiChevronDown size={16} />
      </div>
    </div>
  );
} 