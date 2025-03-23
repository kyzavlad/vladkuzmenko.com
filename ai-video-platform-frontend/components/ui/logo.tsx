'use client';

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { FiFilm } from 'react-icons/fi';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg';
  withText?: boolean;
  className?: string;
}

export default function Logo({ size = 'md', withText = true, className = '' }: LogoProps) {
  const sizes = {
    sm: 'h-6 w-6',
    md: 'h-8 w-8',
    lg: 'h-10 w-10',
  };

  const textSizes = {
    sm: 'text-lg',
    md: 'text-xl',
    lg: 'text-2xl',
  };

  return (
    <Link href="/platform" className={`flex items-center space-x-2 ${className}`}>
      <motion.div
        className={`${sizes[size]} bg-primary rounded-lg flex items-center justify-center text-white`}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <FiFilm className={sizes[size]} />
      </motion.div>
      {withText && (
        <span className={`font-heading font-bold ${textSizes[size]} text-primary`}>
          AI Video
        </span>
      )}
    </Link>
  );
} 