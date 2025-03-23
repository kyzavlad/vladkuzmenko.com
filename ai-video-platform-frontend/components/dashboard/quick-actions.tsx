'use client';

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { 
  FiVideo, 
  FiUsers, 
  FiGlobe, 
  FiScissors, 
  FiUpload, 
  FiFolder,
  FiGrid,
  FiCamera
} from 'react-icons/fi';

interface ActionButtonProps {
  icon: React.ReactNode;
  label: string;
  description: string;
  href: string;
  color: string;
  delay?: number;
}

const ActionButton = ({ icon, label, description, href, color, delay = 0 }: ActionButtonProps) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay }}
    >
      <Link 
        href={href}
        className={`block p-4 bg-neutral-400 border-l-4 ${color} rounded-lg shadow-elevation-1 hover:shadow-elevation-2 transition-all`}
      >
        <div className="flex items-start">
          <div className={`p-3 rounded-lg bg-opacity-10 ${color.replace('border-', 'bg-')}`}>
            {icon}
          </div>
          <div className="ml-3">
            <h3 className="font-medium text-neutral-100">{label}</h3>
            <p className="text-sm text-neutral-200">{description}</p>
          </div>
        </div>
      </Link>
    </motion.div>
  );
};

export default function QuickActions() {
  const actions = [
    {
      icon: <FiVideo size={24} className="text-primary" />,
      label: 'New Video Project',
      description: 'Start editing a new video from scratch',
      href: '/platform/video-editor/new',
      color: 'border-primary',
      delay: 0
    },
    {
      icon: <FiUpload size={24} className="text-orange" />,
      label: 'Upload Media',
      description: 'Upload videos, images, or audio files',
      href: '/platform/upload',
      color: 'border-orange',
      delay: 0.1
    },
    {
      icon: <FiUsers size={24} className="text-purple" />,
      label: 'Create Avatar',
      description: 'Generate AI avatars from text or images',
      href: '/platform/avatar-creator/new',
      color: 'border-purple',
      delay: 0.2
    },
    {
      icon: <FiGlobe size={24} className="text-pink" />,
      label: 'Translate Content',
      description: 'Translate videos to multiple languages',
      href: '/platform/translation/new',
      color: 'border-pink',
      delay: 0.3
    },
    {
      icon: <FiScissors size={24} className="text-primary" />,
      label: 'Create Clip',
      description: 'Generate short clips from longer videos',
      href: '/platform/clip-generator/new',
      color: 'border-primary',
      delay: 0.4
    },
    {
      icon: <FiGrid size={24} className="text-orange" />,
      label: 'Browse Templates',
      description: 'Use pre-made templates for quick editing',
      href: '/platform/templates',
      color: 'border-orange',
      delay: 0.5
    }
  ];

  return (
    <div className="mb-8">
      <h2 className="text-xl font-bold mb-4">Quick Actions</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {actions.map((action, index) => (
          <ActionButton 
            key={index}
            icon={action.icon}
            label={action.label}
            description={action.description}
            href={action.href}
            color={action.color}
            delay={action.delay}
          />
        ))}
      </div>
    </div>
  );
} 