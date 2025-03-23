'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { FiArrowRight, FiX } from 'react-icons/fi';
import { useSelector } from 'react-redux';
import { RootState } from '../../store';

interface WelcomeBannerProps {
  onDismiss?: () => void;
}

export default function WelcomeBanner({ onDismiss }: WelcomeBannerProps) {
  const user = useSelector((state: RootState) => state.auth.user);
  const firstName = user?.name?.split(' ')[0] || 'there';
  
  // Get time of day for greeting
  const getTimeBasedGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };
  
  // Personalized recommendations based on user activity
  // In a real app, these would come from the backend
  const recommendations = [
    {
      id: 1,
      title: 'Complete Your Project',
      description: 'You have an unfinished video project. Continue editing?',
      cta: 'Continue Editing',
      href: '/platform/video-editor/project/1',
      icon: '🎬'
    },
    {
      id: 2,
      title: 'Try Avatar Generation',
      description: 'Create a digital avatar to use in your videos',
      cta: 'Create Avatar',
      href: '/platform/avatar-creator/new',
      icon: '👤'
    },
    {
      id: 3,
      title: 'Explore Templates',
      description: 'See trending templates for promotional videos',
      cta: 'Browse Templates',
      href: '/platform/templates',
      icon: '📋'
    }
  ];
  
  // Show only the most relevant recommendation
  const topRecommendation = recommendations[0];

  return (
    <motion.div 
      className="relative bg-gradient-to-r from-primary-700 to-purple-700 text-white rounded-xl p-6 shadow-elevation-2 overflow-hidden"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Background pattern */}
      <div className="absolute top-0 right-0 opacity-10">
        <svg width="200" height="200" viewBox="0 0 100 100">
          <circle cx="75" cy="25" r="20" fill="currentColor" />
          <circle cx="25" cy="50" r="10" fill="currentColor" />
          <circle cx="85" cy="85" r="15" fill="currentColor" />
        </svg>
      </div>
      
      {/* Close button */}
      {onDismiss && (
        <button
          className="absolute top-2 right-2 p-1 rounded-full hover:bg-white hover:bg-opacity-20 transition-colors"
          onClick={onDismiss}
        >
          <FiX size={20} />
        </button>
      )}
      
      <div className="relative z-10">
        {/* Greeting */}
        <h1 className="text-2xl font-bold">
          {getTimeBasedGreeting()}, {firstName}
        </h1>
        
        <div className="mt-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          {/* Recommendation */}
          <div className="max-w-md">
            <div className="flex items-center">
              <span className="text-2xl mr-2">{topRecommendation.icon}</span>
              <h3 className="text-lg font-medium">{topRecommendation.title}</h3>
            </div>
            <p className="mt-1 text-white text-opacity-80">{topRecommendation.description}</p>
          </div>
          
          {/* Call to action */}
          <div>
            <Link 
              href={topRecommendation.href}
              className="inline-flex items-center px-4 py-2 bg-white text-primary-700 rounded-lg font-medium hover:bg-opacity-90 transition-colors"
            >
              {topRecommendation.cta}
              <FiArrowRight className="ml-2" />
            </Link>
          </div>
        </div>
      </div>
    </motion.div>
  );
} 