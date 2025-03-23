'use client';

import React from 'react';
import Link from 'next/link';
import { 
  FiUser, FiGlobe, FiMic, FiScissors,
  FiArrowRight, FiPlay, FiSmartphone
} from 'react-icons/fi';

export const FeatureHighlights: React.FC = () => {
  const features = [
    {
      id: 'avatar',
      title: 'Avatar Creation',
      description: 'Create realistic digital avatars that look and sound just like you.',
      icon: <FiUser className="w-6 h-6 text-blue-600" />,
      bgClass: 'bg-blue-50',
      borderClass: 'border-blue-200',
      iconBgClass: 'bg-blue-100',
      path: '/avatar/create',
      ctaText: 'Create Avatar',
      stats: {
        label: 'Avatars Created',
        value: '450K+'
      }
    },
    {
      id: 'clip-generator',
      title: 'Clip Generator',
      description: 'Transform long-form videos into engaging short-form vertical clips.',
      icon: <FiScissors className="w-6 h-6 text-purple-600" />,
      bgClass: 'bg-purple-50',
      borderClass: 'border-purple-200',
      iconBgClass: 'bg-purple-100',
      path: '/clip-generator',
      ctaText: 'Generate Clips',
      tag: 'Beta',
      tagColor: 'bg-purple-500',
      stats: {
        label: 'Average Engagement',
        value: '+42%'
      },
      secondaryCTA: {
        text: 'Learn More',
        path: '/clip-generator'
      }
    },
    {
      id: 'translation',
      title: 'Video Translation',
      description: 'Translate your videos into multiple languages with lip-sync technology.',
      icon: <FiGlobe className="w-6 h-6 text-green-600" />,
      bgClass: 'bg-green-50',
      borderClass: 'border-green-200',
      iconBgClass: 'bg-green-100',
      path: '/translation/new',
      ctaText: 'Translate Video',
      stats: {
        label: 'Languages Supported',
        value: '75+'
      }
    },
    {
      id: 'voice',
      title: 'Voice Cloning',
      description: 'Clone your voice for voiceovers, podcasts, and audio content.',
      icon: <FiMic className="w-6 h-6 text-red-600" />,
      bgClass: 'bg-red-50',
      borderClass: 'border-red-200',
      iconBgClass: 'bg-red-100',
      path: '/voice-cloning',
      ctaText: 'Clone Voice',
      stats: {
        label: 'Avg. Training Time',
        value: '10 min'
      }
    }
  ];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-gray-800 dark:text-white">Feature Highlights</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {features.map((feature) => (
          <div 
            key={feature.id}
            className={`rounded-xl border ${feature.borderClass} ${feature.bgClass} overflow-hidden transition-shadow hover:shadow-md`}
          >
            {feature.id === 'clip-generator' ? (
              <div className="relative">
                <div className="absolute top-0 right-0 p-4 z-10">
                  {feature.tag && (
                    <span className={`px-2 py-1 text-xs font-medium text-white rounded-full ${feature.tagColor}`}>
                      {feature.tag}
                    </span>
                  )}
                </div>
                <div className="h-40 bg-purple-900 relative overflow-hidden">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="relative h-full w-1/3 mx-auto mt-4">
                      {/* Mobile Frame */}
                      <div className="absolute inset-0 bg-black rounded-2xl p-1">
                        <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-1/3 h-5 bg-black rounded-b-xl z-10"></div>
                        <div className="h-full w-full rounded-xl overflow-hidden bg-purple-800">
                          <img 
                            src="/clip-generator/demo-thumbnail.jpg" 
                            alt="Clip preview" 
                            className="h-full w-full object-cover"
                          />
                        </div>
                      </div>
                    </div>
                    <FiSmartphone className="absolute top-4 left-4 text-white opacity-20 text-6xl" />
                    <FiPlay className="absolute bottom-4 right-4 text-white opacity-20 text-6xl" />
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 flex justify-between items-start">
                <div className={`p-3 rounded-full ${feature.iconBgClass}`}>
                  {feature.icon}
                </div>
                {feature.tag && (
                  <span className={`px-2 py-1 text-xs font-medium text-white rounded-full ${feature.tagColor}`}>
                    {feature.tag}
                  </span>
                )}
              </div>
            )}
            
            <div className="p-6">
              <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                {feature.description}
              </p>
              
              <div className="flex justify-between items-center mb-4">
                <div>
                  <div className="text-sm text-gray-500 dark:text-gray-400">{feature.stats.label}</div>
                  <div className="text-lg font-bold">{feature.stats.value}</div>
                </div>
                
                <Link 
                  href={feature.path}
                  className={`inline-flex items-center text-sm font-medium ${
                    feature.id === 'clip-generator' ? 'text-purple-600 hover:text-purple-700' : 'text-blue-600 hover:text-blue-700'
                  }`}
                >
                  {feature.secondaryCTA?.text || 'Learn More'}
                  <FiArrowRight className="ml-1" />
                </Link>
              </div>
              
              <Link 
                href={feature.path}
                className={`block text-center py-2 px-4 rounded-lg transition-colors ${
                  feature.id === 'clip-generator' 
                    ? 'bg-purple-600 hover:bg-purple-700 text-white' 
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {feature.ctaText}
              </Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FeatureHighlights; 