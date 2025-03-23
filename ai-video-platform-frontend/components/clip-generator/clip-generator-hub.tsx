'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { 
  FiScissors, FiUpload, FiClock, FiPlay, 
  FiChevronRight, FiChevronLeft, FiAward, 
  FiTrendingUp, FiCpu, FiStar
} from 'react-icons/fi';
import { useClipContext } from './contexts/clip-context';

export const ClipGeneratorHub: React.FC = () => {
  const router = useRouter();
  const { availableFormats } = useClipContext();
  const [activeSlide, setActiveSlide] = useState(0);
  
  // Sample showcase videos for the carousel
  const showcaseVideos = [
    {
      id: 'showcase-1',
      title: 'Product Launch Highlights',
      thumbnail: '/clip-generator/showcase/product-launch.jpg',
      videoUrl: '/clip-generator/showcase/product-launch.mp4',
      format: 'tiktok-vertical',
      duration: 45,
      views: '245K',
      engagement: 92
    },
    {
      id: 'showcase-2',
      title: 'Interview Key Moments',
      thumbnail: '/clip-generator/showcase/interview.jpg',
      videoUrl: '/clip-generator/showcase/interview.mp4',
      format: 'instagram-reels',
      duration: 30,
      views: '128K',
      engagement: 88
    },
    {
      id: 'showcase-3',
      title: 'Tutorial Clips',
      thumbnail: '/clip-generator/showcase/tutorial.jpg',
      videoUrl: '/clip-generator/showcase/tutorial.mp4',
      format: 'youtube-shorts',
      duration: 60,
      views: '312K',
      engagement: 95
    }
  ];
  
  // Token usage estimates
  const tokenUsageEstimates = [
    { duration: '15 min', tokens: '~500', clips: 5 },
    { duration: '30 min', tokens: '~1,000', clips: 8 },
    { duration: '1 hour', tokens: '~2,000', clips: 15 },
    { duration: '2 hours', tokens: '~3,800', clips: 25 }
  ];
  
  // Trending content formats
  const trendingFormats = [
    { 
      name: 'Tutorial Clips', 
      description: 'Short instructional moments from longer tutorials',
      icon: <FiStar className="text-yellow-500" />,
      platforms: ['TikTok', 'YouTube Shorts']
    },
    { 
      name: 'Interview Highlights', 
      description: 'Key quotes and insights from interviews',
      icon: <FiAward className="text-purple-500" />,
      platforms: ['Instagram Reels', 'LinkedIn']
    },
    { 
      name: 'Product Demos', 
      description: 'Feature showcases in vertical format',
      icon: <FiTrendingUp className="text-blue-500" />,
      platforms: ['TikTok', 'Instagram']
    },
    { 
      name: 'Behind the Scenes', 
      description: 'Casual moments that humanize your brand',
      icon: <FiStar className="text-green-500" />,
      platforms: ['TikTok', 'Instagram Stories']
    }
  ];

  const handleNextSlide = () => {
    setActiveSlide((prev) => (prev + 1) % showcaseVideos.length);
  };

  const handlePrevSlide = () => {
    setActiveSlide((prev) => (prev - 1 + showcaseVideos.length) % showcaseVideos.length);
  };
  
  const handleStartClipGeneration = () => {
    router.push('/clip-generator/upload');
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex flex-col md:flex-row justify-between items-start mb-8">
        <div>
          <h1 className="text-3xl font-bold mb-2">Clip Generator</h1>
          <p className="text-gray-600 max-w-2xl">
            Transform your long-form videos into engaging short-form content optimized for social media platforms.
            Generate multiple AI-powered clips with just a few clicks.
          </p>
        </div>
        <div className="mt-4 md:mt-0">
          <button 
            className="bg-blue-600 text-white py-3 px-6 rounded-lg flex items-center hover:bg-blue-700 transition-all shadow-md"
            onClick={handleStartClipGeneration}
          >
            <FiScissors className="mr-2" />
            <span>Start Generating Clips</span>
          </button>
        </div>
      </div>
      
      {/* Hero Card */}
      <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-2xl p-1 mb-12 shadow-xl">
        <div className="bg-white dark:bg-gray-900 rounded-2xl p-6 md:p-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            <div>
              <h2 className="text-2xl font-bold mb-4">Automatic Vertical Video Clips</h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Our AI analyzes your long-form videos to identify the most engaging moments and automatically 
                transforms them into short-form vertical videos optimized for social media platforms.
              </p>
              
              <div className="space-y-4">
                <div className="flex items-center">
                  <div className="bg-purple-100 dark:bg-purple-900 p-3 rounded-full mr-4">
                    <FiCpu className="text-purple-600 dark:text-purple-300" />
                  </div>
                  <div>
                    <h3 className="font-medium">AI-Powered Moment Detection</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Automatically finds the most interesting segments</p>
                  </div>
                </div>
                
                <div className="flex items-center">
                  <div className="bg-blue-100 dark:bg-blue-900 p-3 rounded-full mr-4">
                    <FiScissors className="text-blue-600 dark:text-blue-300" />
                  </div>
                  <div>
                    <h3 className="font-medium">Smart Cropping for Vertical Formats</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Optimizes framing for mobile-first viewing</p>
                  </div>
                </div>
                
                <div className="flex items-center">
                  <div className="bg-green-100 dark:bg-green-900 p-3 rounded-full mr-4">
                    <FiPlay className="text-green-600 dark:text-green-300" />
                  </div>
                  <div>
                    <h3 className="font-medium">Platform-Specific Optimization</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Tailored for TikTok, Instagram Reels, YouTube Shorts, and more</p>
                  </div>
                </div>
              </div>
              
              <button 
                className="mt-8 bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg flex items-center hover:from-blue-700 hover:to-purple-700 transition-all"
                onClick={handleStartClipGeneration}
              >
                <FiUpload className="mr-2" />
                <span>Upload Video & Generate Clips</span>
              </button>
            </div>
            
            <div className="relative aspect-[9/16] max-w-xs mx-auto lg:ml-auto">
              {/* Mobile Phone Frame */}
              <div className="absolute inset-0 bg-black rounded-[3rem] p-2 shadow-xl">
                <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-1/3 h-6 bg-black rounded-b-xl z-10"></div>
                <div className="relative h-full w-full overflow-hidden rounded-[2.5rem] bg-white">
                  {/* Demo Video */}
                  <video 
                    className="absolute inset-0 h-full w-full object-cover"
                    src="/clip-generator/demo-clip.mp4" 
                    autoPlay 
                    muted 
                    loop 
                    playsInline
                  />
                  
                  {/* Mobile UI Overlays */}
                  <div className="absolute bottom-0 inset-x-0 h-16 bg-gradient-to-t from-black to-transparent opacity-70"></div>
                  <div className="absolute bottom-4 inset-x-0 flex justify-between px-4">
                    <div className="flex flex-col items-center">
                      <div className="text-white text-2xl">❤️</div>
                      <span className="text-white text-xs">245K</span>
                    </div>
                    <div className="flex flex-col items-center">
                      <div className="text-white text-2xl">💬</div>
                      <span className="text-white text-xs">1.2K</span>
                    </div>
                    <div className="flex flex-col items-center">
                      <div className="text-white text-2xl">⤴️</div>
                      <span className="text-white text-xs">Share</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Results Carousel */}
      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-6">Sample Results</h2>
        
        <div className="relative">
          <div className="overflow-hidden rounded-xl">
            <div 
              className="flex transition-transform duration-500 ease-in-out" 
              style={{ transform: `translateX(-${activeSlide * 100}%)` }}
            >
              {showcaseVideos.map((video) => (
                <div key={video.id} className="w-full flex-shrink-0 px-1">
                  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-md overflow-hidden">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="aspect-[9/16] relative overflow-hidden">
                        <img 
                          src={video.thumbnail} 
                          alt={video.title} 
                          className="object-cover h-full w-full"
                        />
                        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-30 opacity-0 hover:opacity-100 transition-opacity">
                          <button className="p-4 bg-white bg-opacity-20 rounded-full">
                            <FiPlay className="text-white text-3xl" />
                          </button>
                        </div>
                        <div className="absolute bottom-3 right-3 bg-black bg-opacity-70 text-white py-1 px-2 rounded-md text-sm">
                          {video.duration}s
                        </div>
                      </div>
                      
                      <div className="p-6">
                        <h3 className="text-xl font-semibold mb-3">{video.title}</h3>
                        
                        <div className="grid grid-cols-2 gap-4 mb-6">
                          <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                            <div className="text-gray-500 dark:text-gray-400 text-sm">Format</div>
                            <div className="font-medium">
                              {availableFormats.find(f => f.id === video.format)?.name || video.format}
                            </div>
                          </div>
                          <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                            <div className="text-gray-500 dark:text-gray-400 text-sm">Duration</div>
                            <div className="font-medium">{video.duration} seconds</div>
                          </div>
                          <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                            <div className="text-gray-500 dark:text-gray-400 text-sm">Views</div>
                            <div className="font-medium">{video.views}</div>
                          </div>
                          <div className="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                            <div className="text-gray-500 dark:text-gray-400 text-sm">Engagement</div>
                            <div className="font-medium text-green-600">{video.engagement}%</div>
                          </div>
                        </div>
                        
                        <button 
                          className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors"
                          onClick={handleStartClipGeneration}
                        >
                          Try Now
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <button 
            className="absolute top-1/2 left-4 transform -translate-y-1/2 bg-white dark:bg-gray-800 p-2 rounded-full shadow-md"
            onClick={handlePrevSlide}
          >
            <FiChevronLeft className="text-gray-600 dark:text-gray-200" />
          </button>
          
          <button 
            className="absolute top-1/2 right-4 transform -translate-y-1/2 bg-white dark:bg-gray-800 p-2 rounded-full shadow-md"
            onClick={handleNextSlide}
          >
            <FiChevronRight className="text-gray-600 dark:text-gray-200" />
          </button>
          
          <div className="flex justify-center mt-4">
            {showcaseVideos.map((_, index) => (
              <button
                key={`dot-${index}`}
                className={`mx-1 h-2 w-2 rounded-full ${
                  activeSlide === index ? 'bg-blue-600' : 'bg-gray-300 dark:bg-gray-600'
                }`}
                onClick={() => setActiveSlide(index)}
              />
            ))}
          </div>
        </div>
      </div>
      
      {/* Token Usage and Trending Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
        {/* Token Usage Estimator */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-md p-6">
          <div className="flex items-center mb-4">
            <FiClock className="text-blue-600 dark:text-blue-400 mr-2 text-xl" />
            <h2 className="text-xl font-bold">Token Usage Estimation</h2>
          </div>
          
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            Estimated token usage for different video lengths, assuming default settings:
          </p>
          
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b dark:border-gray-700">
                  <th className="py-3 px-4 text-left">Video Length</th>
                  <th className="py-3 px-4 text-left">Est. Token Usage</th>
                  <th className="py-3 px-4 text-left">Avg. Clips Generated</th>
                </tr>
              </thead>
              <tbody>
                {tokenUsageEstimates.map((estimate, index) => (
                  <tr 
                    key={`estimate-${index}`} 
                    className="border-b dark:border-gray-700 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-700"
                  >
                    <td className="py-3 px-4">{estimate.duration}</td>
                    <td className="py-3 px-4">{estimate.tokens}</td>
                    <td className="py-3 px-4">{estimate.clips}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="mt-4 text-sm text-gray-500 dark:text-gray-400">
            Note: Actual usage may vary based on content complexity and selected features.
          </div>
        </div>
        
        {/* Trending Content Formats */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-md p-6">
          <div className="flex items-center mb-4">
            <FiTrendingUp className="text-purple-600 dark:text-purple-400 mr-2 text-xl" />
            <h2 className="text-xl font-bold">Trending Content Formats</h2>
          </div>
          
          <p className="text-gray-600 dark:text-gray-300 mb-4">
            Popular short-form content types performing well on social platforms:
          </p>
          
          <div className="space-y-4">
            {trendingFormats.map((format, index) => (
              <div 
                key={`format-${index}`}
                className="p-4 border border-gray-100 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              >
                <div className="flex items-center mb-2">
                  {format.icon}
                  <h3 className="font-medium ml-2">{format.name}</h3>
                </div>
                <p className="text-gray-600 dark:text-gray-400 text-sm mb-2">
                  {format.description}
                </p>
                <div className="flex flex-wrap">
                  {format.platforms.map(platform => (
                    <span 
                      key={`${format.name}-${platform}`}
                      className="text-xs bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-2 py-1 rounded-full mr-2 mb-1"
                    >
                      {platform}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      
      {/* Quick Stats & Benefits */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-md p-6 mb-12">
        <h2 className="text-2xl font-bold mb-6 text-center">Why Use AI Clip Generator?</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="text-center p-4">
            <div className="text-blue-600 text-4xl font-bold mb-2">10x</div>
            <h3 className="font-medium mb-1">Faster Production</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Save hours of manual editing and clip creation time
            </p>
          </div>
          
          <div className="text-center p-4">
            <div className="text-purple-600 text-4xl font-bold mb-2">5x</div>
            <h3 className="font-medium mb-1">More Content</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Generate multiple clips from a single source video
            </p>
          </div>
          
          <div className="text-center p-4">
            <div className="text-green-600 text-4xl font-bold mb-2">+42%</div>
            <h3 className="font-medium mb-1">Engagement</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Average increase in engagement with optimized vertical clips
            </p>
          </div>
          
          <div className="text-center p-4">
            <div className="text-amber-600 text-4xl font-bold mb-2">+68%</div>
            <h3 className="font-medium mb-1">Reach</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Expanded audience reach through multi-platform distribution
            </p>
          </div>
        </div>
      </div>
      
      {/* CTA */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl shadow-md p-8 text-center text-white">
        <h2 className="text-2xl font-bold mb-4">Ready to Transform Your Content?</h2>
        <p className="mb-6 max-w-2xl mx-auto">
          Upload your long-form videos now and let our AI do the work. 
          Create compelling short-form content that drives engagement across all platforms.
        </p>
        <button 
          className="bg-white text-blue-600 py-3 px-8 rounded-lg font-medium hover:bg-gray-100 transition-colors"
          onClick={handleStartClipGeneration}
        >
          Start Generating Clips
        </button>
      </div>
    </div>
  );
};

export default ClipGeneratorHub; 