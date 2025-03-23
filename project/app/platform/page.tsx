"use client";

import { useState } from 'react';
import Link from 'next/link';
import { Header } from "@/components/ui/header";
import { FooterSection } from "@/components/FooterSection";

export default function PlatformPage() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="min-h-screen bg-gray-900">
      <Header />
      
      {/* Hero Section */}
      <section className="relative bg-gray-900 text-white py-20 overflow-hidden">
        <div className="absolute inset-0 z-0 opacity-20">
          <div className="absolute top-20 left-10 w-40 h-40 rounded-full bg-blue-600 blur-3xl"></div>
          <div className="absolute bottom-10 right-10 w-60 h-60 rounded-full bg-purple-600 blur-3xl"></div>
        </div>

        <div className="container mx-auto px-6 relative z-10">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6 tracking-tight">
              AI Video Editing <span className="text-blue-400">Platform</span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-10 leading-relaxed">
              Transform your video content with the power of AI. Edit videos, create avatars, 
              translate content, and generate short clips - all in one platform.
            </p>
          </div>
        </div>
      </section>

      {/* Dashboard */}
      <section className="py-20 bg-gray-900">
        <div className="container mx-auto px-6">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-8 text-center">Platform Dashboard</h2>
          
          <div className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700">
            {/* Dashboard Header */}
            <div className="border-b border-gray-700 p-4">
              <div className="flex items-center justify-between">
                <div className="flex space-x-4">
                  <button 
                    onClick={() => setActiveTab('dashboard')}
                    className={`px-4 py-2 rounded ${activeTab === 'dashboard' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}
                  >
                    Dashboard
                  </button>
                  <button 
                    onClick={() => setActiveTab('videos')}
                    className={`px-4 py-2 rounded ${activeTab === 'videos' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}
                  >
                    Videos
                  </button>
                  <button 
                    onClick={() => setActiveTab('avatars')}
                    className={`px-4 py-2 rounded ${activeTab === 'avatars' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}
                  >
                    Avatars
                  </button>
                  <button 
                    onClick={() => setActiveTab('clips')}
                    className={`px-4 py-2 rounded ${activeTab === 'clips' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}
                  >
                    Clip Generator
                  </button>
                </div>
                <div className="text-gray-300 text-sm">
                  <span className="bg-blue-500/20 text-blue-400 px-3 py-1 rounded-full">60 tokens</span>
                </div>
              </div>
            </div>
            
            {/* Dashboard Content */}
            <div className="p-6">
              {activeTab === 'dashboard' && (
                <div>
                  <h3 className="text-xl font-semibold text-white mb-6">Welcome to your AI Video Platform</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {/* Video Editor Card */}
                    <div className="bg-gray-700 rounded-lg p-6 border border-gray-600 hover:border-blue-500 transition-all duration-300">
                      <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-white mb-2">Video Editor</h3>
                      <p className="text-gray-300 mb-4">Edit your videos using AI to remove pauses, add subtitles, and enhance audio.</p>
                      <button className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                        Upload Video
                      </button>
                    </div>
                    
                    {/* AI Avatars Card */}
                    <div className="bg-gray-700 rounded-lg p-6 border border-gray-600 hover:border-blue-500 transition-all duration-300">
                      <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-white mb-2">AI Avatars</h3>
                      <p className="text-gray-300 mb-4">Create lifelike avatars from your videos with voice cloning capabilities.</p>
                      <button className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                        Create Avatar
                      </button>
                    </div>
                    
                    {/* Translation Card */}
                    <div className="bg-gray-700 rounded-lg p-6 border border-gray-600 hover:border-blue-500 transition-all duration-300">
                      <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-white mb-2">Translation</h3>
                      <p className="text-gray-300 mb-4">Translate your videos into 50+ languages with perfect lip synchronization.</p>
                      <button className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                        Translate Video
                      </button>
                    </div>
                    
                    {/* Clip Generator Card */}
                    <div className="bg-gray-700 rounded-lg p-6 border border-gray-600 hover:border-blue-500 transition-all duration-300">
                      <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center mb-4">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
                        </svg>
                      </div>
                      <h3 className="text-lg font-semibold text-white mb-2">Clip Generator</h3>
                      <p className="text-gray-300 mb-4">Automatically create engaging short-form vertical videos from longer content.</p>
                      <button className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                        Generate Clips
                      </button>
                    </div>
                  </div>
                  
                  {/* Recent Activity */}
                  <div className="mt-10">
                    <h3 className="text-xl font-semibold text-white mb-4">Recent Activity</h3>
                    <div className="bg-gray-700 rounded-lg border border-gray-600 overflow-hidden">
                      <div className="p-4 text-center text-gray-400">
                        No recent activity. Start by uploading a video or creating an avatar.
                      </div>
                    </div>
                  </div>
                </div>
              )}
              
              {activeTab === 'videos' && (
                <div className="text-center py-12">
                  <div className="w-20 h-20 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Upload Videos</h3>
                  <p className="text-gray-300 mb-6 max-w-md mx-auto">Drag and drop your video files here, or click to browse your files.</p>
                  <button className="py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                    Upload Video
                  </button>
                </div>
              )}
              
              {activeTab === 'avatars' && (
                <div className="text-center py-12">
                  <div className="w-20 h-20 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Create AI Avatar</h3>
                  <p className="text-gray-300 mb-6 max-w-md mx-auto">Record a video or upload samples to create your AI avatar with voice cloning.</p>
                  <button className="py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                    Create Avatar
                  </button>
                </div>
              )}
              
              {activeTab === 'clips' && (
                <div className="text-center py-12">
                  <div className="w-20 h-20 bg-blue-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">Generate Short Clips</h3>
                  <p className="text-gray-300 mb-6 max-w-md mx-auto">Automatically create engaging short-form clips from your longer videos.</p>
                  <button className="py-3 px-6 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
                    Select Videos
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <FooterSection />
    </div>
  );
}