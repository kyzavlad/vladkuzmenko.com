'use client';

import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  FiVideo, 
  FiMic, 
  FiTrash2, 
  FiPlay, 
  FiPause, 
  FiCheckCircle, 
  FiXCircle, 
  FiAlertTriangle,
  FiInfo
} from 'react-icons/fi';
import { useCreationContext } from '../../contexts/creation-context';

export default function SampleLibrary() {
  const { 
    settings, 
    deleteSampleVideo,
    deleteSampleAudio,
    markSampleAsPreferred 
  } = useCreationContext();
  
  const [activeVideo, setActiveVideo] = useState<string | null>(null);
  const [activeAudio, setActiveAudio] = useState<string | null>(null);
  const [playingVideo, setPlayingVideo] = useState<string | null>(null);
  const [playingAudio, setPlayingAudio] = useState<string | null>(null);
  
  const handlePlayVideo = useCallback((id: string) => {
    if (playingVideo === id) {
      setPlayingVideo(null);
      const videoElement = document.getElementById(`video-${id}`) as HTMLVideoElement;
      if (videoElement) {
        videoElement.pause();
      }
    } else {
      // Pause any playing video
      if (playingVideo) {
        const previousVideo = document.getElementById(`video-${playingVideo}`) as HTMLVideoElement;
        if (previousVideo) {
          previousVideo.pause();
        }
      }
      
      setPlayingVideo(id);
      const videoElement = document.getElementById(`video-${id}`) as HTMLVideoElement;
      if (videoElement) {
        videoElement.play();
      }
    }
  }, [playingVideo]);
  
  const handlePlayAudio = useCallback((id: string) => {
    if (playingAudio === id) {
      setPlayingAudio(null);
      const audioElement = document.getElementById(`audio-${id}`) as HTMLAudioElement;
      if (audioElement) {
        audioElement.pause();
      }
    } else {
      // Pause any playing audio
      if (playingAudio) {
        const previousAudio = document.getElementById(`audio-${playingAudio}`) as HTMLAudioElement;
        if (previousAudio) {
          previousAudio.pause();
        }
      }
      
      setPlayingAudio(id);
      const audioElement = document.getElementById(`audio-${id}`) as HTMLAudioElement;
      if (audioElement) {
        audioElement.play().catch(error => {
          console.error("Error playing audio:", error);
        });
      }
    }
  }, [playingAudio]);
  
  const handleVideoEnded = useCallback((id: string) => {
    if (playingVideo === id) {
      setPlayingVideo(null);
    }
  }, [playingVideo]);
  
  const handleAudioEnded = useCallback((id: string) => {
    if (playingAudio === id) {
      setPlayingAudio(null);
    }
  }, [playingAudio]);
  
  const renderVideoSamples = () => {
    const { videos } = settings;
    
    if (!videos || videos.length === 0) {
      return (
        <div className="p-4 text-center text-neutral-200">
          <p>No video samples recorded yet</p>
        </div>
      );
    }
    
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 p-4">
        {videos.map(video => (
          <div 
            key={video.id}
            className={`rounded-lg bg-neutral-400 overflow-hidden ${
              activeVideo === video.id ? 'ring-2 ring-primary' : ''
            }`}
            onClick={() => setActiveVideo(video.id === activeVideo ? null : video.id)}
          >
            <div className="relative aspect-video">
              <video 
                id={`video-${video.id}`}
                src={video.url} 
                className="w-full h-full object-cover"
                onEnded={() => handleVideoEnded(video.id)}
                loop={false}
              />
              
              <div className="absolute inset-0 flex justify-center items-center">
                <button
                  className="w-10 h-10 rounded-full bg-black bg-opacity-50 flex items-center justify-center text-white hover:bg-opacity-70 transition-all"
                  onClick={(e) => {
                    e.stopPropagation();
                    handlePlayVideo(video.id);
                  }}
                >
                  {playingVideo === video.id ? <FiPause size={20} /> : <FiPlay size={20} />}
                </button>
              </div>
              
              <div className="absolute top-2 right-2 flex space-x-1">
                {video.isPreferred && (
                  <div className="bg-primary text-white px-2 py-0.5 rounded-full text-xs">
                    Preferred
                  </div>
                )}
                
                <div className={`px-2 py-0.5 rounded-full text-xs ${
                  video.quality >= 85 
                    ? 'bg-green-600 text-white' 
                    : video.quality >= 70 
                      ? 'bg-yellow-600 text-white' 
                      : 'bg-red-600 text-white'
                }`}>
                  {video.quality >= 85 
                    ? 'High Quality' 
                    : video.quality >= 70 
                      ? 'Good Quality' 
                      : 'Low Quality'
                  }
                </div>
              </div>
            </div>
            
            <div className="p-2">
              <div className="flex justify-between items-center">
                <div className="text-sm text-neutral-100">
                  {new Date(video.createdAt).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </div>
                
                <div className="flex space-x-1">
                  <button
                    className={`p-1 rounded-full ${
                      video.isPreferred ? 'text-primary' : 'text-neutral-200 hover:text-primary'
                    }`}
                    onClick={(e) => {
                      e.stopPropagation();
                      markSampleAsPreferred('video', video.id);
                    }}
                    title={video.isPreferred ? "Preferred sample" : "Mark as preferred"}
                  >
                    <FiCheckCircle size={16} />
                  </button>
                  
                  <button
                    className="p-1 text-neutral-200 hover:text-red-500"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteSampleVideo(video.id);
                      if (activeVideo === video.id) {
                        setActiveVideo(null);
                      }
                    }}
                    title="Delete sample"
                  >
                    <FiTrash2 size={16} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };
  
  const renderAudioSamples = () => {
    const { audios } = settings;
    
    if (!audios || audios.length === 0) {
      return (
        <div className="p-4 text-center text-neutral-200">
          <p>No audio samples recorded yet</p>
        </div>
      );
    }
    
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-4">
        {audios.map(audio => (
          <div 
            key={audio.id}
            className={`rounded-lg bg-neutral-400 overflow-hidden ${
              activeAudio === audio.id ? 'ring-2 ring-primary' : ''
            }`}
            onClick={() => setActiveAudio(audio.id === activeAudio ? null : audio.id)}
          >
            <div className="relative p-4">
              <audio 
                id={`audio-${audio.id}`}
                src={audio.url} 
                onEnded={() => handleAudioEnded(audio.id)}
              />
              
              <div className="flex items-center space-x-3">
                <button
                  className="w-10 h-10 rounded-full bg-neutral-300 flex items-center justify-center text-neutral-600 hover:bg-primary hover:text-white transition-all"
                  onClick={(e) => {
                    e.stopPropagation();
                    handlePlayAudio(audio.id);
                  }}
                >
                  {playingAudio === audio.id ? <FiPause size={20} /> : <FiPlay size={20} />}
                </button>
                
                <div className="flex-1">
                  <div className="h-8 bg-neutral-300 rounded-full overflow-hidden relative">
                    {/* Simplified waveform visualization */}
                    <div className="absolute inset-0 flex items-center justify-around">
                      {Array.from({ length: 30 }).map((_, i) => {
                        const height = Math.sin(i * 0.5) * 20 + 30; // Generate a wave pattern
                        return (
                          <div 
                            key={i}
                            className="w-1 bg-neutral-600"
                            style={{ height: `${height}%` }}
                          />
                        );
                      })}
                    </div>
                  </div>
                </div>
                
                <div className="text-sm text-neutral-100">
                  {Math.floor(audio.duration)}s
                </div>
              </div>
              
              <div className="flex justify-between items-center mt-2">
                <div className="flex space-x-1">
                  {audio.isPreferred && (
                    <div className="bg-primary text-white px-2 py-0.5 rounded-full text-xs">
                      Preferred
                    </div>
                  )}
                  
                  <div className={`px-2 py-0.5 rounded-full text-xs ${
                    audio.quality >= 85 
                      ? 'bg-green-600 text-white' 
                      : audio.quality >= 70 
                        ? 'bg-yellow-600 text-white' 
                        : 'bg-red-600 text-white'
                  }`}>
                    {audio.quality >= 85 
                      ? 'High Quality' 
                      : audio.quality >= 70 
                        ? 'Good Quality' 
                        : 'Low Quality'
                    }
                  </div>
                </div>
                
                <div className="flex space-x-1">
                  <button
                    className={`p-1 rounded-full ${
                      audio.isPreferred ? 'text-primary' : 'text-neutral-200 hover:text-primary'
                    }`}
                    onClick={(e) => {
                      e.stopPropagation();
                      markSampleAsPreferred('audio', audio.id);
                    }}
                    title={audio.isPreferred ? "Preferred sample" : "Mark as preferred"}
                  >
                    <FiCheckCircle size={16} />
                  </button>
                  
                  <button
                    className="p-1 text-neutral-200 hover:text-red-500"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteSampleAudio(audio.id);
                      if (activeAudio === audio.id) {
                        setActiveAudio(null);
                      }
                    }}
                    title="Delete sample"
                  >
                    <FiTrash2 size={16} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  };
  
  return (
    <div className="bg-neutral-500 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-neutral-400">
        <h2 className="text-lg font-medium text-neutral-100">Sample Library</h2>
        <p className="text-sm text-neutral-200">
          Manage your recorded video and audio samples
        </p>
      </div>
      
      <div className="border-b border-neutral-400">
        <div className="p-4">
          <div className="flex items-center">
            <FiVideo className="text-primary mr-2" size={20} />
            <h3 className="text-md font-medium text-neutral-100">Video Samples</h3>
          </div>
          <p className="text-sm text-neutral-200 mt-1">
            Facial expressions and movements for avatar creation
          </p>
        </div>
        
        {renderVideoSamples()}
        
        <div className="p-4 bg-neutral-400 rounded-lg m-4">
          <div className="flex items-start">
            <FiInfo className="text-neutral-200 mt-0.5 mr-2 flex-shrink-0" size={16} />
            <div className="text-xs text-neutral-200">
              <p>We recommend recording at least 3 different video samples with varying head movements and expressions for the best avatar quality.</p>
              <p className="mt-1">Mark your best sample as "preferred" to prioritize its features in your avatar.</p>
            </div>
          </div>
        </div>
      </div>
      
      <div>
        <div className="p-4">
          <div className="flex items-center">
            <FiMic className="text-primary mr-2" size={20} />
            <h3 className="text-md font-medium text-neutral-100">Audio Samples</h3>
          </div>
          <p className="text-sm text-neutral-200 mt-1">
            Voice recordings for avatar voice synthesis
          </p>
        </div>
        
        {renderAudioSamples()}
        
        <div className="p-4 bg-neutral-400 rounded-lg m-4">
          <div className="flex items-start">
            <FiInfo className="text-neutral-200 mt-0.5 mr-2 flex-shrink-0" size={16} />
            <div className="text-xs text-neutral-200">
              <p>Recording different speech patterns and tones will help create a more natural-sounding voice for your avatar.</p>
              <p className="mt-1">For best results, record in a quiet environment with minimal background noise.</p>
            </div>
          </div>
        </div>
      </div>
      
      <div className="p-4 bg-yellow-600 bg-opacity-20 border-t border-yellow-600 text-yellow-600">
        <div className="flex items-center">
          <FiAlertTriangle className="mr-2" size={16} />
          <p className="text-sm">
            You must have at least one high-quality video and audio sample to proceed to the next step.
          </p>
        </div>
      </div>
    </div>
  );
} 