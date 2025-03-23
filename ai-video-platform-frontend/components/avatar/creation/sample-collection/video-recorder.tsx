'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { FiVideo, FiVideoOff, FiCamera, FiInfo, FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { useCreationContext, CreationSampleVideo } from '../../contexts/creation-context';

export default function VideoRecorder() {
  const { 
    isRecording, 
    videoStream, 
    error,
    startVideoRecording, 
    stopVideoRecording,
    settings
  } = useCreationContext();
  
  const [countdown, setCountdown] = useState<number | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [showGuidelines, setShowGuidelines] = useState(true);
  const [recordingComplete, setRecordingComplete] = useState(false);
  const [recordedSample, setRecordedSample] = useState<CreationSampleVideo | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  
  const MAX_RECORDING_TIME = 30; // Maximum recording time in seconds
  
  // Connect video stream to video element
  useEffect(() => {
    if (videoRef.current && videoStream) {
      videoRef.current.srcObject = videoStream;
    }
  }, [videoStream]);
  
  // Handle recording timer
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= MAX_RECORDING_TIME) {
            handleStopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [isRecording]);
  
  // Handle countdown
  useEffect(() => {
    if (countdown !== null && countdown > 0) {
      const timer = setTimeout(() => {
        setCountdown(prev => (prev !== null ? prev - 1 : null));
      }, 1000);
      
      return () => clearTimeout(timer);
    } else if (countdown === 0) {
      setCountdown(null);
      handleStartRecording();
    }
  }, [countdown]);
  
  const handleStartCountdown = () => {
    setCountdown(3);
    setRecordingTime(0);
    setRecordingComplete(false);
    setRecordedSample(null);
  };
  
  const handleStartRecording = async () => {
    try {
      await startVideoRecording();
    } catch (err) {
      console.error('Failed to start recording:', err);
    }
  };
  
  const handleStopRecording = async () => {
    try {
      const sample = await stopVideoRecording();
      setRecordingComplete(true);
      setRecordedSample(sample);
    } catch (err) {
      console.error('Failed to stop recording:', err);
    }
  };
  
  const handleTakeSnapshot = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    if (!context) return;
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the current video frame on the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to image data URL
    const imageUrl = canvas.toDataURL('image/jpeg');
    
    // In a real app, you would save this image or use it for preview
    console.log('Snapshot taken:', imageUrl);
  };
  
  const toggleGuidelines = () => {
    setShowGuidelines(prev => !prev);
  };
  
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  const renderFaceOutline = () => {
    if (!showGuidelines) return null;
    
    return (
      <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
        <div className="w-1/2 h-3/4 border-2 border-white border-opacity-50 rounded-full flex items-center justify-center">
          <div className="text-white text-opacity-70 text-center">
            <p className="text-sm">Position your face here</p>
          </div>
        </div>
      </div>
    );
  };
  
  const renderGuidelines = () => {
    if (!showGuidelines) return null;
    
    return (
      <div className="absolute bottom-4 left-4 right-4 bg-black bg-opacity-60 p-3 rounded-lg text-white">
        <h3 className="text-sm font-medium mb-2 flex items-center">
          <FiInfo className="mr-2" /> Recording Guidelines
        </h3>
        <ul className="text-xs space-y-1 list-disc pl-5">
          <li>Face the camera directly and center your face in the outline</li>
          <li>Ensure good lighting on your face (avoid backlighting)</li>
          <li>Speak clearly and naturally with neutral expressions</li>
          <li>Avoid rapid head movements during recording</li>
          <li>Record in a quiet environment with minimal background noise</li>
        </ul>
      </div>
    );
  };
  
  const renderRecordingIndicator = () => {
    if (!isRecording) return null;
    
    return (
      <div className="absolute top-4 left-4 flex items-center">
        <div className="h-3 w-3 rounded-full bg-red-500 animate-pulse mr-2"></div>
        <span className="text-white text-sm font-medium">
          Recording: {formatTime(recordingTime)}
        </span>
      </div>
    );
  };
  
  const renderCountdown = () => {
    if (countdown === null) return null;
    
    return (
      <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
        <div className="text-white text-8xl font-bold animate-pulse">
          {countdown}
        </div>
      </div>
    );
  };
  
  const renderQualityIssues = () => {
    if (!recordedSample || !recordedSample.issues || recordedSample.issues.length === 0) {
      return null;
    }
    
    return (
      <div className="mt-4 bg-neutral-400 p-3 rounded-lg">
        <h3 className="text-sm font-medium mb-2 flex items-center text-neutral-100">
          <FiInfo className="mr-2" /> Quality Check
        </h3>
        <ul className="space-y-2">
          {recordedSample.issues.map(issue => (
            <li key={issue.type} className="flex items-start text-xs">
              {issue.severity === 'high' ? (
                <FiXCircle className="text-red-500 mt-0.5 mr-2 flex-shrink-0" />
              ) : issue.severity === 'medium' ? (
                <FiInfo className="text-yellow-500 mt-0.5 mr-2 flex-shrink-0" />
              ) : (
                <FiCheckCircle className="text-green-500 mt-0.5 mr-2 flex-shrink-0" />
              )}
              <span className="text-neutral-200">{issue.message}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  };
  
  return (
    <div className="bg-neutral-500 rounded-lg overflow-hidden">
      <div className="p-4 border-b border-neutral-400">
        <h2 className="text-lg font-medium text-neutral-100">Video Sample</h2>
        <p className="text-sm text-neutral-200">
          Record a clear video of your face for the best avatar results
        </p>
      </div>
      
      <div className="relative bg-black aspect-video">
        {/* Video Preview */}
        {videoStream ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
        ) : recordedSample ? (
          <video
            src={recordedSample.url}
            controls
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-neutral-800">
            <div className="text-center text-neutral-300">
              <FiVideo size={48} className="mx-auto mb-2" />
              <p>Video preview will appear here</p>
            </div>
          </div>
        )}
        
        {/* Hidden canvas for snapshots */}
        <canvas ref={canvasRef} className="hidden" />
        
        {/* UI Overlays */}
        {renderFaceOutline()}
        {renderGuidelines()}
        {renderRecordingIndicator()}
        {renderCountdown()}
      </div>
      
      {/* Controls */}
      <div className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex space-x-2">
            {!isRecording && !recordingComplete && (
              <motion.button
                className="bg-primary text-white px-4 py-2 rounded-lg font-medium flex items-center"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleStartCountdown}
                disabled={!!error}
              >
                <FiVideo className="mr-2" />
                Start Recording
              </motion.button>
            )}
            
            {isRecording && (
              <motion.button
                className="bg-red-500 text-white px-4 py-2 rounded-lg font-medium flex items-center"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleStopRecording}
              >
                <FiVideoOff className="mr-2" />
                Stop Recording
              </motion.button>
            )}
            
            {videoStream && !isRecording && (
              <motion.button
                className="bg-neutral-300 text-neutral-100 px-4 py-2 rounded-lg font-medium flex items-center"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleTakeSnapshot}
              >
                <FiCamera className="mr-2" />
                Take Snapshot
              </motion.button>
            )}
          </div>
          
          <motion.button
            className="text-neutral-200 hover:text-neutral-100 p-2"
            whileHover={{ scale: 1.1 }}
            onClick={toggleGuidelines}
          >
            <FiInfo size={20} />
          </motion.button>
        </div>
        
        {recordingComplete && recordedSample && (
          <div className="mt-4">
            <div className="flex items-center justify-between text-sm text-neutral-200">
              <div>
                <p>Quality Score: <span className="font-medium">{recordedSample.quality}/100</span></p>
                <p>Duration: <span className="font-medium">{recordedSample.duration.toFixed(1)}s</span></p>
              </div>
              <div className="flex items-center">
                {recordedSample.quality >= 85 ? (
                  <><FiCheckCircle className="text-green-500 mr-1" /> Excellent</>
                ) : recordedSample.quality >= 70 ? (
                  <><FiCheckCircle className="text-yellow-500 mr-1" /> Good</>
                ) : (
                  <><FiXCircle className="text-red-500 mr-1" /> Poor</>
                )}
              </div>
            </div>
            
            {renderQualityIssues()}
            
            <div className="mt-4 flex justify-end space-x-3">
              <motion.button
                className="text-neutral-200 hover:text-neutral-100 px-3 py-1.5 border border-neutral-300 rounded-lg text-sm"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleStartCountdown}
              >
                Record Again
              </motion.button>
              
              <motion.button
                className={`text-white px-3 py-1.5 rounded-lg text-sm font-medium ${
                  recordedSample.quality >= 70 ? 'bg-primary' : 'bg-neutral-300'
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                // Here you would typically save or confirm the sample
              >
                {recordedSample.quality >= 70 ? 'Use This Sample' : 'Save Anyway'}
              </motion.button>
            </div>
          </div>
        )}
        
        {error && (
          <div className="mt-4 p-3 bg-red-500 bg-opacity-10 border border-red-500 text-red-500 rounded-lg text-sm">
            {error}
          </div>
        )}
      </div>
    </div>
  );
} 