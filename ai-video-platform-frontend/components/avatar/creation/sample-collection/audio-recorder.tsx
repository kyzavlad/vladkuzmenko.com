'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { FiMic, FiMicOff, FiPlay, FiPause, FiInfo, FiCheckCircle, FiXCircle } from 'react-icons/fi';
import { useCreationContext, CreationSampleAudio } from '../../contexts/creation-context';

export default function AudioRecorder() {
  const { 
    isRecording, 
    audioStream, 
    error,
    startAudioRecording, 
    stopAudioRecording
  } = useCreationContext();
  
  const [countdown, setCountdown] = useState<number | null>(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [recordingComplete, setRecordingComplete] = useState(false);
  const [recordedSample, setRecordedSample] = useState<CreationSampleAudio | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioAnalyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  const MAX_RECORDING_TIME = 60; // Maximum recording time in seconds
  
  // Set up audio analysis
  useEffect(() => {
    if (!audioStream) return;
    
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      
      const source = audioContext.createMediaStreamSource(audioStream);
      source.connect(analyser);
      
      audioAnalyserRef.current = analyser;
      
      // Start visualization
      visualizeAudio();
    } catch (err) {
      console.error('Error setting up audio analysis:', err);
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [audioStream]);
  
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
  
  // Handle audio playback
  useEffect(() => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.play().catch(err => {
          console.error('Error playing audio:', err);
          setIsPlaying(false);
        });
      } else {
        audioRef.current.pause();
      }
    }
  }, [isPlaying]);
  
  const visualizeAudio = () => {
    if (!canvasRef.current || !audioAnalyserRef.current) return;
    
    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext('2d');
    if (!canvasCtx) return;
    
    const analyser = audioAnalyserRef.current;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const draw = () => {
      if (!canvasRef.current) return;
      
      animationFrameRef.current = requestAnimationFrame(draw);
      
      analyser.getByteFrequencyData(dataArray);
      
      // Calculate average audio level for the indicator
      const average = dataArray.reduce((sum, value) => sum + value, 0) / bufferLength;
      setAudioLevel(average / 256); // Normalize to 0-1 range
      
      // Clear canvas
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Set dimensions
      const width = canvas.width;
      const height = canvas.height;
      
      // Draw background
      canvasCtx.fillStyle = 'rgba(30, 30, 30, 0.2)';
      canvasCtx.fillRect(0, 0, width, height);
      
      // Draw waveform/bars
      const barWidth = (width / bufferLength) * 2.5;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 256) * height;
        
        // Generate gradient from blue to purple based on frequency
        const hue = 240 - (i / bufferLength) * 60;
        canvasCtx.fillStyle = `hsl(${hue}, 70%, 60%)`;
        
        canvasCtx.fillRect(x, height - barHeight, barWidth, barHeight);
        
        x += barWidth + 1;
      }
    };
    
    draw();
  };
  
  const handleStartCountdown = () => {
    setCountdown(3);
    setRecordingTime(0);
    setRecordingComplete(false);
    setRecordedSample(null);
  };
  
  const handleStartRecording = async () => {
    try {
      await startAudioRecording();
    } catch (err) {
      console.error('Failed to start recording:', err);
    }
  };
  
  const handleStopRecording = async () => {
    try {
      const sample = await stopAudioRecording();
      setRecordingComplete(true);
      setRecordedSample(sample);
    } catch (err) {
      console.error('Failed to stop recording:', err);
    }
  };
  
  const handlePlayPause = () => {
    if (!recordedSample) return;
    setIsPlaying(prev => !prev);
  };
  
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };
  
  const renderWaveform = () => {
    if (!recordedSample) return null;
    
    return (
      <div className="h-24 bg-neutral-400 rounded-md overflow-hidden">
        <div className="h-full flex items-center p-2">
          {recordedSample.waveform.map((value, index) => (
            <div
              key={index}
              className="w-1 mx-0.5 bg-primary"
              style={{
                height: `${value * 0.8}%`,
                opacity: isPlaying 
                  ? ((audioRef.current?.currentTime || 0) / (audioRef.current?.duration || 1)) > (index / recordedSample.waveform.length) 
                    ? 1 
                    : 0.4
                  : 0.4
              }}
            />
          ))}
        </div>
      </div>
    );
  };
  
  const renderRecordingIndicator = () => {
    if (!isRecording) return null;
    
    return (
      <div className="absolute top-4 left-4 flex items-center">
        <div 
          className="h-3 w-3 rounded-full bg-red-500 mr-2" 
          style={{ transform: `scale(${1 + audioLevel * 0.5})` }}
        />
        <span className="text-white text-sm font-medium">
          Recording: {formatTime(recordingTime)}
        </span>
      </div>
    );
  };
  
  const renderAudioLevelIndicator = () => {
    if (!isRecording) return null;
    
    // Determine level color
    let levelColor = 'bg-green-500';
    if (audioLevel > 0.8) levelColor = 'bg-red-500';
    else if (audioLevel > 0.6) levelColor = 'bg-yellow-500';
    
    return (
      <div className="absolute bottom-0 left-0 right-0 p-4">
        <div className="w-full bg-neutral-400 rounded-full h-2.5">
          <div
            className={`h-2.5 rounded-full ${levelColor}`}
            style={{ width: `${audioLevel * 100}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-neutral-200 mt-1">
          <span>Low</span>
          <span>Good</span>
          <span>Too High</span>
        </div>
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
        <h2 className="text-lg font-medium text-neutral-100">Voice Sample</h2>
        <p className="text-sm text-neutral-200">
          Record a clear audio sample of your voice for the best avatar results
        </p>
      </div>
      
      <div className="relative bg-neutral-800 aspect-video">
        {/* Audio canvas visualization */}
        <canvas
          ref={canvasRef}
          className="w-full h-full"
          width={512}
          height={200}
        />
        
        {/* Hidden audio player for playback */}
        {recordedSample && (
          <audio
            ref={audioRef}
            src={recordedSample.url}
            onEnded={() => setIsPlaying(false)}
            onPause={() => setIsPlaying(false)}
            className="hidden"
          />
        )}
        
        {/* UI Overlays */}
        {renderRecordingIndicator()}
        {renderAudioLevelIndicator()}
        {renderCountdown()}
        
        {/* Center mic icon */}
        {!isRecording && !recordingComplete && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-neutral-300">
              <FiMic size={48} className="mx-auto mb-2" />
              <p>Ready to record your voice</p>
            </div>
          </div>
        )}
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
                <FiMic className="mr-2" />
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
                <FiMicOff className="mr-2" />
                Stop Recording
              </motion.button>
            )}
            
            {recordingComplete && recordedSample && (
              <motion.button
                className="bg-neutral-300 text-neutral-100 px-4 py-2 rounded-lg font-medium flex items-center"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handlePlayPause}
              >
                {isPlaying ? (
                  <>
                    <FiPause className="mr-2" />
                    Pause
                  </>
                ) : (
                  <>
                    <FiPlay className="mr-2" />
                    Play
                  </>
                )}
              </motion.button>
            )}
          </div>
          
          <div className="text-neutral-200 text-sm">
            {recordingComplete && recordedSample && (
              <span>Duration: {formatTime(Math.floor(recordedSample.duration))}</span>
            )}
          </div>
        </div>
        
        {/* Recorded waveform */}
        {recordingComplete && recordedSample && (
          <div className="mt-4">
            {renderWaveform()}
            
            <div className="mt-4 flex items-center justify-between text-sm text-neutral-200">
              <div>
                <p>Quality Score: <span className="font-medium">{recordedSample.quality}/100</span></p>
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