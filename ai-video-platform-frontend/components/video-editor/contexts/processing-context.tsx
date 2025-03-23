'use client';

import React, { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';
import { MediaFile } from './media-context';
import { EditSettings } from './editor-context';

export interface ProcessingJob {
  id: string;
  mediaFileId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  stages: {
    name: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress: number;
  }[];
  settings: EditSettings;
  result?: {
    fileUrl: string;
    thumbnailUrl: string;
    duration: number;
    fileSize: number;
  };
  error?: string;
  estimatedTimeRemaining?: number;
  resourceUsage?: {
    cpu: number;
    memory: number;
    gpu: number;
  };
  createdAt: string;
  updatedAt: string;
}

interface ProcessingContextProps {
  activeJob: ProcessingJob | null;
  recentJobs: ProcessingJob[];
  isProcessing: boolean;
  startProcessing: (mediaFile: MediaFile, settings: EditSettings) => Promise<ProcessingJob>;
  cancelProcessing: (jobId: string) => Promise<void>;
  getJobStatus: (jobId: string) => Promise<ProcessingJob>;
  clearActiveJob: () => void;
}

const ProcessingContext = createContext<ProcessingContextProps>({} as ProcessingContextProps);

export const useProcessingContext = () => useContext(ProcessingContext);

// Processing stages
const PROCESSING_STAGES = [
  { name: 'Analyzing video', duration: 2000 },
  { name: 'Transcribing audio', duration: 3000 },
  { name: 'Generating subtitles', duration: 2000 },
  { name: 'Finding B-roll opportunities', duration: 2500 },
  { name: 'Enhancing audio quality', duration: 1500 },
  { name: 'Removing pauses', duration: 1000 },
  { name: 'Adding music and sound effects', duration: 2000 },
  { name: 'Enhancing video quality', duration: 2500 },
  { name: 'Rendering final output', duration: 3500 }
];

export const ProcessingProvider = ({ children }: { children: ReactNode }) => {
  const [activeJob, setActiveJob] = useState<ProcessingJob | null>(null);
  const [recentJobs, setRecentJobs] = useState<ProcessingJob[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  
  // WebSocket simulation for real-time updates
  useEffect(() => {
    if (!activeJob || activeJob.status !== 'processing') return;
    
    let currentStageIndex = 0;
    let currentProgress = 0;
    let timeElapsed = 0;
    
    const updateInterval = setInterval(() => {
      if (!activeJob) {
        clearInterval(updateInterval);
        return;
      }
      
      const currentStage = PROCESSING_STAGES[currentStageIndex];
      currentProgress += 1;
      timeElapsed += 100;
      
      // Update current stage progress
      const updatedStages = [...activeJob.stages];
      updatedStages[currentStageIndex] = {
        ...updatedStages[currentStageIndex],
        progress: currentProgress,
        status: 'processing'
      };
      
      // Move to next stage if current stage is complete
      if (currentProgress >= 100) {
        updatedStages[currentStageIndex].status = 'completed';
        currentStageIndex++;
        currentProgress = 0;
        
        // Mark next stage as processing if available
        if (currentStageIndex < updatedStages.length) {
          updatedStages[currentStageIndex].status = 'processing';
        }
      }
      
      // Calculate overall progress
      const overallProgress = Math.min(
        Math.round((currentStageIndex * 100 + currentProgress) / PROCESSING_STAGES.length),
        99 // Never reach 100% until fully complete
      );
      
      // Calculate estimated time remaining
      const totalDuration = PROCESSING_STAGES.reduce((acc, stage) => acc + stage.duration, 0);
      const estimatedTimeRemaining = Math.max(0, totalDuration - timeElapsed);
      
      // Simulate resource usage fluctuations
      const resourceUsage = {
        cpu: 50 + Math.sin(timeElapsed / 1000) * 20,
        memory: 40 + Math.cos(timeElapsed / 1500) * 15,
        gpu: 60 + Math.sin(timeElapsed / 2000) * 25
      };
      
      const updatedJob: ProcessingJob = {
        ...activeJob,
        progress: overallProgress,
        stages: updatedStages,
        estimatedTimeRemaining,
        resourceUsage,
        updatedAt: new Date().toISOString()
      };
      
      setActiveJob(updatedJob);
      
      // Complete the job
      if (currentStageIndex >= PROCESSING_STAGES.length) {
        clearInterval(updateInterval);
        
        // Simulate final processing
        setTimeout(() => {
          const completedJob: ProcessingJob = {
            ...updatedJob,
            status: 'completed',
            progress: 100,
            stages: updatedJob.stages.map(stage => ({ ...stage, status: 'completed', progress: 100 })),
            result: {
              fileUrl: `https://storage.example.com/processed/${updatedJob.id}.mp4`,
              thumbnailUrl: `https://storage.example.com/thumbnails/${updatedJob.id}.jpg`,
              duration: 120, // 2 minutes
              fileSize: 15728640, // 15MB
            },
            updatedAt: new Date().toISOString()
          };
          
          setActiveJob(completedJob);
          setIsProcessing(false);
          
          // Add to recent jobs
          setRecentJobs(prev => [completedJob, ...prev].slice(0, 10));
        }, 1000);
      }
    }, 100);
    
    return () => clearInterval(updateInterval);
  }, [activeJob]);
  
  const startProcessing = useCallback(async (mediaFile: MediaFile, settings: EditSettings): Promise<ProcessingJob> => {
    // In a real app, you would make an API call:
    // const response = await fetch('/api/video/edit', {
    //   method: 'POST',
    //   body: JSON.stringify({ mediaFileId: mediaFile.id, settings }),
    //   headers: { 'Content-Type': 'application/json' }
    // });
    // const jobData = await response.json();
    
    // Create processing job with initial state
    const newJob: ProcessingJob = {
      id: `job-${Date.now()}`,
      mediaFileId: mediaFile.id,
      status: 'processing',
      progress: 0,
      stages: PROCESSING_STAGES.map((stage, index) => ({
        name: stage.name,
        status: index === 0 ? 'processing' : 'pending',
        progress: index === 0 ? 0 : 0
      })),
      settings,
      estimatedTimeRemaining: PROCESSING_STAGES.reduce((acc, stage) => acc + stage.duration, 0),
      resourceUsage: {
        cpu: 50,
        memory: 40,
        gpu: 60
      },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    
    setActiveJob(newJob);
    setIsProcessing(true);
    
    return newJob;
  }, []);
  
  const cancelProcessing = useCallback(async (jobId: string): Promise<void> => {
    // In a real app, you would make an API call:
    // await fetch(`/api/video/edit/${jobId}/cancel`, {
    //   method: 'POST'
    // });
    
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    if (activeJob && activeJob.id === jobId) {
      const cancelledJob: ProcessingJob = {
        ...activeJob,
        status: 'failed',
        error: 'Processing cancelled by user',
        updatedAt: new Date().toISOString()
      };
      
      setActiveJob(cancelledJob);
      setIsProcessing(false);
      
      // Add to recent jobs
      setRecentJobs(prev => [cancelledJob, ...prev].slice(0, 10));
    }
  }, [activeJob]);
  
  const getJobStatus = useCallback(async (jobId: string): Promise<ProcessingJob> => {
    // In a real app, you would make an API call:
    // const response = await fetch(`/api/video/edit/status/${jobId}`);
    // return await response.json();
    
    // For the demo, just return the active job or find in recent jobs
    if (activeJob && activeJob.id === jobId) {
      return activeJob;
    }
    
    const recentJob = recentJobs.find(job => job.id === jobId);
    if (recentJob) {
      return recentJob;
    }
    
    throw new Error('Job not found');
  }, [activeJob, recentJobs]);
  
  const clearActiveJob = useCallback(() => {
    setActiveJob(null);
  }, []);
  
  const value = {
    activeJob,
    recentJobs,
    isProcessing,
    startProcessing,
    cancelProcessing,
    getJobStatus,
    clearActiveJob
  };
  
  return (
    <ProcessingContext.Provider value={value}>
      {children}
    </ProcessingContext.Provider>
  );
}; 