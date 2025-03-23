'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { Avatar, useAvatarContext } from './avatar-context';
import { v4 as uuidv4 } from 'uuid';

// Generation models
export type GenerationModel = 'standard' | 'enhanced' | 'ultra';

// Generation status
export type GenerationStatus = 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';

// Script text with formatting and emotions
export interface ScriptSegment {
  id: string;
  text: string;
  emotion?: 'neutral' | 'happy' | 'sad' | 'angry' | 'surprised' | 'thoughtful';
  emphasis?: boolean;
  pause?: number; // Pause after segment in milliseconds
  pronunciation?: string; // Optional pronunciation guide
}

export interface Background {
  id: string;
  name: string;
  thumbnail: string;
  description: string;
  videoUrl?: string;
  type: 'color' | 'gradient' | 'image' | 'video' | 'environment';
  category: string;
}

export interface CameraAngle {
  id: string;
  name: string;
  thumbnail: string;
  description: string;
  zoom: number; // 0-100
  position: {
    x: number; // -100 to 100
    y: number; // -100 to 100
  };
}

export interface LightingSetup {
  id: string;
  name: string;
  thumbnail: string;
  description: string;
  brightness: number;
  contrast: number;
  temperature: number;
  direction: 'front' | 'side' | 'top' | '3-point';
}

export interface Prop {
  id: string;
  name: string;
  thumbnail: string;
  description: string;
  category: string;
  position: {
    x: number;
    y: number;
    z: number;
    rotation: number;
    scale: number;
  };
}

export interface SceneSetup {
  background?: Background;
  cameraAngle?: CameraAngle;
  lightingSetup?: LightingSetup;
  props: Prop[];
  avatarPosition?: {
    x: number;
    y: number;
    scale: number;
  };
}

export interface GenerationSettings {
  avatar: Avatar | null;
  script: ScriptSegment[];
  scene: SceneSetup;
  model: GenerationModel;
  resolution: '720p' | '1080p' | '4k';
  priority: 'normal' | 'high';
  textAlignment: 'center' | 'left' | 'right';
  outputFormat: 'mp4' | 'mov' | 'webm';
}

export interface GenerationJob {
  id: string;
  avatarId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  duration: number;
  sceneSetup: SceneSetup;
  script: {
    segments: ScriptSegment[];
    wordCount?: { [key: number]: number };
    duration?: { [key: number]: number };
  };
  createdAt: Date;
  updatedAt: Date;
  processingStages?: {
    name: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress: number;
  }[];
  error?: string;
  previewUrl?: string;
  outputUrl?: string;
  estimatedCompletionTime?: string;
}

export interface GenerationContextProps {
  activeJob: GenerationJob | null;
  jobs: GenerationJob[];
  settings: GenerationSettings;
  isProcessing: boolean;
  error: string | null;
  progress: number;
  updateSettings: (settings: Partial<GenerationSettings>) => void;
  updateSceneSetup: (setup: Partial<SceneSetup>) => void;
  generateScenePreview: () => Promise<void>;
  addJob: (job: GenerationJob) => void;
  removeJob: (jobId: string) => void;
  setActiveJob: (jobId: string | null) => void;
  updateScript: (script: ScriptSegment[]) => void;
  addScriptSegment: (segment: ScriptSegment) => void;
  removeScriptSegment: (index: number) => void;
  loadScriptTemplate: (template: string) => void;
  generatePreview: () => Promise<void>;
  generateVideo: () => Promise<void>;
  cancelGeneration: () => void;
  clearError: () => void;
  generateScript: (prompt: string) => Promise<ScriptSegment[]>;
}

const GenerationContext = createContext<GenerationContextProps>({} as GenerationContextProps);

export const useGenerationContext = () => useContext(GenerationContext);

// Sample data
export const SAMPLE_BACKGROUNDS: Background[] = [
  {
    id: 'office',
    name: 'Modern Office',
    thumbnail: '/backgrounds/office.jpg',
    description: 'A clean, modern office space with natural lighting',
    type: 'environment',
    category: 'indoor'
  },
  {
    id: 'studio',
    name: 'Professional Studio',
    thumbnail: '/backgrounds/studio.jpg',
    description: 'A professional video production studio',
    type: 'environment',
    category: 'indoor'
  },
  {
    id: 'outdoor',
    name: 'Outdoor Scene',
    thumbnail: '/backgrounds/outdoor.jpg',
    description: 'A natural outdoor environment',
    type: 'environment',
    category: 'outdoor'
  }
];

export const SAMPLE_CAMERA_ANGLES: CameraAngle[] = [
  {
    id: 'front',
    name: 'Front View',
    thumbnail: '/camera-angles/front.jpg',
    description: 'Direct front view of the avatar',
    zoom: 100,
    position: { x: 0, y: 0 }
  },
  {
    id: 'three-quarter',
    name: 'Three-Quarter View',
    thumbnail: '/camera-angles/three-quarter.jpg',
    description: 'Slightly angled view for more depth',
    zoom: 100,
    position: { x: 0, y: 0 }
  },
  {
    id: 'profile',
    name: 'Profile View',
    thumbnail: '/camera-angles/profile.jpg',
    description: 'Side view of the avatar',
    zoom: 100,
    position: { x: 0, y: 0 }
  }
];

export const SAMPLE_LIGHTING_SETUPS: LightingSetup[] = [
  {
    id: 'natural',
    name: 'Natural Lighting',
    thumbnail: '/lighting/natural.jpg',
    description: 'Soft, natural lighting setup',
    brightness: 80,
    contrast: 60,
    temperature: 5500,
    direction: 'front'
  },
  {
    id: 'studio',
    name: 'Studio Lighting',
    thumbnail: '/lighting/studio.jpg',
    description: 'Professional studio lighting setup',
    brightness: 90,
    contrast: 70,
    temperature: 6500,
    direction: 'front'
  },
  {
    id: 'dramatic',
    name: 'Dramatic Lighting',
    thumbnail: '/lighting/dramatic.jpg',
    description: 'Dramatic lighting for emphasis',
    brightness: 70,
    contrast: 80,
    temperature: 4500,
    direction: 'front'
  },
  {
    id: 'side',
    name: 'Side Lighting',
    thumbnail: '/lighting/side.jpg',
    description: 'Side lighting for depth',
    brightness: 75,
    contrast: 65,
    temperature: 5000,
    direction: 'side'
  },
  {
    id: 'three-point',
    name: 'Three-Point Lighting',
    thumbnail: '/lighting/three-point.jpg',
    description: 'Classic three-point lighting setup',
    brightness: 85,
    contrast: 70,
    temperature: 6000,
    direction: '3-point'
  }
];

export const SAMPLE_PROPS: Prop[] = [
  {
    id: 'desk',
    name: 'Desk',
    thumbnail: '/props/desk.jpg',
    description: 'Modern office desk',
    category: 'furniture',
    position: { x: 0, y: 0, z: 0, rotation: 0, scale: 1 }
  },
  {
    id: 'chair',
    name: 'Chair',
    thumbnail: '/props/chair.jpg',
    description: 'Office chair',
    category: 'furniture',
    position: { x: 0, y: 0, z: 0, rotation: 0, scale: 1 }
  },
  {
    id: 'screen',
    name: 'Screen',
    thumbnail: '/props/screen.jpg',
    description: 'Computer screen',
    category: 'electronics',
    position: { x: 0, y: 0, z: 0, rotation: 0, scale: 1 }
  },
  {
    id: 'laptop',
    name: 'Laptop',
    thumbnail: '/props/laptop.jpg',
    description: 'Modern laptop',
    category: 'electronics',
    position: { x: 0, y: 0, z: 0, rotation: 0, scale: 1 }
  },
  {
    id: 'plant',
    name: 'Plant',
    thumbnail: '/props/plant.jpg',
    description: 'Decorative plant',
    category: 'decor',
    position: { x: 0, y: 0, z: 0, rotation: 0, scale: 1 }
  }
];

const SCRIPT_TEMPLATES = [
  {
    id: 'introduction',
    name: 'Self Introduction',
    text: "Hello, my name is [Name] and I'm excited to share with you today about [Topic]. I have extensive experience in this field, and I believe that my insights will help you understand the key aspects of this subject."
  },
  {
    id: 'product-promo',
    name: 'Product Promotion',
    text: "Introducing our latest product, designed to solve the challenges you face every day. With its innovative features and user-friendly interface, you'll wonder how you ever managed without it. Let me walk you through what makes it special."
  },
  {
    id: 'tutorial',
    name: 'Tutorial Introduction',
    text: "Today, I'll be showing you step by step how to [Action]. Whether you're a beginner or experienced, these techniques will help you achieve better results. Let's get started!"
  },
  {
    id: 'presentation',
    name: 'Presentation Opening',
    text: "Welcome everyone! Today's presentation focuses on [Topic]. We'll explore the key trends, challenges, and opportunities in this space. I'm looking forward to sharing these insights with you."
  },
  {
    id: 'testimonial',
    name: 'Customer Testimonial',
    text: "I've been using this solution for [Time Period], and it has completely transformed how I work. The results have been incredible, and I can't imagine going back to my old way of doing things."
  }
];

export const GenerationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { avatars } = useAvatarContext();
  
  const [selectedAvatar, setSelectedAvatar] = useState<Avatar | null>(null);
  const [activeJob, setActiveJob] = useState<GenerationJob | null>(null);
  const [recentJobs, setRecentJobs] = useState<GenerationJob[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  
  const defaultBackground = SAMPLE_BACKGROUNDS[0];
  const defaultCamera = SAMPLE_CAMERA_ANGLES[0];
  const defaultLighting = SAMPLE_LIGHTING_SETUPS[0];
  
  const [settings, setSettings] = useState<GenerationSettings>({
    avatar: null,
    script: [],
    scene: {
      background: defaultBackground,
      cameraAngle: defaultCamera,
      lightingSetup: defaultLighting,
      props: []
    },
    model: 'standard',
    resolution: '1080p',
    priority: 'normal',
    textAlignment: 'center',
    outputFormat: 'mp4'
  });
  
  // Script functions
  const updateScript = useCallback((segments: ScriptSegment[]) => {
    setSettings(prev => ({
      ...prev,
      script: segments
    }));
  }, []);
  
  const addScriptSegment = useCallback((segment: Omit<ScriptSegment, 'id'>) => {
    const newSegment: ScriptSegment = {
      ...segment,
      id: `segment-${Date.now()}-${Math.floor(Math.random() * 1000)}`
    };
    
    setSettings(prev => ({
      ...prev,
      script: [...prev.script, newSegment]
    }));
  }, []);
  
  const updateScriptSegment = useCallback((id: string, data: Partial<Omit<ScriptSegment, 'id'>>) => {
    setSettings(prev => ({
      ...prev,
      script: prev.script.map(segment => 
        segment.id === id 
          ? { ...segment, ...data } 
          : segment
      )
    }));
  }, []);
  
  const removeScriptSegment = useCallback((index: number) => {
    setSettings(prev => ({
      ...prev,
      script: prev.script.filter((_, i) => i !== index)
    }));
  }, []);
  
  const loadScriptTemplate = useCallback((templateId: string) => {
    const template = SCRIPT_TEMPLATES.find(t => t.id === templateId);
    if (!template) return;
    
    // Split template text into sentences to create segments
    const sentences = template.text.split(/(?<=\.|\?|\!)\s+/);
    
    const segments: ScriptSegment[] = sentences.map((text, index) => ({
      id: `segment-${Date.now()}-${index}`,
      text,
      emotion: 'neutral',
      emphasis: false
    }));
    
    setSettings(prev => ({
      ...prev,
      script: segments
    }));
  }, []);
  
  const generateScriptWithAI = useCallback(async (prompt: string): Promise<ScriptSegment[]> => {
    try {
      setError(null);
      setIsProcessing(true);
      
      // In a real app, this would be an API call to a text generation service
      // For this demo, we'll just simulate the response
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Generate some fake script segments
      const sampleText = `Hello everyone! Today I'm going to talk about a very interesting topic.
        This is something I'm passionate about and I think you'll find it valuable too.
        Let me break it down into three main points.
        First, we'll explore the background and history.
        Then, we'll look at current trends and developments.
        Finally, I'll share some insights about future possibilities.`;
      
      const sentences = sampleText.split(/(?<=\.|\?|\!)\s+/);
      
      const segments: ScriptSegment[] = sentences.map((text, index) => {
        // Add some random emotions and emphasis
        const emotions: Array<ScriptSegment['emotion']> = ['neutral', 'happy', 'thoughtful'];
        const emotion = emotions[Math.floor(Math.random() * emotions.length)];
        const emphasis = Math.random() > 0.7;
        
        return {
          id: `segment-${Date.now()}-${index}`,
          text: text.trim(),
          emotion,
          emphasis,
          pause: index === sentences.length - 1 ? 500 : undefined // Add pause after last segment
        };
      });
      
      return segments;
    } catch (err) {
      console.error(err);
      setError('Failed to generate script with AI');
      return [];
    } finally {
      setIsProcessing(false);
    }
  }, []);
  
  // Scene configuration functions
  const updateSceneBackground = useCallback((background: Background) => {
    setSettings(prev => ({
      ...prev,
      scene: {
        ...prev.scene,
        background
      }
    }));
  }, []);
  
  const updateCameraAngle = useCallback((cameraAngle: CameraAngle) => {
    setSettings(prev => ({
      ...prev,
      scene: {
        ...prev.scene,
        cameraAngle
      }
    }));
  }, []);
  
  const updateLighting = useCallback((lighting: LightingSetup) => {
    setSettings(prev => ({
      ...prev,
      scene: {
        ...prev.scene,
        lightingSetup: lighting
      }
    }));
  }, []);
  
  const addProp = useCallback((prop: Prop) => {
    setSettings(prev => ({
      ...prev,
      scene: {
        ...prev.scene,
        props: [...prev.scene.props, prop]
      }
    }));
  }, []);
  
  const removeProp = useCallback((propId: string) => {
    setSettings(prev => ({
      ...prev,
      scene: {
        ...prev.scene,
        props: prev.scene.props.filter(prop => prop.id !== propId)
      }
    }));
  }, []);
  
  const updatePropPosition = useCallback((propId: string, position: Prop['position']) => {
    setSettings(prev => ({
      ...prev,
      scene: {
        ...prev.scene,
        props: prev.scene.props.map(prop => 
          prop.id === propId 
            ? { ...prop, position } 
            : prop
        )
      }
    }));
  }, []);
  
  const updateAvatarPosition = useCallback((position: SceneSetup['avatarPosition']) => {
    setSettings(prev => ({
      ...prev,
      scene: {
        ...prev.scene,
        avatarPosition: position
      }
    }));
  }, []);
  
  // General settings functions
  const updateGenerationModel = useCallback((model: GenerationModel) => {
    setSettings(prev => ({
      ...prev,
      model
    }));
  }, []);
  
  const updateResolution = useCallback((resolution: GenerationSettings['resolution']) => {
    setSettings(prev => ({
      ...prev,
      resolution
    }));
  }, []);
  
  const updatePriority = useCallback((priority: GenerationSettings['priority']) => {
    setSettings(prev => ({
      ...prev,
      priority
    }));
  }, []);
  
  const updateTextAlignment = useCallback((textAlignment: GenerationSettings['textAlignment']) => {
    setSettings(prev => ({
      ...prev,
      textAlignment
    }));
  }, []);
  
  const updateOutputFormat = useCallback((outputFormat: GenerationSettings['outputFormat']) => {
    setSettings(prev => ({
      ...prev,
      outputFormat
    }));
  }, []);
  
  // Generation actions
  const generatePreview = useCallback(async (): Promise<void> => {
    try {
      setError(null);
      setIsProcessing(true);
      
      // In a real app, this would make an API call to generate a preview
      // Here we're just simulating the process
      
      // Validate we have everything we need
      if (!selectedAvatar) {
        throw new Error('Please select an avatar first');
      }
      
      if (settings.script.length === 0) {
        throw new Error('Please add some script text first');
      }
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Create a fake preview URL
      const previewUrl = `/avatars/previews/preview-${Date.now()}.mp4`;
      setPreviewUrl(previewUrl);
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to generate preview';
      setError(errorMessage);
      throw err;
    } finally {
      setIsProcessing(false);
    }
  }, [selectedAvatar, settings.script]);
  
  const startGeneration = useCallback(async () => {
    if (!settings.avatar) {
      setError("No avatar selected");
      return null;
    }

    const newJob: GenerationJob = {
      id: uuidv4(),
      avatarId: settings.avatar.id,
      status: 'pending',
      progress: 0,
      duration: 0,
      sceneSetup: {
        background: defaultBackground,
        cameraAngle: defaultCamera,
        lightingSetup: defaultLighting,
        props: [],
        avatarPosition: {
          x: 0,
          y: 0,
          scale: 1
        }
      },
      script: {
        segments: [],
        wordCount: {},
        duration: {}
      },
      createdAt: new Date(),
      updatedAt: new Date(),
      processingStages: [
        { name: 'Initializing', status: 'pending', progress: 0 },
        { name: 'Generating', status: 'pending', progress: 0 },
        { name: 'Finalizing', status: 'pending', progress: 0 }
      ]
    };
    
    // Start the fake job processing
    simulateJobProgress(newJob);
    
    setActiveJob(newJob);
    return newJob;
  }, [settings]);
  
  // Helper function to simulate job progress
  const simulateJobProgress = (job: GenerationJob) => {
    let currentStage = 0;
    
    const updateJob = (updatedJob: GenerationJob) => {
      setActiveJob(updatedJob);
      
      // Also update in recent jobs if it exists there
      setRecentJobs(prev => prev.map(j => 
        j.id === updatedJob.id ? updatedJob : j
      ));
    };
    
    // Start with "processing" status
    updateJob({
      ...job,
      status: 'processing',
      updatedAt: new Date(),
      processingStages: job.processingStages?.map((stage, idx) => ({
        ...stage,
        status: idx === 0 ? 'processing' : 'pending'
      }))
    });
    
    const intervalId = setInterval(() => {
      setActiveJob(prevJob => {
        if (!prevJob || prevJob.id !== job.id) {
          clearInterval(intervalId);
          return prevJob;
        }
        
        // Update the current stage progress
        const updatedStages = [...(prevJob.processingStages || [])];
        if (updatedStages[currentStage]) {
          updatedStages[currentStage] = {
            ...updatedStages[currentStage],
            progress: Math.min(updatedStages[currentStage].progress + Math.random() * 15, 100)
          };
        }
        
        // Check if the current stage is complete
        if (updatedStages[currentStage]?.progress >= 100) {
          updatedStages[currentStage] = {
            ...updatedStages[currentStage],
            status: 'completed',
            progress: 100
          };
          
          // Move to the next stage
          currentStage++;
          
          // If there's a next stage, set it to processing
          if (currentStage < updatedStages.length) {
            updatedStages[currentStage] = {
              ...updatedStages[currentStage],
              status: 'processing'
            };
          }
        }
        
        // Calculate overall progress
        const overallProgress = Math.min(
          Math.floor(
            ((currentStage * 100) + 
             (currentStage < updatedStages.length ? updatedStages[currentStage].progress : 0)) / 
            updatedStages.length
          ),
          99 // Never reach 100% until fully complete
        );
        
        // Check if job is complete
        if (currentStage >= updatedStages.length) {
          clearInterval(intervalId);
          
          // Set a timeout to finalize the job
          setTimeout(() => {
            const completedJob: GenerationJob = {
              ...prevJob,
              status: 'completed',
              progress: 100,
              processingStages: updatedStages.map(stage => ({
                ...stage,
                status: 'completed',
                progress: 100
              })),
              previewUrl: `preview-${prevJob.id}.mp4`,
              outputUrl: `output-${prevJob.id}.mp4`,
              updatedAt: new Date()
            };
            
            updateJob(completedJob);
            
            // Add to recent jobs if not already there
            setRecentJobs(prev => {
              const exists = prev.some(j => j.id === completedJob.id);
              return exists 
                ? prev.map(j => j.id === completedJob.id ? completedJob : j)
                : [completedJob, ...prev].slice(0, 10);
            });
          }, 1000);
          
          return prevJob;
        }
        
        // Return the updated job
        const updatedJob = {
          ...prevJob,
          progress: overallProgress,
          processingStages: updatedStages,
          updatedAt: new Date()
        };
        
        return updatedJob;
      });
    }, 1000);
  };
  
  const cancelGeneration = useCallback(async (jobId: string): Promise<void> => {
    try {
      setError(null);
      
      // In a real app, this would call an API to cancel
      // Here we just update the local state
      
      setActiveJob(prev => {
        if (!prev || prev.id !== jobId) {
          throw new Error('Job not found');
        }
        
        const cancelledJob: GenerationJob = {
          ...prev,
          status: 'failed',
          error: 'Cancelled by user',
          updatedAt: new Date()
        };
        
        // Update in recent jobs too
        setRecentJobs(jobs => jobs.map(job => 
          job.id === jobId ? cancelledJob : job
        ));
        
        return cancelledJob;
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to cancel generation';
      setError(errorMessage);
      throw err;
    }
  }, []);
  
  const getJobStatus = useCallback(async (jobId: string): Promise<GenerationJob> => {
    // In a real app, this would call an API to get the latest status
    // Here we just return the local state
    
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
  
  // Update settings when avatar changes
  useEffect(() => {
    if (selectedAvatar) {
      setSettings(prev => ({
        ...prev,
        avatar: selectedAvatar
      }));
    }
  }, [selectedAvatar]);
  
  const updateSceneSetup = useCallback((setup: Partial<SceneSetup>) => {
    setSettings(prev => ({
      ...prev,
      scene: {
        ...prev.scene,
        ...setup
      }
    }));
  }, []);

  const generateScenePreview = useCallback(async () => {
    setIsProcessing(true);
    setError(null);
    setProgress(0);

    try {
      // Симуляция генерации превью сцены
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
        setProgress(i);
      }
      
      // В реальном приложении здесь будет API-запрос
      console.log("Scene preview generated successfully");
    } catch (err) {
      setError("Failed to generate scene preview");
      console.error(err);
    } finally {
      setIsProcessing(false);
      setProgress(0);
    }
  }, []);

  const value = {
    activeJob,
    jobs: recentJobs,
    settings,
    isProcessing,
    error,
    progress,
    updateSettings: (partialSettings: Partial<GenerationSettings>) => {
      setSettings(prev => ({
        ...prev,
        ...partialSettings
      }));
    },
    updateSceneSetup,
    generateScenePreview,
    addJob: (job: GenerationJob) => setRecentJobs(prev => [...prev, job].slice(0, 10)),
    removeJob: (jobId: string) => setRecentJobs(prev => prev.filter(job => job.id !== jobId)),
    setActiveJob: (jobId: string | null) => {
      if (jobId === null) {
        setActiveJob(null);
      } else {
        const job = recentJobs.find(j => j.id === jobId);
        setActiveJob(job || null);
      }
    },
    updateScript,
    addScriptSegment,
    removeScriptSegment,
    loadScriptTemplate,
    generatePreview,
    generateVideo: () => Promise.resolve(),
    cancelGeneration: () => Promise.resolve(),
    clearError: () => setError(null),
    generateScript: generateScriptWithAI
  };
  
  return (
    <GenerationContext.Provider value={value}>
      {children}
    </GenerationContext.Provider>
  );
};

// Missing effect import
import { useEffect } from 'react';