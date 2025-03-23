'use client';

import React, { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';
import { AvatarStyle, AvatarSample, useAvatarContext } from './avatar-context';

// Creation step types
export type CreationStep = 
  | 'intro'
  | 'sample-collection'
  | 'sample-quality'
  | 'voice-recording'
  | 'reference-upload' 
  | 'style-selection'
  | 'appearance-customize'
  | 'voice-customize'
  | 'animation-customize'
  | 'background-selection'
  | 'preview'
  | 'complete';

export interface CreationSampleVideo {
  id: string;
  blob: Blob;
  url: string;
  thumbnail?: string;
  duration: number;
  quality: number;
  isPreferred?: boolean;
  issues?: Array<{
    type: 'lighting' | 'position' | 'noise' | 'resolution' | 'stability';
    severity: 'low' | 'medium' | 'high';
    message: string;
  }>;
  createdAt: string;
}

export interface CreationSampleAudio {
  id: string;
  blob: Blob;
  url: string;
  duration: number;
  quality: number;
  waveform: number[]; // Audio visualization data
  isPreferred?: boolean;
  issues?: Array<{
    type: 'noise' | 'volume' | 'clarity';
    severity: 'low' | 'medium' | 'high';
    message: string;
  }>;
  createdAt: string;
}

export interface CreationReferenceImage {
  id: string;
  blob: Blob;
  url: string;
  quality: number;
  tags?: string[];
  createdAt: string;
}

interface CreationSettings {
  name: string;
  style: AvatarStyle | null;
  videos: CreationSampleVideo[];
  audios: CreationSampleAudio[];
  referenceImages: CreationReferenceImage[];
  selectedVideoId: string | null;
  selectedAudioId: string | null;
  voiceSettings: {
    pitch: number;
    speed: number;
    clarity: number;
    expressiveness: number;
  };
  appearanceSettings: {
    skinTone: number;
    hairStyle: string;
    hairColor: string;
    eyeColor: string;
    facialFeatures: number;
  };
  animationSettings: {
    gestureIntensity: number;
    expressionIntensity: number;
    movementStyle: string;
  };
  backgroundId: string | null;
  appearance: {
    skinTone: string;
    hairStyle: string;
    hairColor: string;
    eyeColor: string;
    facialFeatures: number;
    age: number;
    jawline: number;
    lightMode: 'day' | 'night';
  };
}

interface CreationContextProps {
  currentStep: CreationStep;
  settings: CreationSettings;
  isRecording: boolean;
  isProcessing: boolean;
  progress: number;
  previewUrl: string | null;
  error: string | null;
  videoStream: MediaStream | null;
  audioStream: MediaStream | null;
  
  goToStep: (step: CreationStep) => void;
  goToNextStep: () => void;
  goToPreviousStep: () => void;
  
  updateName: (name: string) => void;
  setSelectedStyle: (style: AvatarStyle) => void;
  
  startVideoRecording: () => Promise<void>;
  stopVideoRecording: () => Promise<CreationSampleVideo>;
  deleteVideo: (id: string) => void;
  selectVideo: (id: string) => void;
  deleteSampleVideo: (id: string) => void;
  
  startAudioRecording: () => Promise<void>;
  stopAudioRecording: () => Promise<CreationSampleAudio>;
  deleteAudio: (id: string) => void;
  selectAudio: (id: string) => void;
  deleteSampleAudio: (id: string) => void;
  
  uploadReferenceImage: (file: File) => Promise<CreationReferenceImage>;
  deleteReferenceImage: (id: string) => void;
  
  updateVoiceSettings: (settings: Partial<CreationSettings['voiceSettings']>) => void;
  updateAppearanceSettings: (settings: Partial<CreationSettings['appearanceSettings']>) => void;
  updateAnimationSettings: (settings: Partial<CreationSettings['animationSettings']>) => void;
  setBackground: (id: string) => void;
  
  markSampleAsPreferred: (type: 'video' | 'audio', id: string) => void;
  updateSettings: (settings: CreationSettings) => void;
  generateAppearancePreview: (appearance: CreationSettings['appearance']) => Promise<void>;
  generateStylePreview: (style: AvatarStyle) => Promise<void>;
  
  generatePreview: () => Promise<string>;
  saveAvatar: () => Promise<string>;
  resetCreation: () => void;
}

const CreationContext = createContext<CreationContextProps>({} as CreationContextProps);

export const useCreationContext = () => useContext(CreationContext);

const DEFAULT_VOICE_SETTINGS = {
  pitch: 50,
  speed: 50,
  clarity: 80,
  expressiveness: 60
};

const DEFAULT_APPEARANCE_SETTINGS = {
  skinTone: 50,
  hairStyle: 'short',
  hairColor: '#3a3a3a',
  eyeColor: '#724b34',
  facialFeatures: 50
};

const DEFAULT_ANIMATION_SETTINGS = {
  gestureIntensity: 60,
  expressionIntensity: 50,
  movementStyle: 'natural'
};

const DEFAULT_APPEARANCE = {
  skinTone: 'medium',
  hairStyle: 'short',
  hairColor: 'black',
  eyeColor: 'brown',
  facialFeatures: 50,
  age: 30,
  jawline: 50,
  lightMode: 'day' as const
};

export const CreationProvider = ({ children }: { children: ReactNode }) => {
  const { createAvatar } = useAvatarContext();
  
  const [currentStep, setCurrentStep] = useState<CreationStep>('intro');
  const [settings, setSettings] = useState<CreationSettings>({
    name: '',
    style: null,
    videos: [],
    audios: [],
    referenceImages: [],
    selectedVideoId: null,
    selectedAudioId: null,
    voiceSettings: DEFAULT_VOICE_SETTINGS,
    appearanceSettings: DEFAULT_APPEARANCE_SETTINGS,
    animationSettings: DEFAULT_ANIMATION_SETTINGS,
    backgroundId: null,
    appearance: DEFAULT_APPEARANCE
  });
  
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [videoStream, setVideoStream] = useState<MediaStream | null>(null);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  
  const [videoRecorder, setVideoRecorder] = useState<MediaRecorder | null>(null);
  const [audioRecorder, setAudioRecorder] = useState<MediaRecorder | null>(null);
  const [videoChunks, setVideoChunks] = useState<Blob[]>([]);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  
  // Clean up media streams when component unmounts
  useEffect(() => {
    return () => {
      if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
      }
      if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [videoStream, audioStream]);
  
  // Navigation functions
  const goToStep = useCallback((step: CreationStep) => {
    setCurrentStep(step);
  }, []);
  
  const goToNextStep = useCallback(() => {
    const steps: CreationStep[] = [
      'intro',
      'sample-collection',
      'sample-quality',
      'voice-recording',
      'reference-upload',
      'style-selection',
      'appearance-customize',
      'voice-customize',
      'animation-customize',
      'background-selection',
      'preview',
      'complete'
    ];
    
    const currentIndex = steps.indexOf(currentStep);
    if (currentIndex < steps.length - 1) {
      setCurrentStep(steps[currentIndex + 1]);
    }
  }, [currentStep]);
  
  const goToPreviousStep = useCallback(() => {
    const steps: CreationStep[] = [
      'intro',
      'sample-collection',
      'sample-quality',
      'voice-recording',
      'reference-upload',
      'style-selection',
      'appearance-customize',
      'voice-customize',
      'animation-customize',
      'background-selection',
      'preview',
      'complete'
    ];
    
    const currentIndex = steps.indexOf(currentStep);
    if (currentIndex > 0) {
      setCurrentStep(steps[currentIndex - 1]);
    }
  }, [currentStep]);
  
  // Basic settings functions
  const updateName = useCallback((name: string) => {
    setSettings(prev => ({ ...prev, name }));
  }, []);
  
  const setSelectedStyle = useCallback((style: AvatarStyle) => {
    setSettings(prev => ({ ...prev, style }));
  }, []);
  
  // Video recording functions
  const startVideoRecording = useCallback(async () => {
    try {
      setError(null);
      
      // Request camera and mic access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: true
      });
      
      setVideoStream(stream);
      
      // Initialize recorder
      const recorder = new MediaRecorder(stream);
      setVideoRecorder(recorder);
      
      // Set up recorder events
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          setVideoChunks(prev => [...prev, e.data]);
        }
      };
      
      // Start recording
      setVideoChunks([]);
      recorder.start(1000); // Collect data every second
      setIsRecording(true);
    } catch (err) {
      setError('Could not access camera. Please ensure you have granted camera permission.');
      console.error(err);
    }
  }, []);
  
  const stopVideoRecording = useCallback(async (): Promise<CreationSampleVideo> => {
    return new Promise((resolve, reject) => {
      if (!videoRecorder || !videoStream) {
        setError('No active recording found');
        reject('No active recording found');
        return;
      }
      
      // Set up handler for when recording stops
      videoRecorder.onstop = async () => {
        try {
          // Create video blob from chunks
          const videoBlob = new Blob(videoChunks, { type: 'video/webm' });
          const videoUrl = URL.createObjectURL(videoBlob);
          
          // Create video element to get duration
          const video = document.createElement('video');
          video.src = videoUrl;
          
          video.onloadedmetadata = async () => {
            // Generate a thumbnail (in a real app this would be a more complex process)
            const canvas = document.createElement('canvas');
            canvas.width = 320;
            canvas.height = 180;
            video.currentTime = 1; // Set to 1 second in
            
            // Wait for seek to complete
            await new Promise(resolve => {
              video.onseeked = resolve;
            });
            
            const ctx = canvas.getContext('2d');
            if (ctx) {
              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
              const thumbnailUrl = canvas.toDataURL('image/jpeg');
              
              // In a real app, this would be a complex quality analysis
              // Here we're just generating a random quality score
              const quality = Math.floor(Math.random() * 30) + 70; // 70-99
              
              // Generate sample issues based on quality
              const issues = [];
              if (quality < 80) {
                issues.push({
                  type: 'lighting' as const,
                  severity: 'medium' as const,
                  message: 'Lighting is a bit dim, consider adding more light to your face'
                });
              }
              if (quality < 85) {
                issues.push({
                  type: 'position' as const,
                  severity: 'low' as const,
                  message: 'Try to center your face more in the frame'
                });
              }
              
              // Create sample object
              const newSample: CreationSampleVideo = {
                id: `video-${Date.now()}`,
                blob: videoBlob,
                url: videoUrl,
                thumbnail: thumbnailUrl,
                duration: video.duration,
                quality,
                isPreferred: false,
                issues: issues.length > 0 ? issues : undefined,
                createdAt: new Date().toISOString()
              };
              
              // Update state
              setSettings(prev => ({
                ...prev,
                videos: [...prev.videos, newSample],
                selectedVideoId: newSample.id
              }));
              
              // Clean up and resolve
              setVideoChunks([]);
              resolve(newSample);
            } else {
              reject('Failed to generate thumbnail');
            }
          };
        } catch (err) {
          console.error(err);
          reject('Error processing video');
        } finally {
          // Stop all tracks
          videoStream.getTracks().forEach(track => track.stop());
          setVideoStream(null);
          setVideoRecorder(null);
        }
      };
      
      // Stop the recording
      videoRecorder.stop();
      setIsRecording(false);
    });
  }, [videoRecorder, videoStream, videoChunks]);
  
  const deleteVideo = useCallback((id: string) => {
    setSettings(prev => {
      // Filter out the deleted video
      const updatedVideos = prev.videos.filter(video => video.id !== id);
      
      // Update selected video if necessary
      let updatedSelectedId = prev.selectedVideoId;
      if (prev.selectedVideoId === id) {
        updatedSelectedId = updatedVideos.length > 0 ? updatedVideos[0].id : null;
      }
      
      return {
        ...prev,
        videos: updatedVideos,
        selectedVideoId: updatedSelectedId
      };
    });
  }, []);
  
  const selectVideo = useCallback((id: string) => {
    setSettings(prev => ({
      ...prev,
      selectedVideoId: id
    }));
  }, []);
  
  // Audio recording functions
  const startAudioRecording = useCallback(async () => {
    try {
      setError(null);
      
      // Request mic access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false
      });
      
      setAudioStream(stream);
      
      // Initialize recorder
      const recorder = new MediaRecorder(stream);
      setAudioRecorder(recorder);
      
      // Set up recorder events
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          setAudioChunks(prev => [...prev, e.data]);
        }
      };
      
      // Start recording
      setAudioChunks([]);
      recorder.start(1000); // Collect data every second
      setIsRecording(true);
    } catch (err) {
      setError('Could not access microphone. Please ensure you have granted microphone permission.');
      console.error(err);
    }
  }, []);
  
  const stopAudioRecording = useCallback(async (): Promise<CreationSampleAudio> => {
    return new Promise((resolve, reject) => {
      if (!audioRecorder || !audioStream) {
        setError('No active recording found');
        reject('No active recording found');
        return;
      }
      
      // Set up handler for when recording stops
      audioRecorder.onstop = async () => {
        try {
          // Create audio blob from chunks
          const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
          const audioUrl = URL.createObjectURL(audioBlob);
          
          // Create audio element to get duration
          const audio = document.createElement('audio');
          audio.src = audioUrl;
          
          audio.onloadedmetadata = async () => {
            // In a real app, we would analyze the audio and generate a waveform
            // Here we're just generating random waveform data
            const waveform = Array.from({ length: 100 }, () => Math.random() * 100);
            
            // In a real app, this would be a complex quality analysis
            // Here we're just generating a random quality score
            const quality = Math.floor(Math.random() * 20) + 80; // 80-99
            
            // Generate sample issues based on quality
            const issues = [];
            if (quality < 85) {
              issues.push({
                type: 'noise' as const,
                severity: 'low' as const,
                message: 'Some background noise detected'
              });
            }
            if (quality < 90) {
              issues.push({
                type: 'volume' as const,
                severity: 'low' as const,
                message: 'Audio volume could be higher'
              });
            }
            
            // Create sample object
            const newSample: CreationSampleAudio = {
              id: `audio-${Date.now()}`,
              blob: audioBlob,
              url: audioUrl,
              duration: audio.duration,
              quality,
              waveform,
              isPreferred: false,
              issues: issues.length > 0 ? issues : undefined,
              createdAt: new Date().toISOString()
            };
            
            // Update state
            setSettings(prev => ({
              ...prev,
              audios: [...prev.audios, newSample],
              selectedAudioId: newSample.id
            }));
            
            // Clean up and resolve
            setAudioChunks([]);
            resolve(newSample);
          };
        } catch (err) {
          console.error(err);
          reject('Error processing audio');
        } finally {
          // Stop all tracks
          audioStream.getTracks().forEach(track => track.stop());
          setAudioStream(null);
          setAudioRecorder(null);
        }
      };
      
      // Stop the recording
      audioRecorder.stop();
      setIsRecording(false);
    });
  }, [audioRecorder, audioStream, audioChunks]);
  
  const deleteAudio = useCallback((id: string) => {
    setSettings(prev => {
      // Filter out the deleted audio
      const updatedAudios = prev.audios.filter(audio => audio.id !== id);
      
      // Update selected audio if necessary
      let updatedSelectedId = prev.selectedAudioId;
      if (prev.selectedAudioId === id) {
        updatedSelectedId = updatedAudios.length > 0 ? updatedAudios[0].id : null;
      }
      
      return {
        ...prev,
        audios: updatedAudios,
        selectedAudioId: updatedSelectedId
      };
    });
  }, []);
  
  const selectAudio = useCallback((id: string) => {
    setSettings(prev => ({
      ...prev,
      selectedAudioId: id
    }));
  }, []);
  
  // Reference image functions
  const uploadReferenceImage = useCallback(async (file: File): Promise<CreationReferenceImage> => {
    try {
      setError(null);
      setIsProcessing(true);
      
      // Create URL for the file
      const imageUrl = URL.createObjectURL(file);
      
      // In a real app, this would be a complex quality analysis
      // Here we're just generating a random quality score
      const quality = Math.floor(Math.random() * 20) + 80; // 80-99
      
      // Create reference image object
      const newImage: CreationReferenceImage = {
        id: `image-${Date.now()}`,
        blob: file,
        url: imageUrl,
        quality,
        tags: ['reference', 'uploaded'],
        createdAt: new Date().toISOString()
      };
      
      // Update state
      setSettings(prev => ({
        ...prev,
        referenceImages: [...prev.referenceImages, newImage]
      }));
      
      return newImage;
    } catch (err) {
      console.error(err);
      setError('Error uploading image');
      throw err;
    } finally {
      setIsProcessing(false);
    }
  }, []);
  
  const deleteReferenceImage = useCallback((id: string) => {
    setSettings(prev => ({
      ...prev,
      referenceImages: prev.referenceImages.filter(image => image.id !== id)
    }));
  }, []);
  
  // Settings update functions
  const updateVoiceSettings = useCallback((newSettings: Partial<CreationSettings['voiceSettings']>) => {
    setSettings(prev => ({
      ...prev,
      voiceSettings: {
        ...prev.voiceSettings,
        ...newSettings
      }
    }));
  }, []);
  
  const updateAppearanceSettings = useCallback((newSettings: Partial<CreationSettings['appearanceSettings']>) => {
    setSettings(prev => ({
      ...prev,
      appearanceSettings: {
        ...prev.appearanceSettings,
        ...newSettings
      }
    }));
  }, []);
  
  const updateAnimationSettings = useCallback((newSettings: Partial<CreationSettings['animationSettings']>) => {
    setSettings(prev => ({
      ...prev,
      animationSettings: {
        ...prev.animationSettings,
        ...newSettings
      }
    }));
  }, []);
  
  const setBackground = useCallback((id: string) => {
    setSettings(prev => ({
      ...prev,
      backgroundId: id
    }));
  }, []);
  
  // Final actions
  const generatePreview = useCallback(async (): Promise<string> => {
    try {
      setError(null);
      setIsProcessing(true);
      setProgress(0);
      
      // In a real app, this would make an API call to generate a preview
      // Here we're just simulating the process
      
      // Check that we have the minimum required settings
      if (!settings.name || !settings.style || !settings.selectedVideoId) {
        throw new Error('Missing required settings for preview generation');
      }
      
      // Simulate progress updates
      const updateInterval = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + Math.random() * 10;
          return newProgress >= 100 ? 99 : newProgress;
        });
      }, 500);
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      clearInterval(updateInterval);
      setProgress(100);
      
      // Create a fake preview URL (in a real app this would come from the API)
      const previewUrl = settings.style 
        ? settings.style.thumbnail.replace('styles', 'previews')
        : '/avatars/previews/default.mp4';
      
      setPreviewUrl(previewUrl);
      return previewUrl;
    } catch (err) {
      console.error(err);
      setError('Error generating preview');
      throw err;
    } finally {
      setIsProcessing(false);
    }
  }, [settings]);
  
  const saveAvatar = useCallback(async (): Promise<string> => {
    try {
      setError(null);
      setIsProcessing(true);
      setProgress(0);
      
      // In a real app, this would make an API call to finalize the avatar
      // Here we're using our avatar context to create a new avatar
      
      // Check that we have the minimum required settings
      if (!settings.name || !settings.style || !settings.selectedVideoId) {
        throw new Error('Missing required settings for avatar creation');
      }
      
      // Simulate progress updates
      const updateInterval = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + Math.random() * 10;
          return newProgress >= 100 ? 99 : newProgress;
        });
      }, 300);
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Create the avatar using our context
      const newAvatar = await createAvatar(settings.name, settings.style);
      
      clearInterval(updateInterval);
      setProgress(100);
      
      // Reset state and return the new avatar ID
      setCurrentStep('complete');
      return newAvatar.id;
    } catch (err) {
      console.error(err);
      setError('Error saving avatar');
      throw err;
    } finally {
      setIsProcessing(false);
    }
  }, [settings, createAvatar]);
  
  const resetCreation = useCallback(() => {
    // Stop any active streams
    if (videoStream) {
      videoStream.getTracks().forEach(track => track.stop());
    }
    if (audioStream) {
      audioStream.getTracks().forEach(track => track.stop());
    }
    
    // Reset all state
    setCurrentStep('intro');
    setSettings({
      name: '',
      style: null,
      videos: [],
      audios: [],
      referenceImages: [],
      selectedVideoId: null,
      selectedAudioId: null,
      voiceSettings: DEFAULT_VOICE_SETTINGS,
      appearanceSettings: DEFAULT_APPEARANCE_SETTINGS,
      animationSettings: DEFAULT_ANIMATION_SETTINGS,
      backgroundId: null,
      appearance: DEFAULT_APPEARANCE
    });
    setIsRecording(false);
    setIsProcessing(false);
    setProgress(0);
    setPreviewUrl(null);
    setError(null);
    setVideoStream(null);
    setAudioStream(null);
    setVideoRecorder(null);
    setAudioRecorder(null);
    setVideoChunks([]);
    setAudioChunks([]);
  }, [videoStream, audioStream]);
  
  const deleteSampleVideo = useCallback((id: string) => {
    setSettings(prev => ({
      ...prev,
      videos: prev.videos.filter(video => video.id !== id),
      selectedVideoId: prev.selectedVideoId === id ? null : prev.selectedVideoId
    }));
  }, []);

  const deleteSampleAudio = useCallback((id: string) => {
    setSettings(prev => ({
      ...prev,
      audios: prev.audios.filter(audio => audio.id !== id),
      selectedAudioId: prev.selectedAudioId === id ? null : prev.selectedAudioId
    }));
  }, []);

  const markSampleAsPreferred = useCallback((type: 'video' | 'audio', id: string) => {
    setSettings(prev => {
      if (type === 'video') {
        const videoIndex = prev.videos.findIndex(v => v.id === id);
        if (videoIndex !== -1) {
          const updatedVideos = prev.videos.map(v => ({
            ...v,
            isPreferred: v.id === id
          }));
          return {
            ...prev,
            videos: updatedVideos,
            selectedVideoId: id
          };
        }
      } else if (type === 'audio') {
        const audioIndex = prev.audios.findIndex(a => a.id === id);
        if (audioIndex !== -1) {
          const updatedAudios = prev.audios.map(a => ({
            ...a,
            isPreferred: a.id === id
          }));
          return {
            ...prev,
            audios: updatedAudios,
            selectedAudioId: id
          };
        }
      }
      return prev;
    });
  }, []);

  const updateSettings = useCallback((newSettings: CreationSettings) => {
    setSettings(newSettings);
  }, []);

  const generateAppearancePreview = useCallback(async (appearance: CreationSettings['appearance']) => {
    try {
      setIsProcessing(true);
      setProgress(0);
      
      // Convert appearance settings to the format expected by the API
      const appearanceSettings = {
        skinTone: parseInt(appearance.skinTone),
        hairStyle: appearance.hairStyle,
        hairColor: appearance.hairColor,
        eyeColor: appearance.eyeColor,
        facialFeatures: appearance.facialFeatures
      };
      
      // Simulate preview generation with progress updates
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
        setProgress(i);
      }
      
      // Here you would typically call your API to generate the preview
      const previewUrl = '/images/avatar-preview.png'; // Replace with actual API call
      setPreviewUrl(previewUrl);
      
      setProgress(100);
    } catch (error) {
      setError('Failed to generate appearance preview');
      console.error('Error generating appearance preview:', error);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const generateStylePreview = useCallback(async (style: AvatarStyle) => {
    try {
      setIsProcessing(true);
      setProgress(0);
      
      // Simulate preview generation with progress updates
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 200));
        setProgress(i);
      }
      
      // Here you would typically call your API to generate the preview
      const previewUrl = style.thumbnail; // Replace with actual API call
      setPreviewUrl(previewUrl);
      
      setProgress(100);
    } catch (error) {
      setError('Failed to generate style preview');
      console.error('Error generating style preview:', error);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const value = {
    currentStep,
    settings,
    isRecording,
    isProcessing,
    progress,
    previewUrl,
    error,
    videoStream,
    audioStream,
    
    goToStep,
    goToNextStep,
    goToPreviousStep,
    
    updateName,
    setSelectedStyle,
    
    startVideoRecording,
    stopVideoRecording,
    deleteVideo,
    selectVideo,
    deleteSampleVideo,
    
    startAudioRecording,
    stopAudioRecording,
    deleteAudio,
    selectAudio,
    deleteSampleAudio,
    
    uploadReferenceImage,
    deleteReferenceImage,
    
    updateVoiceSettings,
    updateAppearanceSettings,
    updateAnimationSettings,
    setBackground,
    
    markSampleAsPreferred,
    updateSettings,
    generateAppearancePreview,
    generateStylePreview,
    
    generatePreview,
    saveAvatar,
    resetCreation
  };
  
  return (
    <CreationContext.Provider value={value}>
      {children}
    </CreationContext.Provider>
  );
}; 