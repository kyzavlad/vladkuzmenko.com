'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface Language {
  id: string;
  name: string;
  nativeName: string;
  flag: string;
  dialectOptions?: {
    id: string;
    name: string;
    region: string;
  }[];
  voiceOptions?: {
    id: string;
    name: string;
    gender: 'male' | 'female' | 'neutral';
    sampleUrl: string;
  }[];
}

export interface TranslationTerminology {
  id: string;
  name: string;
  terms: {
    source: string;
    target: string;
  }[];
}

export interface TranslationSegment {
  id: string;
  startTime: number;
  endTime: number;
  sourceText: string;
  translatedText: string;
  confidence: number;
  lipSyncScore: number;
  isApproved: boolean;
  needsReview: boolean;
  alternatives?: string[];
}

export interface TranslationJob {
  id: string;
  videoId: string;
  videoUrl: string;
  videoDuration: number;
  thumbnailUrl: string;
  sourceLanguage: string;
  targetLanguages: string[];
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  createdAt: string;
  updatedAt: string;
  customizationOptions: {
    preserveNames: boolean;
    preserveFormatting: boolean;
    preserveTechnicalTerms: boolean;
    formality: 'formal' | 'neutral' | 'informal';
    keepOriginalVoice: boolean;
    useTranslationMemory: boolean;
  };
  results?: {
    [languageId: string]: {
      translatedVideoUrl: string;
      subtitlesUrl: string;
      segments: TranslationSegment[];
      processingTime: number;
      quality: number;
    }
  };
  terminologies: string[];
}

export interface TranslationMemory {
  id: string;
  name: string;
  sourceLanguage: string;
  targetLanguage: string;
  entries: {
    source: string;
    target: string;
    lastUsed: string;
    frequency: number;
  }[];
  createdAt: string;
  updatedAt: string;
}

interface TranslationContextProps {
  // State
  availableLanguages: Language[];
  detectedLanguage: string | null;
  selectedSourceLanguage: string | null;
  selectedTargetLanguages: string[];
  currentJob: TranslationJob | null;
  translationJobs: TranslationJob[];
  terminologies: TranslationTerminology[];
  translationMemories: TranslationMemory[];
  selectedTerminologies: string[];
  isProcessing: boolean;
  isDetectingLanguage: boolean;
  currentEditingSegment: string | null;
  error: string | null;
  
  // Language management
  setSourceLanguage: (languageId: string) => void;
  detectSourceLanguage: (videoId: string) => Promise<string>;
  addTargetLanguage: (languageId: string) => void;
  removeTargetLanguage: (languageId: string) => void;
  clearTargetLanguages: () => void;
  
  // Translation job management
  createTranslationJob: (videoId: string, options?: Partial<TranslationJob['customizationOptions']>) => Promise<TranslationJob>;
  fetchTranslationJob: (jobId: string) => Promise<TranslationJob>;
  cancelTranslationJob: (jobId: string) => Promise<void>;
  deleteTranslationJob: (jobId: string) => Promise<void>;
  fetchTranslationJobs: () => Promise<TranslationJob[]>;
  setCurrentJob: (job: TranslationJob | null) => void;
  
  // Terminology and translation memory
  addTerminology: (terminology: Omit<TranslationTerminology, 'id'>) => Promise<TranslationTerminology>;
  updateTerminology: (id: string, updates: Partial<Omit<TranslationTerminology, 'id'>>) => Promise<TranslationTerminology>;
  deleteTerminology: (id: string) => Promise<void>;
  toggleTerminology: (id: string) => void;
  
  // Segment editing
  updateSegment: (languageId: string, segmentId: string, text: string) => void;
  approveSegment: (languageId: string, segmentId: string, approved: boolean) => void;
  markSegmentForReview: (languageId: string, segmentId: string, needsReview: boolean) => void;
  setCurrentEditingSegment: (segmentId: string | null) => void;
  
  // Preview generation
  generatePreview: (jobId: string, languageId: string, segmentId?: string) => Promise<string>;
  
  // Export functions
  exportTranslatedVideo: (jobId: string, languageId: string, format?: 'mp4' | 'mov' | 'webm') => Promise<string>;
  exportSubtitles: (jobId: string, languageId: string, format?: 'srt' | 'vtt' | 'txt') => Promise<string>;
}

const TranslationContext = createContext<TranslationContextProps | undefined>(undefined);

export const useTranslationContext = () => {
  const context = useContext(TranslationContext);
  if (!context) {
    throw new Error('useTranslationContext must be used within a TranslationProvider');
  }
  return context;
};

export const TranslationProvider = ({ children }: { children: ReactNode }) => {
  // States
  const [availableLanguages, setAvailableLanguages] = useState<Language[]>([
    {
      id: 'en',
      name: 'English',
      nativeName: 'English',
      flag: '🇺🇸',
      dialectOptions: [
        { id: 'en-us', name: 'American', region: 'United States' },
        { id: 'en-gb', name: 'British', region: 'United Kingdom' },
        { id: 'en-au', name: 'Australian', region: 'Australia' },
        { id: 'en-ca', name: 'Canadian', region: 'Canada' },
      ],
      voiceOptions: [
        { id: 'en-us-male-1', name: 'Michael', gender: 'male', sampleUrl: '/samples/en-us-male-1.mp3' },
        { id: 'en-us-female-1', name: 'Sarah', gender: 'female', sampleUrl: '/samples/en-us-female-1.mp3' },
        { id: 'en-gb-male-1', name: 'William', gender: 'male', sampleUrl: '/samples/en-gb-male-1.mp3' },
        { id: 'en-gb-female-1', name: 'Emma', gender: 'female', sampleUrl: '/samples/en-gb-female-1.mp3' },
      ],
    },
    {
      id: 'es',
      name: 'Spanish',
      nativeName: 'Español',
      flag: '🇪🇸',
      dialectOptions: [
        { id: 'es-es', name: 'Castilian', region: 'Spain' },
        { id: 'es-mx', name: 'Mexican', region: 'Mexico' },
        { id: 'es-ar', name: 'Argentinian', region: 'Argentina' },
      ],
      voiceOptions: [
        { id: 'es-es-male-1', name: 'Carlos', gender: 'male', sampleUrl: '/samples/es-es-male-1.mp3' },
        { id: 'es-es-female-1', name: 'Sofia', gender: 'female', sampleUrl: '/samples/es-es-female-1.mp3' },
        { id: 'es-mx-male-1', name: 'Miguel', gender: 'male', sampleUrl: '/samples/es-mx-male-1.mp3' },
      ],
    },
    {
      id: 'fr',
      name: 'French',
      nativeName: 'Français',
      flag: '🇫🇷',
      dialectOptions: [
        { id: 'fr-fr', name: 'Metropolitan', region: 'France' },
        { id: 'fr-ca', name: 'Canadian', region: 'Canada' },
        { id: 'fr-be', name: 'Belgian', region: 'Belgium' },
      ],
      voiceOptions: [
        { id: 'fr-fr-male-1', name: 'Pierre', gender: 'male', sampleUrl: '/samples/fr-fr-male-1.mp3' },
        { id: 'fr-fr-female-1', name: 'Marie', gender: 'female', sampleUrl: '/samples/fr-fr-female-1.mp3' },
      ],
    },
    {
      id: 'de',
      name: 'German',
      nativeName: 'Deutsch',
      flag: '🇩🇪',
      dialectOptions: [
        { id: 'de-de', name: 'Standard', region: 'Germany' },
        { id: 'de-at', name: 'Austrian', region: 'Austria' },
        { id: 'de-ch', name: 'Swiss', region: 'Switzerland' },
      ],
      voiceOptions: [
        { id: 'de-de-male-1', name: 'Hans', gender: 'male', sampleUrl: '/samples/de-de-male-1.mp3' },
        { id: 'de-de-female-1', name: 'Anna', gender: 'female', sampleUrl: '/samples/de-de-female-1.mp3' },
      ],
    },
    {
      id: 'ja',
      name: 'Japanese',
      nativeName: '日本語',
      flag: '🇯🇵',
      voiceOptions: [
        { id: 'ja-jp-male-1', name: 'Hiro', gender: 'male', sampleUrl: '/samples/ja-jp-male-1.mp3' },
        { id: 'ja-jp-female-1', name: 'Yuki', gender: 'female', sampleUrl: '/samples/ja-jp-female-1.mp3' },
      ],
    },
    {
      id: 'zh',
      name: 'Chinese',
      nativeName: '中文',
      flag: '🇨🇳',
      dialectOptions: [
        { id: 'zh-cn', name: 'Mandarin (Simplified)', region: 'China' },
        { id: 'zh-tw', name: 'Mandarin (Traditional)', region: 'Taiwan' },
        { id: 'zh-hk', name: 'Cantonese', region: 'Hong Kong' },
      ],
      voiceOptions: [
        { id: 'zh-cn-male-1', name: 'Li Wei', gender: 'male', sampleUrl: '/samples/zh-cn-male-1.mp3' },
        { id: 'zh-cn-female-1', name: 'Wang Fang', gender: 'female', sampleUrl: '/samples/zh-cn-female-1.mp3' },
      ],
    },
  ]);
  
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null);
  const [selectedSourceLanguage, setSelectedSourceLanguage] = useState<string | null>(null);
  const [selectedTargetLanguages, setSelectedTargetLanguages] = useState<string[]>([]);
  const [currentJob, setCurrentJob] = useState<TranslationJob | null>(null);
  const [translationJobs, setTranslationJobs] = useState<TranslationJob[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDetectingLanguage, setIsDetectingLanguage] = useState(false);
  const [currentEditingSegment, setCurrentEditingSegment] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [terminologies, setTerminologies] = useState<TranslationTerminology[]>([
    {
      id: '1',
      name: 'Technical Terms',
      terms: [
        { source: 'artificial intelligence', target: 'artificial intelligence' },
        { source: 'machine learning', target: 'machine learning' },
        { source: 'neural network', target: 'neural network' },
      ],
    },
    {
      id: '2',
      name: 'Brand Names',
      terms: [
        { source: 'AI Video Platform', target: 'AI Video Platform' },
        { source: 'Avatar Creator', target: 'Avatar Creator' },
      ],
    },
  ]);
  
  const [selectedTerminologies, setSelectedTerminologies] = useState<string[]>([]);
  
  const [translationMemories, setTranslationMemories] = useState<TranslationMemory[]>([
    {
      id: '1',
      name: 'General Translations',
      sourceLanguage: 'en',
      targetLanguage: 'es',
      entries: [
        {
          source: 'Welcome to our platform',
          target: 'Bienvenido a nuestra plataforma',
          lastUsed: new Date().toISOString(),
          frequency: 5,
        },
        {
          source: 'Please subscribe to our channel',
          target: 'Por favor suscríbase a nuestro canal',
          lastUsed: new Date().toISOString(),
          frequency: 3,
        },
      ],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
  ]);
  
  // Mock implementations for API interactions
  const detectSourceLanguage = async (videoId: string): Promise<string> => {
    setIsDetectingLanguage(true);
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      await new Promise(resolve => setTimeout(resolve, 2000));
      const detectedLang = 'en'; // Mock detected language
      setDetectedLanguage(detectedLang);
      setSelectedSourceLanguage(detectedLang);
      return detectedLang;
    } catch (err) {
      setError('Failed to detect language');
      console.error('Error detecting language:', err);
      throw err;
    } finally {
      setIsDetectingLanguage(false);
    }
  };
  
  const setSourceLanguage = (languageId: string) => {
    setSelectedSourceLanguage(languageId);
  };
  
  const addTargetLanguage = (languageId: string) => {
    if (!selectedTargetLanguages.includes(languageId)) {
      setSelectedTargetLanguages([...selectedTargetLanguages, languageId]);
    }
  };
  
  const removeTargetLanguage = (languageId: string) => {
    setSelectedTargetLanguages(selectedTargetLanguages.filter(id => id !== languageId));
  };
  
  const clearTargetLanguages = () => {
    setSelectedTargetLanguages([]);
  };
  
  const createTranslationJob = async (
    videoId: string, 
    options?: Partial<TranslationJob['customizationOptions']>
  ): Promise<TranslationJob> => {
    setIsProcessing(true);
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      const defaultOptions: TranslationJob['customizationOptions'] = {
        preserveNames: true,
        preserveFormatting: true,
        preserveTechnicalTerms: true,
        formality: 'neutral',
        keepOriginalVoice: false,
        useTranslationMemory: true,
      };
      
      const newJob: TranslationJob = {
        id: `job-${Date.now()}`,
        videoId,
        videoUrl: `/videos/${videoId}`,
        videoDuration: 120, // Mock duration
        thumbnailUrl: `/thumbnails/${videoId}.jpg`,
        sourceLanguage: selectedSourceLanguage || 'en',
        targetLanguages: selectedTargetLanguages,
        status: 'pending',
        progress: 0,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        customizationOptions: {
          ...defaultOptions,
          ...options,
        },
        terminologies: selectedTerminologies,
      };
      
      setTranslationJobs(prev => [newJob, ...prev]);
      setCurrentJob(newJob);
      
      // Simulate job progression
      const progressInterval = setInterval(() => {
        setTranslationJobs(prev => 
          prev.map(job => {
            if (job.id === newJob.id) {
              const newProgress = Math.min(job.progress + 5, 100);
              const newStatus = newProgress === 100 ? 'completed' : 'processing';
              
              // Add mock results when completed
              let jobUpdate: Partial<TranslationJob> = {
                progress: newProgress,
                status: newStatus,
                updatedAt: new Date().toISOString(),
              };
              
              if (newStatus === 'completed') {
                clearInterval(progressInterval);
                
                const mockResults: TranslationJob['results'] = {};
                job.targetLanguages.forEach(langId => {
                  mockResults[langId] = {
                    translatedVideoUrl: `/videos/translated/${job.id}/${langId}`,
                    subtitlesUrl: `/subtitles/${job.id}/${langId}.vtt`,
                    segments: Array.from({ length: 10 }, (_, i) => ({
                      id: `segment-${i}`,
                      startTime: i * 10,
                      endTime: (i + 1) * 10 - 0.5,
                      sourceText: `This is source text segment ${i + 1}`,
                      translatedText: `This is translated text segment ${i + 1} in ${langId}`,
                      confidence: Math.random() * 40 + 60, // 60-100
                      lipSyncScore: Math.random() * 40 + 60, // 60-100
                      isApproved: false,
                      needsReview: Math.random() > 0.7,
                      alternatives: [
                        `Alternative ${i}.1 for ${langId}`,
                        `Alternative ${i}.2 for ${langId}`,
                      ],
                    })),
                    processingTime: Math.floor(Math.random() * 300) + 60, // 60-360 seconds
                    quality: Math.floor(Math.random() * 20) + 80, // 80-100
                  };
                });
                
                jobUpdate.results = mockResults;
              }
              
              return { ...job, ...jobUpdate };
            }
            return job;
          })
        );
        
        // Update current job if it's the active one
        setCurrentJob(prev => {
          if (prev?.id === newJob.id) {
            return translationJobs.find(j => j.id === newJob.id) || prev;
          }
          return prev;
        });
        
      }, 1000);
      
      return newJob;
    } catch (err) {
      setError('Failed to create translation job');
      console.error('Error creating translation job:', err);
      throw err;
    } finally {
      setIsProcessing(false);
    }
  };
  
  const fetchTranslationJob = async (jobId: string): Promise<TranslationJob> => {
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      const job = translationJobs.find(j => j.id === jobId);
      
      if (!job) {
        throw new Error('Translation job not found');
      }
      
      setCurrentJob(job);
      return job;
    } catch (err) {
      setError('Failed to fetch translation job');
      console.error('Error fetching translation job:', err);
      throw err;
    }
  };
  
  const cancelTranslationJob = async (jobId: string): Promise<void> => {
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      setTranslationJobs(prev => 
        prev.map(job => 
          job.id === jobId 
            ? { ...job, status: 'failed', updatedAt: new Date().toISOString() } 
            : job
        )
      );
      
      // Update current job if it's the active one
      setCurrentJob(prev => {
        if (prev?.id === jobId) {
          return { ...prev, status: 'failed', updatedAt: new Date().toISOString() };
        }
        return prev;
      });
    } catch (err) {
      setError('Failed to cancel translation job');
      console.error('Error canceling translation job:', err);
      throw err;
    }
  };
  
  const deleteTranslationJob = async (jobId: string): Promise<void> => {
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      setTranslationJobs(prev => prev.filter(job => job.id !== jobId));
      
      // If the current job is deleted, clear it
      if (currentJob?.id === jobId) {
        setCurrentJob(null);
      }
    } catch (err) {
      setError('Failed to delete translation job');
      console.error('Error deleting translation job:', err);
      throw err;
    }
  };
  
  const fetchTranslationJobs = async (): Promise<TranslationJob[]> => {
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      return translationJobs;
    } catch (err) {
      setError('Failed to fetch translation jobs');
      console.error('Error fetching translation jobs:', err);
      throw err;
    }
  };
  
  const addTerminology = async (terminology: Omit<TranslationTerminology, 'id'>): Promise<TranslationTerminology> => {
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      const newTerminology: TranslationTerminology = {
        ...terminology,
        id: `term-${Date.now()}`,
      };
      
      setTerminologies(prev => [...prev, newTerminology]);
      return newTerminology;
    } catch (err) {
      setError('Failed to add terminology');
      console.error('Error adding terminology:', err);
      throw err;
    }
  };
  
  const updateTerminology = async (
    id: string, 
    updates: Partial<Omit<TranslationTerminology, 'id'>>
  ): Promise<TranslationTerminology> => {
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      const terminology = terminologies.find(t => t.id === id);
      
      if (!terminology) {
        throw new Error('Terminology not found');
      }
      
      const updatedTerminology = { ...terminology, ...updates };
      
      setTerminologies(prev => 
        prev.map(t => t.id === id ? updatedTerminology : t)
      );
      
      return updatedTerminology;
    } catch (err) {
      setError('Failed to update terminology');
      console.error('Error updating terminology:', err);
      throw err;
    }
  };
  
  const deleteTerminology = async (id: string): Promise<void> => {
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      setTerminologies(prev => prev.filter(t => t.id !== id));
      setSelectedTerminologies(prev => prev.filter(tId => tId !== id));
    } catch (err) {
      setError('Failed to delete terminology');
      console.error('Error deleting terminology:', err);
      throw err;
    }
  };
  
  const toggleTerminology = (id: string) => {
    setSelectedTerminologies(prev => 
      prev.includes(id) 
        ? prev.filter(tId => tId !== id) 
        : [...prev, id]
    );
  };
  
  const updateSegment = (languageId: string, segmentId: string, text: string) => {
    if (!currentJob || !currentJob.results?.[languageId]) return;
    
    setCurrentJob(prev => {
      if (!prev || !prev.results?.[languageId]) return prev;
      
      const updatedResults = { ...prev.results };
      updatedResults[languageId] = {
        ...updatedResults[languageId],
        segments: updatedResults[languageId].segments.map(segment => 
          segment.id === segmentId 
            ? { ...segment, translatedText: text, isApproved: false } 
            : segment
        ),
      };
      
      return { ...prev, results: updatedResults, updatedAt: new Date().toISOString() };
    });
  };
  
  const approveSegment = (languageId: string, segmentId: string, approved: boolean) => {
    if (!currentJob || !currentJob.results?.[languageId]) return;
    
    setCurrentJob(prev => {
      if (!prev || !prev.results?.[languageId]) return prev;
      
      const updatedResults = { ...prev.results };
      updatedResults[languageId] = {
        ...updatedResults[languageId],
        segments: updatedResults[languageId].segments.map(segment => 
          segment.id === segmentId 
            ? { ...segment, isApproved: approved, needsReview: false } 
            : segment
        ),
      };
      
      return { ...prev, results: updatedResults, updatedAt: new Date().toISOString() };
    });
  };
  
  const markSegmentForReview = (languageId: string, segmentId: string, needsReview: boolean) => {
    if (!currentJob || !currentJob.results?.[languageId]) return;
    
    setCurrentJob(prev => {
      if (!prev || !prev.results?.[languageId]) return prev;
      
      const updatedResults = { ...prev.results };
      updatedResults[languageId] = {
        ...updatedResults[languageId],
        segments: updatedResults[languageId].segments.map(segment => 
          segment.id === segmentId 
            ? { ...segment, needsReview } 
            : segment
        ),
      };
      
      return { ...prev, results: updatedResults, updatedAt: new Date().toISOString() };
    });
  };
  
  const generatePreview = async (jobId: string, languageId: string, segmentId?: string): Promise<string> => {
    setIsProcessing(true);
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Return a mock URL for the preview
      const url = segmentId 
        ? `/previews/${jobId}/${languageId}/segments/${segmentId}` 
        : `/previews/${jobId}/${languageId}/full`;
      
      return url;
    } catch (err) {
      setError('Failed to generate preview');
      console.error('Error generating preview:', err);
      throw err;
    } finally {
      setIsProcessing(false);
    }
  };
  
  const exportTranslatedVideo = async (
    jobId: string, 
    languageId: string, 
    format: 'mp4' | 'mov' | 'webm' = 'mp4'
  ): Promise<string> => {
    setIsProcessing(true);
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Return a mock URL for the exported video
      const url = `/exports/${jobId}/${languageId}/video.${format}`;
      
      return url;
    } catch (err) {
      setError('Failed to export translated video');
      console.error('Error exporting translated video:', err);
      throw err;
    } finally {
      setIsProcessing(false);
    }
  };
  
  const exportSubtitles = async (
    jobId: string, 
    languageId: string, 
    format: 'srt' | 'vtt' | 'txt' = 'vtt'
  ): Promise<string> => {
    setIsProcessing(true);
    setError(null);
    
    try {
      // This would be an API call in a real implementation
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Return a mock URL for the exported subtitles
      const url = `/exports/${jobId}/${languageId}/subtitles.${format}`;
      
      return url;
    } catch (err) {
      setError('Failed to export subtitles');
      console.error('Error exporting subtitles:', err);
      throw err;
    } finally {
      setIsProcessing(false);
    }
  };
  
  // Provide all the values and functions to the context
  const contextValue: TranslationContextProps = {
    availableLanguages,
    detectedLanguage,
    selectedSourceLanguage,
    selectedTargetLanguages,
    currentJob,
    translationJobs,
    terminologies,
    translationMemories,
    selectedTerminologies,
    isProcessing,
    isDetectingLanguage,
    currentEditingSegment,
    error,
    
    setSourceLanguage,
    detectSourceLanguage,
    addTargetLanguage,
    removeTargetLanguage,
    clearTargetLanguages,
    
    createTranslationJob,
    fetchTranslationJob,
    cancelTranslationJob,
    deleteTranslationJob,
    fetchTranslationJobs,
    setCurrentJob,
    
    addTerminology,
    updateTerminology,
    deleteTerminology,
    toggleTerminology,
    
    updateSegment,
    approveSegment,
    markSegmentForReview,
    setCurrentEditingSegment,
    
    generatePreview,
    
    exportTranslatedVideo,
    exportSubtitles,
  };
  
  return (
    <TranslationContext.Provider value={contextValue}>
      {children}
    </TranslationContext.Provider>
  );
}; 