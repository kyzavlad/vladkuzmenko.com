'use client';

import React, { createContext, useContext, useState, ReactNode, useCallback } from 'react';

export interface MediaFile {
  id: string;
  name: string;
  type: string;
  size: number;
  duration?: number;
  width?: number;
  height?: number;
  url: string;
  thumbnail?: string;
  createdAt: string;
  updatedAt: string;
  status: 'processing' | 'ready' | 'error';
  processingProgress?: number;
  isSelected?: boolean;
  metadata?: Record<string, any>;
  tags?: string[];
  description?: string;
}

interface MediaContextProps {
  mediaFiles: MediaFile[];
  selectedFiles: MediaFile[];
  isLoading: boolean;
  error: string | null;
  sortBy: 'name' | 'date' | 'size' | 'duration';
  sortDirection: 'asc' | 'desc';
  viewMode: 'grid' | 'list';
  filterOptions: {
    status: ('processing' | 'ready' | 'error')[];
    type: string[];
    tags: string[];
  };
  searchQuery: string;
  currentPage: number;
  totalPages: number;
  fetchMediaFiles: () => Promise<void>;
  selectFile: (id: string) => void;
  deselectFile: (id: string) => void;
  selectAllFiles: () => void;
  deselectAllFiles: () => void;
  deleteFiles: (ids: string[]) => Promise<void>;
  updateFile: (id: string, data: Partial<MediaFile>) => Promise<void>;
  setSortBy: (sortBy: 'name' | 'date' | 'size' | 'duration') => void;
  setSortDirection: (direction: 'asc' | 'desc') => void;
  setViewMode: (mode: 'grid' | 'list') => void;
  setFilterOptions: (options: Partial<MediaContextProps['filterOptions']>) => void;
  setSearchQuery: (query: string) => void;
  setCurrentPage: (page: number) => void;
  uploadMedia: (files: File[]) => Promise<void>;
}

const MediaContext = createContext<MediaContextProps | undefined>(undefined);

export const useMediaContext = () => {
  const context = useContext(MediaContext);
  if (context === undefined) {
    throw new Error('useMediaContext must be used within a MediaProvider');
  }
  return context;
};

export const MediaProvider = ({ children }: { children: ReactNode }) => {
  const [mediaFiles, setMediaFiles] = useState<MediaFile[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<MediaFile[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'name' | 'date' | 'size' | 'duration'>('date');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [filterOptions, setFilterOptions] = useState<MediaContextProps['filterOptions']>({
    status: ['processing', 'ready', 'error'],
    type: [],
    tags: []
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  // Sample data for development
  const sampleMediaFiles: MediaFile[] = [
    {
      id: '1',
      name: 'Product Demo.mp4',
      type: 'video/mp4',
      size: 15728640, // 15MB
      duration: 120, // 2min
      width: 1920,
      height: 1080,
      url: 'https://storage.example.com/videos/product-demo.mp4',
      thumbnail: 'https://images.unsplash.com/photo-1560419015-7c427e8ae5ba?q=80&w=300&auto=format&fit=crop',
      createdAt: '2023-12-10T14:30:00Z',
      updatedAt: '2023-12-10T14:30:00Z',
      status: 'ready',
      tags: ['product', 'demo']
    },
    {
      id: '2',
      name: 'Interview Recording.mp4',
      type: 'video/mp4',
      size: 52428800, // 50MB
      duration: 600, // 10min
      width: 1280,
      height: 720,
      url: 'https://storage.example.com/videos/interview.mp4',
      thumbnail: 'https://images.unsplash.com/photo-1566140967404-b8b3932483f5?q=80&w=300&auto=format&fit=crop',
      createdAt: '2023-12-05T09:15:00Z',
      updatedAt: '2023-12-05T09:15:00Z',
      status: 'ready',
      tags: ['interview', 'corporate']
    },
    {
      id: '3',
      name: 'Marketing Campaign.mp4',
      type: 'video/mp4',
      size: 104857600, // 100MB
      duration: 180, // 3min
      width: 3840,
      height: 2160,
      url: 'https://storage.example.com/videos/marketing-campaign.mp4',
      thumbnail: 'https://images.unsplash.com/photo-1611162618071-b39a2ec055fb?q=80&w=300&auto=format&fit=crop',
      createdAt: '2023-12-01T11:45:00Z',
      updatedAt: '2023-12-01T11:45:00Z',
      status: 'ready',
      tags: ['marketing', '4k']
    },
    {
      id: '4',
      name: 'Tutorial.mp4',
      type: 'video/mp4',
      size: 31457280, // 30MB
      duration: 900, // 15min
      width: 1920,
      height: 1080,
      url: 'https://storage.example.com/videos/tutorial.mp4',
      thumbnail: 'https://images.unsplash.com/photo-1611162617474-5b21e879e113?q=80&w=300&auto=format&fit=crop',
      createdAt: '2023-11-28T16:20:00Z',
      updatedAt: '2023-11-28T16:20:00Z',
      status: 'ready',
      tags: ['tutorial', 'education']
    },
    {
      id: '5',
      name: 'Conference Presentation.mp4',
      type: 'video/mp4',
      size: 78643200, // 75MB
      duration: 1800, // 30min
      width: 1920,
      height: 1080,
      url: 'https://storage.example.com/videos/conference.mp4',
      thumbnail: 'https://images.unsplash.com/photo-1505373877841-8d25f7d46678?q=80&w=300&auto=format&fit=crop',
      createdAt: '2023-11-20T13:00:00Z',
      updatedAt: '2023-11-20T13:00:00Z',
      status: 'ready',
      tags: ['conference', 'presentation']
    },
    {
      id: '6',
      name: 'New Project.mp4',
      type: 'video/mp4',
      size: 10485760, // 10MB
      url: 'https://storage.example.com/videos/new-project.mp4',
      createdAt: '2023-12-15T10:30:00Z',
      updatedAt: '2023-12-15T10:30:00Z',
      status: 'processing',
      processingProgress: 45,
    }
  ];

  const fetchMediaFiles = useCallback(async () => {
    setIsLoading(true);
    try {
      // In a real app, this would be an API call
      // const response = await fetch('/api/media');
      // const data = await response.json();
      
      // Using sample data for now
      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate network delay
      setMediaFiles(sampleMediaFiles);
      setTotalPages(1);
      setError(null);
    } catch (error) {
      console.error('Error fetching media files:', error);
      setError('Failed to fetch media files');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const selectFile = useCallback((id: string) => {
    setMediaFiles(prev => 
      prev.map(file => 
        file.id === id ? { ...file, isSelected: true } : file
      )
    );
    
    const fileToSelect = mediaFiles.find(file => file.id === id);
    if (fileToSelect) {
      setSelectedFiles(prev => [...prev, fileToSelect]);
    }
  }, [mediaFiles]);

  const deselectFile = useCallback((id: string) => {
    setMediaFiles(prev => 
      prev.map(file => 
        file.id === id ? { ...file, isSelected: false } : file
      )
    );
    
    setSelectedFiles(prev => prev.filter(file => file.id !== id));
  }, []);

  const selectAllFiles = useCallback(() => {
    setMediaFiles(prev => 
      prev.map(file => ({ ...file, isSelected: true }))
    );
    setSelectedFiles(mediaFiles);
  }, [mediaFiles]);

  const deselectAllFiles = useCallback(() => {
    setMediaFiles(prev => 
      prev.map(file => ({ ...file, isSelected: false }))
    );
    setSelectedFiles([]);
  }, []);

  const deleteFiles = useCallback(async (ids: string[]) => {
    setIsLoading(true);
    try {
      // In a real app, this would be an API call
      // await Promise.all(ids.map(id => fetch(`/api/media/${id}`, { method: 'DELETE' })));
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setMediaFiles(prev => prev.filter(file => !ids.includes(file.id)));
      setSelectedFiles(prev => prev.filter(file => !ids.includes(file.id)));
      setError(null);
    } catch (error) {
      console.error('Error deleting files:', error);
      setError('Failed to delete files');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const updateFile = useCallback(async (id: string, data: Partial<MediaFile>) => {
    setIsLoading(true);
    try {
      // In a real app, this would be an API call
      // await fetch(`/api/media/${id}`, { 
      //   method: 'PATCH', 
      //   body: JSON.stringify(data),
      //   headers: { 'Content-Type': 'application/json' }
      // });
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 300));
      
      setMediaFiles(prev => 
        prev.map(file => 
          file.id === id ? { ...file, ...data } : file
        )
      );
      
      // Also update in selectedFiles if it exists there
      setSelectedFiles(prev =>
        prev.map(file =>
          file.id === id ? { ...file, ...data } : file
        )
      );
      
      setError(null);
    } catch (error) {
      console.error('Error updating file:', error);
      setError('Failed to update file');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleSetFilterOptions = useCallback((options: Partial<MediaContextProps['filterOptions']>) => {
    setFilterOptions(prev => ({ ...prev, ...options }));
    setCurrentPage(1); // Reset to first page when filters change
  }, []);

  const uploadMedia = useCallback(async (files: File[]) => {
    // Здесь будет логика загрузки файлов
    console.log('Uploading files:', files);
  }, []);

  const value = {
    mediaFiles,
    selectedFiles,
    isLoading,
    error,
    sortBy,
    sortDirection,
    viewMode,
    filterOptions,
    searchQuery,
    currentPage,
    totalPages,
    fetchMediaFiles,
    selectFile,
    deselectFile,
    selectAllFiles,
    deselectAllFiles,
    deleteFiles,
    updateFile,
    setSortBy,
    setSortDirection,
    setViewMode,
    setFilterOptions: handleSetFilterOptions,
    setSearchQuery,
    setCurrentPage,
    uploadMedia
  };

  return (
    <MediaContext.Provider value={value}>
      {children}
    </MediaContext.Provider>
  );
}; 