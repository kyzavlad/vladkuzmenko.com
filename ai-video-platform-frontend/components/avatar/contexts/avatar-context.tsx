'use client';

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export interface AvatarStyle {
  id: string;
  name: string;
  thumbnail: string;
  description: string;
}

export interface AvatarSample {
  id: string;
  type: 'video' | 'image' | 'audio';
  url: string;
  thumbnail?: string;
  duration?: number;
  createdAt: string;
  quality?: number; // 0-100 quality score
}

export interface Avatar {
  id: string;
  name: string;
  thumbnail: string;
  previewVideo?: string;
  style: AvatarStyle;
  samples: AvatarSample[];
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
  versions: {
    id: string;
    name: string;
    createdAt: string;
    thumbnail: string;
  }[];
  isPublic: boolean;
  usageStats: {
    totalVideos: number;
    totalDuration: number;
    lastUsed: string;
  };
  createdAt: string;
  updatedAt: string;
}

export interface AvatarContextProps {
  avatars: Avatar[];
  selectedAvatar: Avatar | null;
  error: string | null;
  isLoading: boolean;
  styles: AvatarStyle[];
  fetchAvatars: () => Promise<void>;
  getAvatar: (id: string) => Promise<Avatar>;
  createAvatar: (name: string, style: AvatarStyle) => Promise<Avatar>;
  updateAvatar: (id: string, data: Partial<Avatar>) => Promise<Avatar>;
  deleteAvatar: (id: string) => Promise<void>;
  selectAvatar: (avatar: Avatar | null) => void;
  duplicateAvatar: (id: string, newName: string) => Promise<Avatar>;
  addSample: (avatarId: string, sample: Omit<AvatarSample, 'id' | 'createdAt'>) => Promise<AvatarSample>;
  removeSample: (avatarId: string, sampleId: string) => Promise<void>;
  updateAvatarSettings: (
    avatarId: string, 
    settings: { 
      voice?: Partial<Avatar['voiceSettings']>,
      appearance?: Partial<Avatar['appearanceSettings']>,
      animation?: Partial<Avatar['animationSettings']>
    }
  ) => Promise<Avatar>;
  setAvatarPublic: (avatarId: string, isPublic: boolean) => Promise<void>;
  clearError: () => void;
}

const AvatarContext = createContext<AvatarContextProps>({} as AvatarContextProps);

export const useAvatarContext = () => useContext(AvatarContext);

// Sample avatar styles
const AVATAR_STYLES: AvatarStyle[] = [
  {
    id: 'realistic',
    name: 'Realistic',
    thumbnail: '/avatars/styles/realistic.jpg',
    description: 'Photorealistic avatar that closely resembles the original person'
  },
  {
    id: 'stylized',
    name: 'Stylized',
    thumbnail: '/avatars/styles/stylized.jpg',
    description: 'Modern, slightly stylized version with enhanced features'
  },
  {
    id: 'animated',
    name: 'Animated',
    thumbnail: '/avatars/styles/animated.jpg',
    description: '3D animated character style with more expressive features'
  },
  {
    id: 'professional',
    name: 'Professional',
    thumbnail: '/avatars/styles/professional.jpg',
    description: 'Business-ready avatar with professional styling and background'
  },
  {
    id: 'minimalist',
    name: 'Minimalist',
    thumbnail: '/avatars/styles/minimalist.jpg',
    description: 'Clean, simplified avatar with minimal details and neutral background'
  }
];

// Sample avatars for development
const SAMPLE_AVATARS: Avatar[] = [
  {
    id: 'avatar-1',
    name: 'Business Presenter',
    thumbnail: '/avatars/samples/avatar1.jpg',
    previewVideo: '/avatars/samples/avatar1.mp4',
    style: AVATAR_STYLES[3], // Professional
    samples: [
      {
        id: 'sample-1',
        type: 'video',
        url: '/avatars/samples/sample1.mp4',
        thumbnail: '/avatars/samples/sample1.jpg',
        duration: 15,
        createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        quality: 92
      }
    ],
    voiceSettings: {
      pitch: 50,
      speed: 50,
      clarity: 80,
      expressiveness: 60
    },
    appearanceSettings: {
      skinTone: 50,
      hairStyle: 'short',
      hairColor: '#3a3a3a',
      eyeColor: '#724b34',
      facialFeatures: 50
    },
    animationSettings: {
      gestureIntensity: 60,
      expressionIntensity: 50,
      movementStyle: 'professional'
    },
    versions: [
      {
        id: 'v1',
        name: 'Initial Version',
        createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        thumbnail: '/avatars/samples/avatar1-v1.jpg'
      }
    ],
    isPublic: false,
    usageStats: {
      totalVideos: 5,
      totalDuration: 300, // in seconds
      lastUsed: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
    },
    createdAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
  },
  {
    id: 'avatar-2',
    name: 'Creative Explainer',
    thumbnail: '/avatars/samples/avatar2.jpg',
    previewVideo: '/avatars/samples/avatar2.mp4',
    style: AVATAR_STYLES[1], // Stylized
    samples: [
      {
        id: 'sample-2-1',
        type: 'video',
        url: '/avatars/samples/sample2-1.mp4',
        thumbnail: '/avatars/samples/sample2-1.jpg',
        duration: 20,
        createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
        quality: 88
      },
      {
        id: 'sample-2-2',
        type: 'audio',
        url: '/avatars/samples/sample2-2.mp3',
        duration: 45,
        createdAt: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
        quality: 95
      }
    ],
    voiceSettings: {
      pitch: 45,
      speed: 60,
      clarity: 85,
      expressiveness: 75
    },
    appearanceSettings: {
      skinTone: 40,
      hairStyle: 'medium',
      hairColor: '#6a4f3c',
      eyeColor: '#2c8a99',
      facialFeatures: 55
    },
    animationSettings: {
      gestureIntensity: 75,
      expressionIntensity: 70,
      movementStyle: 'energetic'
    },
    versions: [
      {
        id: 'v1',
        name: 'Initial Version',
        createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
        thumbnail: '/avatars/samples/avatar2-v1.jpg'
      },
      {
        id: 'v2',
        name: 'More Expressive',
        createdAt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
        thumbnail: '/avatars/samples/avatar2-v2.jpg'
      }
    ],
    isPublic: true,
    usageStats: {
      totalVideos: 8,
      totalDuration: 720, // in seconds
      lastUsed: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString()
    },
    createdAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    updatedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString()
  }
];

export const AvatarProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [avatars, setAvatars] = useState<Avatar[]>([]);
  const [selectedAvatar, setSelectedAvatar] = useState<Avatar | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [styles, setStyles] = useState<AvatarStyle[]>([]);

  const fetchAvatars = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, we would fetch avatars from an API
      // const response = await fetch('/api/avatars');
      // const data = await response.json();
      // setAvatars(data);
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      setAvatars(SAMPLE_AVATARS);
    } catch (err) {
      setError('Failed to fetch avatars');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const getAvatar = useCallback(async (id: string): Promise<Avatar> => {
    // In a real app, we would fetch a specific avatar from an API
    // const response = await fetch(`/api/avatars/${id}`);
    // const data = await response.json();
    // return data;
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const avatar = avatars.find(a => a.id === id);
    if (!avatar) {
      throw new Error('Avatar not found');
    }
    
    return avatar;
  }, [avatars]);

  const createAvatar = useCallback(async (name: string, style: AvatarStyle): Promise<Avatar> => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, we would create an avatar via an API
      // const response = await fetch('/api/avatars', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ name, style })
      // });
      // const data = await response.json();
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 800));
      
      const newAvatar: Avatar = {
        id: `avatar-${Date.now()}`,
        name,
        thumbnail: style.thumbnail,
        style,
        samples: [],
        voiceSettings: {
          pitch: 50,
          speed: 50,
          clarity: 80,
          expressiveness: 60
        },
        appearanceSettings: {
          skinTone: 50,
          hairStyle: 'short',
          hairColor: '#3a3a3a',
          eyeColor: '#724b34',
          facialFeatures: 50
        },
        animationSettings: {
          gestureIntensity: 60,
          expressionIntensity: 50,
          movementStyle: 'natural'
        },
        versions: [],
        isPublic: false,
        usageStats: {
          totalVideos: 0,
          totalDuration: 0,
          lastUsed: new Date().toISOString()
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      
      setAvatars(prev => [...prev, newAvatar]);
      return newAvatar;
    } catch (err) {
      setError('Failed to create avatar');
      console.error(err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const updateAvatar = useCallback(async (id: string, data: Partial<Avatar>): Promise<Avatar> => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, we would update an avatar via an API
      // const response = await fetch(`/api/avatars/${id}`, {
      //   method: 'PATCH',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(data)
      // });
      // const updatedData = await response.json();
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      let updatedAvatar: Avatar | undefined;
      
      setAvatars(prev => prev.map(avatar => {
        if (avatar.id === id) {
          updatedAvatar = {
            ...avatar,
            ...data,
            updatedAt: new Date().toISOString()
          };
          return updatedAvatar;
        }
        return avatar;
      }));
      
      if (!updatedAvatar) {
        throw new Error('Avatar not found');
      }
      
      // If this is the selected avatar, update it too
      if (selectedAvatar && selectedAvatar.id === id) {
        setSelectedAvatar(updatedAvatar);
      }
      
      return updatedAvatar;
    } catch (err) {
      setError('Failed to update avatar');
      console.error(err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [selectedAvatar]);

  const deleteAvatar = useCallback(async (id: string): Promise<void> => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAvatars(prev => prev.filter(avatar => avatar.id !== id));
      if (selectedAvatar?.id === id) {
        setSelectedAvatar(null);
      }
    } catch (err) {
      setError('Failed to delete avatar');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedAvatar]);

  const selectAvatar = useCallback((avatar: Avatar | null) => {
    setSelectedAvatar(avatar);
  }, []);

  const duplicateAvatar = useCallback(async (id: string, newName: string): Promise<Avatar> => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Find the avatar to duplicate
      const originalAvatar = avatars.find(a => a.id === id);
      if (!originalAvatar) {
        throw new Error('Avatar not found');
      }
      
      // In a real app, we would call an API to duplicate
      // const response = await fetch(`/api/avatars/${id}/duplicate`, {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ name: newName })
      // });
      // const data = await response.json();
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 600));
      
      // Create a new avatar based on the original
      const newAvatar: Avatar = {
        ...originalAvatar,
        id: `avatar-${Date.now()}`,
        name: newName,
        versions: [],
        usageStats: {
          totalVideos: 0,
          totalDuration: 0,
          lastUsed: new Date().toISOString()
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      
      setAvatars(prev => [...prev, newAvatar]);
      return newAvatar;
    } catch (err) {
      setError('Failed to duplicate avatar');
      console.error(err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [avatars]);

  const addSample = useCallback(async (
    avatarId: string, 
    sample: Omit<AvatarSample, 'id' | 'createdAt'>
  ): Promise<AvatarSample> => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, we would call an API to add a sample
      // const response = await fetch(`/api/avatars/${avatarId}/samples`, {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(sample)
      // });
      // const data = await response.json();
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 700));
      
      const newSample: AvatarSample = {
        ...sample,
        id: `sample-${Date.now()}`,
        createdAt: new Date().toISOString(),
      };
      
      let updatedAvatar: Avatar | undefined;
      
      setAvatars(prev => prev.map(avatar => {
        if (avatar.id === avatarId) {
          updatedAvatar = {
            ...avatar,
            samples: [...avatar.samples, newSample],
            updatedAt: new Date().toISOString()
          };
          return updatedAvatar;
        }
        return avatar;
      }));
      
      if (!updatedAvatar) {
        throw new Error('Avatar not found');
      }
      
      // If this is the selected avatar, update it too
      if (selectedAvatar && selectedAvatar.id === avatarId) {
        setSelectedAvatar(updatedAvatar);
      }
      
      return newSample;
    } catch (err) {
      setError('Failed to add sample');
      console.error(err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [selectedAvatar]);

  const removeSample = useCallback(async (avatarId: string, sampleId: string): Promise<void> => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, we would call an API to remove a sample
      // await fetch(`/api/avatars/${avatarId}/samples/${sampleId}`, {
      //   method: 'DELETE'
      // });
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 400));
      
      let updatedAvatar: Avatar | undefined;
      
      setAvatars(prev => prev.map(avatar => {
        if (avatar.id === avatarId) {
          updatedAvatar = {
            ...avatar,
            samples: avatar.samples.filter(sample => sample.id !== sampleId),
            updatedAt: new Date().toISOString()
          };
          return updatedAvatar;
        }
        return avatar;
      }));
      
      if (!updatedAvatar) {
        throw new Error('Avatar not found');
      }
      
      // If this is the selected avatar, update it too
      if (selectedAvatar && selectedAvatar.id === avatarId) {
        setSelectedAvatar(updatedAvatar);
      }
    } catch (err) {
      setError('Failed to remove sample');
      console.error(err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [selectedAvatar]);

  const updateAvatarSettings = useCallback(async (
    avatarId: string, 
    settings: { 
      voice?: Partial<Avatar['voiceSettings']>,
      appearance?: Partial<Avatar['appearanceSettings']>,
      animation?: Partial<Avatar['animationSettings']>
    }
  ): Promise<Avatar> => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, we would call an API to update settings
      // const response = await fetch(`/api/avatars/${avatarId}/settings`, {
      //   method: 'PATCH',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(settings)
      // });
      // const data = await response.json();
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      let updatedAvatar: Avatar | undefined;
      
      setAvatars(prev => prev.map(avatar => {
        if (avatar.id === avatarId) {
          updatedAvatar = {
            ...avatar,
            voiceSettings: settings.voice 
              ? { ...avatar.voiceSettings, ...settings.voice } 
              : avatar.voiceSettings,
            appearanceSettings: settings.appearance 
              ? { ...avatar.appearanceSettings, ...settings.appearance } 
              : avatar.appearanceSettings,
            animationSettings: settings.animation 
              ? { ...avatar.animationSettings, ...settings.animation } 
              : avatar.animationSettings,
            updatedAt: new Date().toISOString()
          };
          return updatedAvatar;
        }
        return avatar;
      }));
      
      if (!updatedAvatar) {
        throw new Error('Avatar not found');
      }
      
      // If this is the selected avatar, update it too
      if (selectedAvatar && selectedAvatar.id === avatarId) {
        setSelectedAvatar(updatedAvatar);
      }
      
      return updatedAvatar;
    } catch (err) {
      setError('Failed to update avatar settings');
      console.error(err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [selectedAvatar]);

  const setAvatarPublic = useCallback(async (avatarId: string, isPublic: boolean): Promise<void> => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, we would call an API to update visibility
      // await fetch(`/api/avatars/${avatarId}/visibility`, {
      //   method: 'PATCH',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ isPublic })
      // });
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 300));
      
      let updatedAvatar: Avatar | undefined;
      
      setAvatars(prev => prev.map(avatar => {
        if (avatar.id === avatarId) {
          updatedAvatar = {
            ...avatar,
            isPublic,
            updatedAt: new Date().toISOString()
          };
          return updatedAvatar;
        }
        return avatar;
      }));
      
      if (!updatedAvatar) {
        throw new Error('Avatar not found');
      }
      
      // If this is the selected avatar, update it too
      if (selectedAvatar && selectedAvatar.id === avatarId) {
        setSelectedAvatar(updatedAvatar);
      }
    } catch (err) {
      setError('Failed to update avatar visibility');
      console.error(err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [selectedAvatar]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const value = {
    avatars,
    selectedAvatar,
    isLoading,
    error,
    styles: AVATAR_STYLES,
    fetchAvatars,
    getAvatar,
    createAvatar,
    updateAvatar,
    deleteAvatar,
    selectAvatar,
    duplicateAvatar,
    addSample,
    removeSample,
    updateAvatarSettings,
    setAvatarPublic,
    clearError
  };

  return (
    <AvatarContext.Provider value={value}>
      {children}
    </AvatarContext.Provider>
  );
}; 