'use client';

import React, { createContext, useContext, useState, ReactNode, useCallback } from 'react';
import { MediaFile } from './media-context';

export interface EditSettings {
  subtitles: {
    enabled: boolean;
    style: {
      font: string;
      size: number;
      color: string;
      backgroundColor: string;
      position: 'top' | 'bottom' | 'middle';
      outline: boolean;
      outlineColor: string;
    };
  };
  bRoll: {
    enabled: boolean;
    intensity: number; // 1-10
    categories: string[];
    autoCutaways: boolean;
  };
  music: {
    enabled: boolean;
    genre: string;
    volume: number; // 0-100
    fadeIn: boolean;
    fadeOut: boolean;
  };
  soundEffects: {
    enabled: boolean;
    categories: string[];
    intensity: number; // 1-10
  };
  audioEnhancement: {
    enabled: boolean;
    noiseReduction: number; // 0-100
    normalization: boolean;
    voiceEnhancement: number; // 0-100
  };
  pauseRemoval: {
    enabled: boolean;
    threshold: number; // milliseconds
    maxPause: number; // milliseconds
  };
  videoEnhancement: {
    enabled: boolean;
    stabilization: boolean;
    colorCorrection: boolean;
    denoising: boolean;
    sharpening: number; // 0-100
  };
}

export interface Preset {
  id: string;
  name: string;
  description?: string;
  settings: EditSettings;
  isDefault?: boolean;
  createdAt: string;
  updatedAt: string;
}

interface EditorContextProps {
  activeFile: MediaFile | null;
  settings: EditSettings;
  availablePresets: Preset[];
  activePreset: Preset | null;
  beforeAfterMode: boolean;
  setActiveFile: (file: MediaFile | null) => void;
  updateSettings: <K extends keyof EditSettings>(
    category: K,
    values: Partial<EditSettings[K]>
  ) => void;
  resetSettings: () => void;
  loadPreset: (presetId: string) => void;
  savePreset: (name: string, description?: string) => Promise<Preset>;
  deletePreset: (presetId: string) => Promise<void>;
  toggleBeforeAfterMode: () => void;
}

const EditorContext = createContext<EditorContextProps>({} as EditorContextProps);

export const useEditorContext = () => useContext(EditorContext);

const DEFAULT_SETTINGS: EditSettings = {
  subtitles: {
    enabled: true,
    style: {
      font: 'Arial',
      size: 24,
      color: '#FFFFFF',
      backgroundColor: '#000000',
      position: 'bottom',
      outline: true,
      outlineColor: '#000000',
    },
  },
  bRoll: {
    enabled: true,
    intensity: 5,
    categories: ['stock', 'illustrations', 'diagrams'],
    autoCutaways: true,
  },
  music: {
    enabled: true,
    genre: 'corporate',
    volume: 30,
    fadeIn: true,
    fadeOut: true,
  },
  soundEffects: {
    enabled: true,
    categories: ['transitions', 'ambient', 'ui'],
    intensity: 3,
  },
  audioEnhancement: {
    enabled: true,
    noiseReduction: 60,
    normalization: true,
    voiceEnhancement: 70,
  },
  pauseRemoval: {
    enabled: true,
    threshold: 500,
    maxPause: 2000,
  },
  videoEnhancement: {
    enabled: true,
    stabilization: true,
    colorCorrection: true,
    denoising: true,
    sharpening: 40,
  },
};

const DEFAULT_PRESETS: Preset[] = [
  {
    id: 'default',
    name: 'Balanced',
    description: 'Balanced settings for general purpose videos',
    settings: DEFAULT_SETTINGS,
    isDefault: true,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: 'interview',
    name: 'Interview',
    description: 'Optimized for interview videos with clear subtitles',
    settings: {
      ...DEFAULT_SETTINGS,
      subtitles: {
        ...DEFAULT_SETTINGS.subtitles,
        style: {
          ...DEFAULT_SETTINGS.subtitles.style,
          size: 28,
          backgroundColor: 'rgba(0,0,0,0.7)',
        }
      },
      pauseRemoval: {
        ...DEFAULT_SETTINGS.pauseRemoval,
        threshold: 1000,
        maxPause: 1500,
      },
      bRoll: {
        ...DEFAULT_SETTINGS.bRoll,
        intensity: 3,
      }
    },
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
  {
    id: 'tutorial',
    name: 'Tutorial/Educational',
    description: 'Settings optimized for educational content',
    settings: {
      ...DEFAULT_SETTINGS,
      subtitles: {
        ...DEFAULT_SETTINGS.subtitles,
        style: {
          ...DEFAULT_SETTINGS.subtitles.style,
          size: 26,
          position: 'top',
        }
      },
      pauseRemoval: {
        ...DEFAULT_SETTINGS.pauseRemoval,
        enabled: false,
      },
      bRoll: {
        ...DEFAULT_SETTINGS.bRoll,
        intensity: 7,
        categories: ['diagrams', 'illustrations', 'demos'],
      }
    },
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  },
];

export const EditorProvider = ({ children }: { children: ReactNode }) => {
  const [activeFile, setActiveFile] = useState<MediaFile | null>(null);
  const [settings, setSettings] = useState<EditSettings>(DEFAULT_SETTINGS);
  const [availablePresets, setAvailablePresets] = useState<Preset[]>(DEFAULT_PRESETS);
  const [activePreset, setActivePreset] = useState<Preset | null>(DEFAULT_PRESETS[0]);
  const [beforeAfterMode, setBeforeAfterMode] = useState(false);
  
  const updateSettings = useCallback(<K extends keyof EditSettings>(
    category: K,
    values: Partial<EditSettings[K]>
  ) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        ...values,
      }
    }));
    
    // If we have an active preset and we're changing settings, we're no longer using it exactly
    if (activePreset && activePreset.isDefault !== true) {
      setActivePreset(null);
    }
  }, [activePreset]);
  
  const resetSettings = useCallback(() => {
    setSettings(DEFAULT_SETTINGS);
    setActivePreset(DEFAULT_PRESETS[0]);
  }, []);
  
  const loadPreset = useCallback((presetId: string) => {
    const preset = availablePresets.find(p => p.id === presetId);
    if (preset) {
      setSettings(preset.settings);
      setActivePreset(preset);
    }
  }, [availablePresets]);
  
  const savePreset = useCallback(async (name: string, description?: string): Promise<Preset> => {
    // In a real app, you would save this to an API
    // const response = await fetch('/api/presets', {
    //   method: 'POST',
    //   body: JSON.stringify({ name, description, settings }),
    //   headers: { 'Content-Type': 'application/json' }
    // });
    // const data = await response.json();
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 500));
    
    const newPreset: Preset = {
      id: `preset-${Date.now()}`,
      name,
      description,
      settings,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    
    setAvailablePresets(prev => [...prev, newPreset]);
    setActivePreset(newPreset);
    
    return newPreset;
  }, [settings]);
  
  const deletePreset = useCallback(async (presetId: string): Promise<void> => {
    // In a real app, you would delete this via an API
    // await fetch(`/api/presets/${presetId}`, { method: 'DELETE' });
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 300));
    
    setAvailablePresets(prev => prev.filter(preset => preset.id !== presetId));
    
    // If the active preset is being deleted, reset to default
    if (activePreset && activePreset.id === presetId) {
      setActivePreset(DEFAULT_PRESETS[0]);
      setSettings(DEFAULT_PRESETS[0].settings);
    }
  }, [activePreset]);
  
  const toggleBeforeAfterMode = useCallback(() => {
    setBeforeAfterMode(prev => !prev);
  }, []);
  
  const value = {
    activeFile,
    settings,
    availablePresets,
    activePreset,
    beforeAfterMode,
    setActiveFile,
    updateSettings,
    resetSettings,
    loadPreset,
    savePreset,
    deletePreset,
    toggleBeforeAfterMode,
  };
  
  return (
    <EditorContext.Provider value={value}>
      {children}
    </EditorContext.Provider>
  );
}; 