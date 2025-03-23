import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Avatar } from '../services/avatarService';

interface AvatarState {
  currentAvatar: Avatar | null;
  isGenerating: boolean;
  generationProgress: number;
  currentTaskId: string | null;
  generatedVideoUrl: string | null;
  isPreviewMode: boolean;
  customization: {
    skinTone: string;
    hairColor: string;
    eyeColor: string;
    outfitColor: string;
    background: string;
  };
  script: string;
  isLoading: boolean;
  error: string | null;
}

const initialState: AvatarState = {
  currentAvatar: null,
  isGenerating: false,
  generationProgress: 0,
  currentTaskId: null,
  generatedVideoUrl: null,
  isPreviewMode: false,
  customization: {
    skinTone: '#F5D0C5',
    hairColor: '#4A4A4A',
    eyeColor: '#634E34',
    outfitColor: '#2B6CB0',
    background: '#FFFFFF',
  },
  script: '',
  isLoading: false,
  error: null,
};

const avatarSlice = createSlice({
  name: 'avatar',
  initialState,
  reducers: {
    setCurrentAvatar: (state, action: PayloadAction<Avatar | null>) => {
      state.currentAvatar = action.payload;
      if (action.payload?.customization) {
        state.customization = {
          ...state.customization,
          ...action.payload.customization,
        };
      }
    },

    setGenerating: (state, action: PayloadAction<boolean>) => {
      state.isGenerating = action.payload;
      if (!action.payload) {
        state.generationProgress = 0;
      }
    },

    setGenerationProgress: (state, action: PayloadAction<number>) => {
      state.generationProgress = Math.min(100, Math.max(0, action.payload));
    },

    setCurrentTaskId: (state, action: PayloadAction<string | null>) => {
      state.currentTaskId = action.payload;
    },

    setGeneratedVideoUrl: (state, action: PayloadAction<string | null>) => {
      state.generatedVideoUrl = action.payload;
    },

    togglePreviewMode: (state) => {
      state.isPreviewMode = !state.isPreviewMode;
    },

    updateCustomization: (
      state,
      action: PayloadAction<Partial<AvatarState['customization']>>
    ) => {
      state.customization = {
        ...state.customization,
        ...action.payload,
      };
    },

    setScript: (state, action: PayloadAction<string>) => {
      state.script = action.payload;
    },

    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },

    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },

    resetAvatarState: (state) => {
      Object.assign(state, initialState);
    },
  },
});

export const {
  setCurrentAvatar,
  setGenerating,
  setGenerationProgress,
  setCurrentTaskId,
  setGeneratedVideoUrl,
  togglePreviewMode,
  updateCustomization,
  setScript,
  setLoading,
  setError,
  resetAvatarState,
} = avatarSlice.actions;

export default avatarSlice.reducer; 