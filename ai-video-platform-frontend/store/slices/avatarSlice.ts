import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface AvatarState {
  avatarId: string | null;
  isGenerating: boolean;
  error: string | null;
  settings: {
    appearance: {
      skinTone: string;
      hairStyle: string;
      hairColor: string;
      eyeColor: string;
      facialFeatures: number;
      age: number;
      jawline: number;
    };
    voice: {
      pitch: number;
      speed: number;
      accent: string;
    };
  };
}

const initialState: AvatarState = {
  avatarId: null,
  isGenerating: false,
  error: null,
  settings: {
    appearance: {
      skinTone: 'medium',
      hairStyle: 'short',
      hairColor: 'black',
      eyeColor: 'brown',
      facialFeatures: 50,
      age: 30,
      jawline: 50,
    },
    voice: {
      pitch: 50,
      speed: 50,
      accent: 'neutral',
    },
  },
};

const avatarSlice = createSlice({
  name: 'avatar',
  initialState,
  reducers: {
    setAvatarId: (state, action: PayloadAction<string>) => {
      state.avatarId = action.payload;
    },
    setGenerating: (state, action: PayloadAction<boolean>) => {
      state.isGenerating = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    updateAppearanceSettings: (state, action: PayloadAction<Partial<AvatarState['settings']['appearance']>>) => {
      state.settings.appearance = {
        ...state.settings.appearance,
        ...action.payload,
      };
    },
    updateVoiceSettings: (state, action: PayloadAction<Partial<AvatarState['settings']['voice']>>) => {
      state.settings.voice = {
        ...state.settings.voice,
        ...action.payload,
      };
    },
    resetSettings: (state) => {
      state.settings = initialState.settings;
    },
  },
});

export const {
  setAvatarId,
  setGenerating,
  setError,
  updateAppearanceSettings,
  updateVoiceSettings,
  resetSettings,
} = avatarSlice.actions;

export default avatarSlice.reducer; 