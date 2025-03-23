import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Translation } from '../services/translationService';

interface TranslationState {
  currentTranslation: Translation | null;
  sourceLanguage: string;
  targetLanguage: string;
  detectedLanguage: string | null;
  supportedLanguages: Array<{
    code: string;
    name: string;
    nativeName: string;
  }>;
  preserveFormatting: boolean;
  history: Translation[];
  isTranslating: boolean;
  error: string | null;
}

const initialState: TranslationState = {
  currentTranslation: null,
  sourceLanguage: 'auto',
  targetLanguage: 'en',
  detectedLanguage: null,
  supportedLanguages: [],
  preserveFormatting: true,
  history: [],
  isTranslating: false,
  error: null,
};

const translationSlice = createSlice({
  name: 'translation',
  initialState,
  reducers: {
    setCurrentTranslation: (state, action: PayloadAction<Translation | null>) => {
      state.currentTranslation = action.payload;
      if (action.payload) {
        state.history = [action.payload, ...state.history].slice(0, 50);
      }
    },

    setSourceLanguage: (state, action: PayloadAction<string>) => {
      state.sourceLanguage = action.payload;
    },

    setTargetLanguage: (state, action: PayloadAction<string>) => {
      state.targetLanguage = action.payload;
    },

    setDetectedLanguage: (state, action: PayloadAction<string | null>) => {
      state.detectedLanguage = action.payload;
    },

    setSupportedLanguages: (
      state,
      action: PayloadAction<TranslationState['supportedLanguages']>
    ) => {
      state.supportedLanguages = action.payload;
    },

    togglePreserveFormatting: (state) => {
      state.preserveFormatting = !state.preserveFormatting;
    },

    clearHistory: (state) => {
      state.history = [];
    },

    setIsTranslating: (state, action: PayloadAction<boolean>) => {
      state.isTranslating = action.payload;
    },

    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },

    swapLanguages: (state) => {
      if (state.sourceLanguage !== 'auto') {
        [state.sourceLanguage, state.targetLanguage] = [
          state.targetLanguage,
          state.sourceLanguage,
        ];
      }
    },

    resetTranslationState: (state) => {
      Object.assign(state, initialState);
    },
  },
});

export const {
  setCurrentTranslation,
  setSourceLanguage,
  setTargetLanguage,
  setDetectedLanguage,
  setSupportedLanguages,
  togglePreserveFormatting,
  clearHistory,
  setIsTranslating,
  setError,
  swapLanguages,
  resetTranslationState,
} = translationSlice.actions;

export default translationSlice.reducer; 