import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface TranslationJob {
  id: string;
  sourceLanguage: string;
  targetLanguages: string[];
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  result?: {
    url: string;
    subtitles: string;
  };
  error?: string;
}

interface TranslationState {
  jobs: TranslationJob[];
  activeJobId: string | null;
  isLoading: boolean;
  error: string | null;
  settings: {
    preserveNames: boolean;
    preserveFormatting: boolean;
    preserveTechnicalTerms: boolean;
    formality: 'formal' | 'neutral' | 'informal';
    keepOriginalVoice: boolean;
    useTranslationMemory: boolean;
  };
}

const initialState: TranslationState = {
  jobs: [],
  activeJobId: null,
  isLoading: false,
  error: null,
  settings: {
    preserveNames: true,
    preserveFormatting: true,
    preserveTechnicalTerms: true,
    formality: 'neutral',
    keepOriginalVoice: false,
    useTranslationMemory: true,
  },
};

const translationSlice = createSlice({
  name: 'translation',
  initialState,
  reducers: {
    setJobs: (state, action: PayloadAction<TranslationJob[]>) => {
      state.jobs = action.payload;
    },
    addJob: (state, action: PayloadAction<TranslationJob>) => {
      state.jobs.push(action.payload);
    },
    updateJob: (state, action: PayloadAction<Partial<TranslationJob> & { id: string }>) => {
      const index = state.jobs.findIndex(job => job.id === action.payload.id);
      if (index !== -1) {
        state.jobs[index] = { ...state.jobs[index], ...action.payload };
      }
    },
    setActiveJobId: (state, action: PayloadAction<string | null>) => {
      state.activeJobId = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    updateSettings: (state, action: PayloadAction<Partial<TranslationState['settings']>>) => {
      state.settings = {
        ...state.settings,
        ...action.payload,
      };
    },
    resetSettings: (state) => {
      state.settings = initialState.settings;
    },
  },
});

export const {
  setJobs,
  addJob,
  updateJob,
  setActiveJobId,
  setLoading,
  setError,
  updateSettings,
  resetSettings,
} = translationSlice.actions;

export default translationSlice.reducer; 