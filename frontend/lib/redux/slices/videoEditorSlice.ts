import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { VideoEditSettings } from '@/components/video/editor/SettingsPanel';
import { Effect } from '@/components/video/editor/EffectsPanel';
import { videoProcessingService } from '@/lib/services/videoProcessingService';

interface VideoEditorState {
  currentVideoId: string | null;
  currentTime: number;
  duration: number;
  isPlaying: boolean;
  isMuted: boolean;
  volume: number;
  settings: VideoEditSettings;
  effects: Effect[];
  processing: {
    status: 'idle' | 'processing' | 'completed' | 'error';
    progress: number;
    message: string;
  };
}

const initialState: VideoEditorState = {
  currentVideoId: null,
  currentTime: 0,
  duration: 0,
  isPlaying: false,
  isMuted: false,
  volume: 1,
  settings: {
    subtitles: {
      enabled: true,
      style: 'overlay',
      fontSize: 24,
      position: 'bottom',
    },
    bRoll: {
      enabled: true,
      intensity: 50,
      timing: 'auto',
    },
    audio: {
      music: {
        enabled: true,
        genre: 'ambient',
        volume: 30,
      },
      soundEffects: {
        enabled: true,
        categories: ['transitions', 'emphasis'],
      },
      voiceEnhancement: {
        enabled: true,
        clarity: 70,
        denoising: 50,
      },
    },
    timing: {
      pauseThreshold: 0.5,
      transitionDuration: 0.5,
    },
  },
  effects: [],
  processing: {
    status: 'idle',
    progress: 0,
    message: '',
  },
};

// Async thunks
export const processVideo = createAsyncThunk(
  'videoEditor/processVideo',
  async (
    {
      videoId,
      settings,
      effects,
    }: {
      videoId: string;
      settings: VideoEditSettings;
      effects: Effect[];
    },
    { rejectWithValue }
  ) => {
    try {
      const result = await videoProcessingService.processVideo(
        videoId,
        settings,
        effects,
        (progress) => {
          // Handle progress updates
        }
      );
      return result;
    } catch (error) {
      return rejectWithValue(
        error instanceof Error ? error.message : 'Failed to process video'
      );
    }
  }
);

export const applyEffect = createAsyncThunk(
  'videoEditor/applyEffect',
  async (
    {
      videoId,
      effect,
    }: {
      videoId: string;
      effect: Effect;
    },
    { rejectWithValue }
  ) => {
    try {
      const result = await videoProcessingService.applyEffect(
        videoId,
        effect,
        (progress) => {
          // Handle progress updates
        }
      );
      return { result, effect };
    } catch (error) {
      return rejectWithValue(
        error instanceof Error ? error.message : 'Failed to apply effect'
      );
    }
  }
);

export const updateEffect = createAsyncThunk(
  'videoEditor/updateEffect',
  async (
    {
      videoId,
      effectId,
      updates,
    }: {
      videoId: string;
      effectId: string;
      updates: Partial<Effect>;
    },
    { rejectWithValue }
  ) => {
    try {
      const result = await videoProcessingService.updateEffect(
        videoId,
        effectId,
        updates,
        (progress) => {
          // Handle progress updates
        }
      );
      return { result, effectId, updates };
    } catch (error) {
      return rejectWithValue(
        error instanceof Error ? error.message : 'Failed to update effect'
      );
    }
  }
);

export const removeEffect = createAsyncThunk(
  'videoEditor/removeEffect',
  async (
    {
      videoId,
      effectId,
    }: {
      videoId: string;
      effectId: string;
    },
    { rejectWithValue }
  ) => {
    try {
      const result = await videoProcessingService.removeEffect(
        videoId,
        effectId,
        (progress) => {
          // Handle progress updates
        }
      );
      return { result, effectId };
    } catch (error) {
      return rejectWithValue(
        error instanceof Error ? error.message : 'Failed to remove effect'
      );
    }
  }
);

const videoEditorSlice = createSlice({
  name: 'videoEditor',
  initialState,
  reducers: {
    setCurrentVideo(state, action: PayloadAction<string>) {
      state.currentVideoId = action.payload;
    },
    updateTime(state, action: PayloadAction<number>) {
      state.currentTime = action.payload;
    },
    setDuration(state, action: PayloadAction<number>) {
      state.duration = action.payload;
    },
    togglePlayPause(state) {
      state.isPlaying = !state.isPlaying;
    },
    toggleMute(state) {
      state.isMuted = !state.isMuted;
    },
    setVolume(state, action: PayloadAction<number>) {
      state.volume = action.payload;
    },
    updateSettings(state, action: PayloadAction<Partial<VideoEditSettings>>) {
      state.settings = { ...state.settings, ...action.payload };
    },
  },
  extraReducers: (builder) => {
    // Process Video
    builder
      .addCase(processVideo.pending, (state) => {
        state.processing = {
          status: 'processing',
          progress: 0,
          message: 'Processing video...',
        };
      })
      .addCase(processVideo.fulfilled, (state) => {
        state.processing = {
          status: 'completed',
          progress: 100,
          message: 'Video processing completed',
        };
      })
      .addCase(processVideo.rejected, (state, action) => {
        state.processing = {
          status: 'error',
          progress: 0,
          message: action.payload as string,
        };
      });

    // Apply Effect
    builder
      .addCase(applyEffect.pending, (state) => {
        state.processing = {
          status: 'processing',
          progress: 0,
          message: 'Applying effect...',
        };
      })
      .addCase(applyEffect.fulfilled, (state, action) => {
        state.effects.push(action.payload.effect);
        state.processing = {
          status: 'completed',
          progress: 100,
          message: 'Effect applied successfully',
        };
      })
      .addCase(applyEffect.rejected, (state, action) => {
        state.processing = {
          status: 'error',
          progress: 0,
          message: action.payload as string,
        };
      });

    // Update Effect
    builder
      .addCase(updateEffect.pending, (state) => {
        state.processing = {
          status: 'processing',
          progress: 0,
          message: 'Updating effect...',
        };
      })
      .addCase(updateEffect.fulfilled, (state, action) => {
        const index = state.effects.findIndex(
          (effect) => effect.id === action.payload.effectId
        );
        if (index !== -1) {
          state.effects[index] = {
            ...state.effects[index],
            ...action.payload.updates,
          };
        }
        state.processing = {
          status: 'completed',
          progress: 100,
          message: 'Effect updated successfully',
        };
      })
      .addCase(updateEffect.rejected, (state, action) => {
        state.processing = {
          status: 'error',
          progress: 0,
          message: action.payload as string,
        };
      });

    // Remove Effect
    builder
      .addCase(removeEffect.pending, (state) => {
        state.processing = {
          status: 'processing',
          progress: 0,
          message: 'Removing effect...',
        };
      })
      .addCase(removeEffect.fulfilled, (state, action) => {
        state.effects = state.effects.filter(
          (effect) => effect.id !== action.payload.effectId
        );
        state.processing = {
          status: 'completed',
          progress: 100,
          message: 'Effect removed successfully',
        };
      })
      .addCase(removeEffect.rejected, (state, action) => {
        state.processing = {
          status: 'error',
          progress: 0,
          message: action.payload as string,
        };
      });
  },
});

export const {
  setCurrentVideo,
  updateTime,
  setDuration,
  togglePlayPause,
  toggleMute,
  setVolume,
  updateSettings,
} = videoEditorSlice.actions;

export default videoEditorSlice.reducer; 