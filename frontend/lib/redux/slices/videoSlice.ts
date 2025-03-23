import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Video } from '../services/videoService';

interface VideoState {
  currentVideo: Video | null;
  selectedClips: string[];
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  volume: number;
  isMuted: boolean;
  playbackSpeed: number;
  isLoading: boolean;
  error: string | null;
}

const initialState: VideoState = {
  currentVideo: null,
  selectedClips: [],
  isPlaying: false,
  currentTime: 0,
  duration: 0,
  volume: 1,
  isMuted: false,
  playbackSpeed: 1,
  isLoading: false,
  error: null,
};

const videoSlice = createSlice({
  name: 'video',
  initialState,
  reducers: {
    setCurrentVideo: (state, action: PayloadAction<Video | null>) => {
      state.currentVideo = action.payload;
      state.currentTime = 0;
      state.isPlaying = false;
    },

    togglePlay: (state) => {
      state.isPlaying = !state.isPlaying;
    },

    setIsPlaying: (state, action: PayloadAction<boolean>) => {
      state.isPlaying = action.payload;
    },

    setCurrentTime: (state, action: PayloadAction<number>) => {
      state.currentTime = action.payload;
    },

    setDuration: (state, action: PayloadAction<number>) => {
      state.duration = action.payload;
    },

    setVolume: (state, action: PayloadAction<number>) => {
      state.volume = Math.max(0, Math.min(1, action.payload));
      if (state.volume > 0) {
        state.isMuted = false;
      }
    },

    toggleMute: (state) => {
      state.isMuted = !state.isMuted;
    },

    setPlaybackSpeed: (state, action: PayloadAction<number>) => {
      state.playbackSpeed = action.payload;
    },

    addSelectedClip: (state, action: PayloadAction<string>) => {
      if (!state.selectedClips.includes(action.payload)) {
        state.selectedClips.push(action.payload);
      }
    },

    removeSelectedClip: (state, action: PayloadAction<string>) => {
      state.selectedClips = state.selectedClips.filter(
        (clipId) => clipId !== action.payload
      );
    },

    clearSelectedClips: (state) => {
      state.selectedClips = [];
    },

    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },

    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },

    resetVideoState: (state) => {
      Object.assign(state, initialState);
    },
  },
});

export const {
  setCurrentVideo,
  togglePlay,
  setIsPlaying,
  setCurrentTime,
  setDuration,
  setVolume,
  toggleMute,
  setPlaybackSpeed,
  addSelectedClip,
  removeSelectedClip,
  clearSelectedClips,
  setLoading,
  setError,
  resetVideoState,
} = videoSlice.actions;

export default videoSlice.reducer; 