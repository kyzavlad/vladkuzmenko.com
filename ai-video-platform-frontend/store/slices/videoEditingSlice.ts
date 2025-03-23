import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { apiSlice } from '../apiSlice';

export interface VideoJob {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  created_at: string;
  completed_at?: string;
  input_url: string;
  output_url?: string;
  settings: VideoSettings;
  error?: string;
}

export interface VideoSettings {
  resolution: string;
  format: string;
  effects: VideoEffect[];
  trim: {
    start?: number;
    end?: number;
  };
  scenes?: {
    id: string;
    start: number;
    end: number;
    label: string;
  }[];
}

export interface VideoEffect {
  type: string;
  params: Record<string, any>;
  start_time?: number;
  end_time?: number;
}

interface VideoEditingState {
  currentJob: VideoJob | null;
  recentJobs: VideoJob[];
  isUploading: boolean;
  uploadProgress: number;
  selectedVideoUrl: string | null;
  previewUrl: string | null;
  isProcessing: boolean;
  error: string | null;
}

const initialState: VideoEditingState = {
  currentJob: null,
  recentJobs: [],
  isUploading: false,
  uploadProgress: 0,
  selectedVideoUrl: null,
  previewUrl: null,
  isProcessing: false,
  error: null,
};

const videoEditingSlice = createSlice({
  name: 'videoEditing',
  initialState,
  reducers: {
    setCurrentJob: (state, action: PayloadAction<VideoJob | null>) => {
      state.currentJob = action.payload;
    },
    setRecentJobs: (state, action: PayloadAction<VideoJob[]>) => {
      state.recentJobs = action.payload;
    },
    setIsUploading: (state, action: PayloadAction<boolean>) => {
      state.isUploading = action.payload;
      if (!action.payload) {
        state.uploadProgress = 0;
      }
    },
    setUploadProgress: (state, action: PayloadAction<number>) => {
      state.uploadProgress = action.payload;
    },
    setSelectedVideoUrl: (state, action: PayloadAction<string | null>) => {
      state.selectedVideoUrl = action.payload;
    },
    setPreviewUrl: (state, action: PayloadAction<string | null>) => {
      state.previewUrl = action.payload;
    },
    setIsProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    resetState: (state) => {
      return initialState;
    },
  },
});

// Export the extended API slice with video editing endpoints
export const videoApiSlice = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    // Get all video jobs for the current user
    getVideoJobs: builder.query<VideoJob[], void>({
      query: () => '/video',
      providesTags: ['Video', 'Job'],
    }),
    
    // Get a specific video job by ID
    getVideoJob: builder.query<VideoJob, string>({
      query: (id) => `/video/${id}`,
      providesTags: (result, error, id) => [{ type: 'Video', id }],
    }),
    
    // Create a new video job
    createVideoJob: builder.mutation<VideoJob, { name: string; input_url: string; settings: VideoSettings }>({
      query: (videoData) => ({
        url: '/video/process',
        method: 'POST',
        body: videoData,
      }),
      invalidatesTags: ['Video', 'Job'],
    }),
    
    // Update video settings
    updateVideoSettings: builder.mutation<VideoJob, { id: string; settings: Partial<VideoSettings> }>({
      query: ({ id, settings }) => ({
        url: `/video/${id}/settings`,
        method: 'PATCH',
        body: settings,
      }),
      invalidatesTags: (result, error, { id }) => [{ type: 'Video', id }],
    }),
    
    // Cancel a video job
    cancelVideoJob: builder.mutation<{ success: boolean }, string>({
      query: (id) => ({
        url: `/video/${id}/cancel`,
        method: 'POST',
      }),
      invalidatesTags: (result, error, id) => [{ type: 'Video', id }],
    }),
    
    // Upload a video file
    uploadVideo: builder.mutation<{ url: string }, FormData>({
      query: (formData) => ({
        url: '/video/upload',
        method: 'POST',
        body: formData,
        formData: true,
      }),
    }),
  }),
});

export const {
  setCurrentJob,
  setRecentJobs,
  setIsUploading,
  setUploadProgress,
  setSelectedVideoUrl,
  setPreviewUrl,
  setIsProcessing,
  setError,
  resetState,
} = videoEditingSlice.actions;

export const {
  useGetVideoJobsQuery,
  useGetVideoJobQuery,
  useCreateVideoJobMutation,
  useUpdateVideoSettingsMutation,
  useCancelVideoJobMutation,
  useUploadVideoMutation,
} = videoApiSlice;

export default videoEditingSlice.reducer; 