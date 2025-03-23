import { apiSlice } from './apiSlice';

export interface ClipConfig {
  duration: number;
  aspectRatio: '9:16' | '16:9' | '1:1';
  addCaptions: boolean;
  addBackground: boolean;
  addIntro: boolean;
  addOutro: boolean;
  musicStyle: string;
  captionStyle: string;
}

export interface ClipJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  error?: string;
  outputUrl?: string;
  config: ClipConfig;
  createdAt: string;
  updatedAt: string;
}

export const clipGeneratorApi = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    generateClip: builder.mutation<ClipJob, { videoId: string; config: ClipConfig }>({
      query: ({ videoId, config }) => ({
        url: `/api/videos/${videoId}/clips`,
        method: 'POST',
        body: config,
      }),
    }),

    getClipStatus: builder.query<ClipJob, string>({
      query: (jobId) => `/api/jobs/${jobId}/status`,
      async onCacheEntryAdded(
        jobId,
        { updateCachedData, cacheDataLoaded, cacheEntryRemoved }
      ) {
        let ws: WebSocket;
        try {
          await cacheDataLoaded;

          ws = new WebSocket(
            `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${
              window.location.host
            }/api/ws/jobs/${jobId}`
          );

          ws.onmessage = (event) => {
            const data = JSON.parse(event.data) as ClipJob;
            updateCachedData((draft) => {
              Object.assign(draft, data);
            });
          };
        } catch {
          // Handle WebSocket connection error
        }
        await cacheEntryRemoved;
        ws?.close();
      },
    }),

    listClips: builder.query<ClipJob[], string>({
      query: (videoId) => `/api/videos/${videoId}/clips`,
    }),

    deleteClip: builder.mutation<void, { videoId: string; clipId: string }>({
      query: ({ videoId, clipId }) => ({
        url: `/api/videos/${videoId}/clips/${clipId}`,
        method: 'DELETE',
      }),
    }),
  }),
});

export const {
  useGenerateClipMutation,
  useGetClipStatusQuery,
  useListClipsQuery,
  useDeleteClipMutation,
} = clipGeneratorApi; 