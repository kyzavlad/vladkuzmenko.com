import { apiSlice } from './apiSlice';

export interface Language {
  code: string;
  name: string;
  nativeName: string;
  dialects?: { code: string; name: string }[];
}

export interface TranslationConfig {
  sourceLanguage: string;
  sourceDialect?: string;
  targetLanguages: Array<{
    code: string;
    dialect?: string;
  }>;
  preserveTone: boolean;
  preserveAccent: boolean;
  generateSubtitles: boolean;
}

export interface TranslationJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  error?: string;
  outputs: Array<{
    language: string;
    dialect?: string;
    url: string;
    subtitlesUrl?: string;
  }>;
  config: TranslationConfig;
  createdAt: string;
  updatedAt: string;
}

export interface Subtitle {
  id: string;
  startTime: number;
  endTime: number;
  text: string;
  translation: string;
}

export const translationApi = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    translateVideo: builder.mutation<
      TranslationJob,
      { videoId: string; config: TranslationConfig }
    >({
      query: ({ videoId, config }) => ({
        url: `/api/videos/${videoId}/translate`,
        method: 'POST',
        body: config,
      }),
    }),

    getTranslationStatus: builder.query<TranslationJob, string>({
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
            const data = JSON.parse(event.data) as TranslationJob;
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

    listTranslations: builder.query<TranslationJob[], string>({
      query: (videoId) => `/api/videos/${videoId}/translations`,
    }),

    getSubtitles: builder.query<
      Subtitle[],
      { videoId: string; jobId: string; language: string }
    >({
      query: ({ videoId, jobId, language }) =>
        `/api/videos/${videoId}/translations/${jobId}/subtitles/${language}`,
    }),

    updateSubtitle: builder.mutation<
      void,
      {
        videoId: string;
        jobId: string;
        language: string;
        subtitle: Subtitle;
      }
    >({
      query: ({ videoId, jobId, language, subtitle }) => ({
        url: `/api/videos/${videoId}/translations/${jobId}/subtitles/${language}/${subtitle.id}`,
        method: 'PATCH',
        body: subtitle,
      }),
    }),

    getSupportedLanguages: builder.query<Language[], void>({
      query: () => '/api/languages',
    }),
  }),
});

export const {
  useTranslateVideoMutation,
  useGetTranslationStatusQuery,
  useListTranslationsQuery,
  useGetSubtitlesQuery,
  useUpdateSubtitleMutation,
  useGetSupportedLanguagesQuery,
} = translationApi; 