import { apiSlice } from './apiSlice';

export interface Video {
  id: string;
  title: string;
  description: string;
  url: string;
  thumbnailUrl: string;
  duration: number;
  createdAt: string;
  updatedAt: string;
  status: 'processing' | 'ready' | 'failed';
  metadata: {
    resolution: string;
    format: string;
    size: number;
  };
}

export interface CreateVideoRequest {
  title: string;
  description?: string;
  file: File;
}

export interface UpdateVideoRequest {
  id: string;
  title?: string;
  description?: string;
}

export const videoApi = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    getVideos: builder.query<Video[], void>({
      query: () => 'videos',
      providesTags: ['Videos'],
    }),

    getVideoById: builder.query<Video, string>({
      query: (id) => `videos/${id}`,
      providesTags: (_result, _error, id) => [{ type: 'Videos', id }],
    }),

    createVideo: builder.mutation<Video, CreateVideoRequest>({
      query: (body) => {
        const formData = new FormData();
        formData.append('title', body.title);
        if (body.description) {
          formData.append('description', body.description);
        }
        formData.append('file', body.file);

        return {
          url: 'videos',
          method: 'POST',
          body: formData,
        };
      },
      invalidatesTags: ['Videos'],
    }),

    updateVideo: builder.mutation<Video, UpdateVideoRequest>({
      query: ({ id, ...patch }) => ({
        url: `videos/${id}`,
        method: 'PATCH',
        body: patch,
      }),
      invalidatesTags: (_result, _error, { id }) => [{ type: 'Videos', id }],
    }),

    deleteVideo: builder.mutation<void, string>({
      query: (id) => ({
        url: `videos/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Videos'],
    }),
  }),
});

export const {
  useGetVideosQuery,
  useGetVideoByIdQuery,
  useCreateVideoMutation,
  useUpdateVideoMutation,
  useDeleteVideoMutation,
} = videoApi; 