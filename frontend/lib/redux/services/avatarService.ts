import { apiSlice } from './apiSlice';

export interface Avatar {
  id: string;
  name: string;
  imageUrl: string;
  voiceId: string;
  style: 'realistic' | 'cartoon' | 'anime';
  gender: 'male' | 'female' | 'neutral';
  createdAt: string;
  updatedAt: string;
  customization: {
    skinTone: string;
    hairColor: string;
    eyeColor: string;
    outfitColor: string;
    background: string;
  };
}

export interface CreateAvatarRequest {
  name: string;
  style: Avatar['style'];
  gender: Avatar['gender'];
  voiceId: string;
  customization?: Partial<Avatar['customization']>;
}

export interface UpdateAvatarRequest {
  id: string;
  name?: string;
  voiceId?: string;
  customization?: Partial<Avatar['customization']>;
}

export const avatarApi = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    getAvatars: builder.query<Avatar[], void>({
      query: () => 'avatars',
      providesTags: ['Avatars'],
    }),

    getAvatarById: builder.query<Avatar, string>({
      query: (id) => `avatars/${id}`,
      providesTags: (_result, _error, id) => [{ type: 'Avatars', id }],
    }),

    createAvatar: builder.mutation<Avatar, CreateAvatarRequest>({
      query: (body) => ({
        url: 'avatars',
        method: 'POST',
        body,
      }),
      invalidatesTags: ['Avatars'],
    }),

    updateAvatar: builder.mutation<Avatar, UpdateAvatarRequest>({
      query: ({ id, ...patch }) => ({
        url: `avatars/${id}`,
        method: 'PATCH',
        body: patch,
      }),
      invalidatesTags: (_result, _error, { id }) => [{ type: 'Avatars', id }],
    }),

    deleteAvatar: builder.mutation<void, string>({
      query: (id) => ({
        url: `avatars/${id}`,
        method: 'DELETE',
      }),
      invalidatesTags: ['Avatars'],
    }),

    generateAvatarVideo: builder.mutation<{ taskId: string }, { avatarId: string; script: string }>({
      query: ({ avatarId, script }) => ({
        url: `avatars/${avatarId}/generate`,
        method: 'POST',
        body: { script },
      }),
    }),

    getGenerationStatus: builder.query<{ status: 'pending' | 'processing' | 'completed' | 'failed', videoUrl?: string }, string>({
      query: (taskId) => `tasks/${taskId}`,
    }),
  }),
});

export const {
  useGetAvatarsQuery,
  useGetAvatarByIdQuery,
  useCreateAvatarMutation,
  useUpdateAvatarMutation,
  useDeleteAvatarMutation,
  useGenerateAvatarVideoMutation,
  useGetGenerationStatusQuery,
} = avatarApi; 