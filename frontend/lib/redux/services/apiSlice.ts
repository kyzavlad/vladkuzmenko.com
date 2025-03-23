import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { RootState } from '../store';

export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: process.env.NEXT_PUBLIC_API_URL,
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token;
      if (token) {
        headers.set('authorization', `Bearer ${token}`);
      }
      return headers;
    },
  }),
  tagTypes: ['Videos', 'Projects', 'Avatars', 'Translations'],
  endpoints: () => ({}),
});

// Export hooks for usage in components
export const {
  middleware: apiMiddleware,
  reducer: apiReducer,
  reducerPath: apiReducerPath,
} = apiSlice; 