import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { RootState } from '.';

// Define base URL from environment variable
const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

export const apiSlice = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl,
    prepareHeaders: (headers, { getState }) => {
      // Get the token from the auth state
      const token = (getState() as RootState).auth.token;
      
      // If we have a token, add it to the request headers
      if (token) {
        headers.set('authorization', `Bearer ${token}`);
      }
      
      return headers;
    },
  }),
  tagTypes: ['User', 'Video', 'Avatar', 'Translation', 'Job'],
  endpoints: (builder) => ({}),
});

// Export hooks for usage in functional components
export const {
  // No generated hooks yet
} = apiSlice; 