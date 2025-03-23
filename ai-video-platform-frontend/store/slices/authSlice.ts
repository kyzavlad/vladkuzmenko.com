import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { apiSlice } from '../apiSlice';
import Cookies from 'js-cookie';

export interface User {
  id: string;
  email: string;
  name: string;
  role: string;
  avatar?: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

const initialState: AuthState = {
  user: null,
  token: typeof window !== 'undefined' ? Cookies.get('token') || null : null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    setCredentials: (
      state,
      action: PayloadAction<{ user: User; token: string }>
    ) => {
      const { user, token } = action.payload;
      state.user = user;
      state.token = token;
      state.isAuthenticated = true;
      // Save token to cookies for persistence
      Cookies.set('token', token, { expires: 7 }); // 7 days
    },
    logout: (state) => {
      state.user = null;
      state.token = null;
      state.isAuthenticated = false;
      // Remove token from cookies
      Cookies.remove('token');
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

// Export the extended API slice with auth endpoints
export const authApiSlice = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    login: builder.mutation<{ user: User; token: string }, { email: string; password: string }>({
      query: (credentials) => ({
        url: '/auth/login',
        method: 'POST',
        body: credentials,
      }),
    }),
    
    register: builder.mutation<{ user: User; token: string }, { name: string; email: string; password: string }>({
      query: (userData) => ({
        url: '/auth/register',
        method: 'POST',
        body: userData,
      }),
    }),
    
    getCurrentUser: builder.query<User, void>({
      query: () => '/auth/me',
    }),
    
    logout: builder.mutation<{ success: boolean }, void>({
      query: () => ({
        url: '/auth/logout',
        method: 'POST',
      }),
    }),
  }),
});

export const { setCredentials, logout, setLoading, setError } = authSlice.actions;

export const {
  useLoginMutation,
  useRegisterMutation,
  useGetCurrentUserQuery,
  useLogoutMutation,
} = authApiSlice;

export default authSlice.reducer; 