import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { User } from '../services/authService';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

const initialState: AuthState = {
  user: null,
  token: typeof window !== 'undefined' ? localStorage.getItem('token') : null,
  isAuthenticated: false,
  isLoading: true,
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
      state.error = null;
      if (typeof window !== 'undefined') {
        localStorage.setItem('token', token);
      }
    },
    
    updateUser: (state, action: PayloadAction<User>) => {
      state.user = action.payload;
    },

    logout: (state) => {
      state.user = null;
      state.token = null;
      state.isAuthenticated = false;
      state.error = null;
      if (typeof window !== 'undefined') {
        localStorage.removeItem('token');
      }
    },

    setError: (state, action: PayloadAction<string>) => {
      state.error = action.payload;
    },

    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },

    clearError: (state) => {
      state.error = null;
    },
  },
});

export const {
  setCredentials,
  updateUser,
  logout,
  setError,
  setLoading,
  clearError,
} = authSlice.actions;

export default authSlice.reducer;
