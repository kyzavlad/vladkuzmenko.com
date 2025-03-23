import { apiSlice } from './apiSlice';

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  role: 'user' | 'admin';
  createdAt: string;
  updatedAt: string;
  preferences: {
    theme: 'light' | 'dark' | 'system';
    notifications: boolean;
    language: string;
  };
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  name: string;
}

export interface AuthResponse {
  user: User;
  token: string;
}

export interface UpdateProfileRequest {
  name?: string;
  avatar?: File;
  preferences?: Partial<User['preferences']>;
}

export interface ChangePasswordRequest {
  currentPassword: string;
  newPassword: string;
}

export const authApi = apiSlice.injectEndpoints({
  endpoints: (builder) => ({
    login: builder.mutation<AuthResponse, LoginRequest>({
      query: (credentials) => ({
        url: 'auth/login',
        method: 'POST',
        body: credentials,
      }),
    }),

    register: builder.mutation<AuthResponse, RegisterRequest>({
      query: (userData) => ({
        url: 'auth/register',
        method: 'POST',
        body: userData,
      }),
    }),

    logout: builder.mutation<void, void>({
      query: () => ({
        url: 'auth/logout',
        method: 'POST',
      }),
    }),

    getCurrentUser: builder.query<User, void>({
      query: () => 'auth/me',
    }),

    updateProfile: builder.mutation<User, UpdateProfileRequest>({
      query: (data) => {
        const formData = new FormData();
        if (data.name) formData.append('name', data.name);
        if (data.avatar) formData.append('avatar', data.avatar);
        if (data.preferences) formData.append('preferences', JSON.stringify(data.preferences));

        return {
          url: 'auth/profile',
          method: 'PATCH',
          body: formData,
        };
      },
    }),

    changePassword: builder.mutation<void, ChangePasswordRequest>({
      query: (passwords) => ({
        url: 'auth/password',
        method: 'POST',
        body: passwords,
      }),
    }),

    requestPasswordReset: builder.mutation<void, { email: string }>({
      query: (body) => ({
        url: 'auth/password/reset',
        method: 'POST',
        body,
      }),
    }),

    resetPassword: builder.mutation<void, { token: string; password: string }>({
      query: (body) => ({
        url: 'auth/password/reset/confirm',
        method: 'POST',
        body,
      }),
    }),
  }),
});

export const {
  useLoginMutation,
  useRegisterMutation,
  useLogoutMutation,
  useGetCurrentUserQuery,
  useUpdateProfileMutation,
  useChangePasswordMutation,
  useRequestPasswordResetMutation,
  useResetPasswordMutation,
} = authApi; 