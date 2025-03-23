import axios, { AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';
import { getToken, removeToken } from '../auth/authUtils';

// Create base API instance
export const baseAPI = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor
baseAPI.interceptors.request.use(
  (config) => {
    const token = getToken();
    
    // If token exists, add to headers
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add a response interceptor
baseAPI.interceptors.response.use(
  (response) => {
    return response;
  },
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Handle unauthorized errors (e.g., token expired)
      removeToken();
      // Redirect to login if on client-side
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
    }
    
    return Promise.reject(error);
  }
);

/**
 * Interface for API error responses
 */
export interface ApiErrorResponse {
  status: number;
  message: string;
  errors?: Record<string, string[]>;
}

interface ApiResponseData {
  message?: string;
  errors?: Record<string, string[]>;
}

/**
 * Format error from API response
 */
export function formatApiError(error: unknown): ApiErrorResponse {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ApiResponseData>;
    
    if (axiosError.response) {
      return {
        status: axiosError.response.status,
        message: axiosError.response.data?.message || axiosError.message,
        errors: axiosError.response.data?.errors,
      };
    }
    
    return {
      status: 500,
      message: axiosError.message,
    };
  }
  
  return {
    status: 500,
    message: error instanceof Error ? error.message : 'An unknown error occurred',
  };
}

/**
 * Upload file to the API with progress tracking
 */
export async function uploadFile(
  url: string,
  file: File,
  onProgress?: (percentage: number) => void
): Promise<AxiosResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const config: AxiosRequestConfig = {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const percentage = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(percentage);
      }
    },
  };
  
  return baseAPI.post(url, formData, config);
}

/**
 * Check if API error has validation errors for a specific field
 */
export function hasFieldError(
  error: ApiErrorResponse | null,
  field: string
): boolean {
  return Boolean(error?.errors && error.errors[field] && error.errors[field].length > 0);
}

/**
 * Get validation error message for a specific field
 */
export function getFieldErrorMessage(
  error: ApiErrorResponse | null,
  field: string
): string | null {
  if (hasFieldError(error, field)) {
    return error?.errors?.[field][0] || null;
  }
  
  return null;
} 