import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

interface TestEnvironment {
  authToken: string;
  userId: string;
  cleanup: () => Promise<void>;
}

export async function setupTestEnvironment(): Promise<TestEnvironment> {
  const api = axios.create({
    baseURL: process.env.BASE_URL || 'http://localhost:3000'
  });

  // Create a test user
  const userId = uuidv4();
  const response = await api.post('/api/auth/register', {
    email: `test-${userId}@example.com`,
    password: 'Test123!@#',
    name: 'Test User'
  });

  const authToken = response.data.token;

  // Initialize test data
  const authenticatedApi = axios.create({
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    headers: { Authorization: `Bearer ${authToken}` }
  });

  // Add initial tokens
  await authenticatedApi.post('/api/tokens/add', {
    amount: 1000,
    reason: 'test-setup'
  });

  const cleanup = async () => {
    try {
      // Clean up test data
      await authenticatedApi.delete('/api/user/account');
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  };

  return {
    authToken,
    userId,
    cleanup
  };
}

export async function cleanupTestEnvironment(authToken: string): Promise<void> {
  const api = axios.create({
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    headers: { Authorization: `Bearer ${authToken}` }
  });

  try {
    // Get all test resources
    const [videos, avatars, translations] = await Promise.all([
      api.get('/api/videos').then(res => res.data),
      api.get('/api/avatars').then(res => res.data),
      api.get('/api/translations').then(res => res.data)
    ]);

    // Delete all test resources
    await Promise.all([
      ...videos.map((video: any) => api.delete(`/api/videos/${video.id}`)),
      ...avatars.map((avatar: any) => api.delete(`/api/avatars/${avatar.id}`)),
      ...translations.map((translation: any) => api.delete(`/api/translations/${translation.id}`))
    ]);

    // Delete test account
    await api.delete('/api/user/account');
  } catch (error) {
    console.error('Error during environment cleanup:', error);
    throw error;
  }
} 