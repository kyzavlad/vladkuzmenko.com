import { Page } from '@playwright/test';
import path from 'path';
import fs from 'fs';
import axios from 'axios';

export async function generateTestVideo(page: Page, authToken: string) {
  // Upload test video
  const testVideoPath = path.join(__dirname, '../fixtures/test-video.mp4');
  const formData = new FormData();
  formData.append('file', fs.createReadStream(testVideoPath));

  const api = axios.create({
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    headers: { Authorization: `Bearer ${authToken}` }
  });

  const response = await api.post('/api/videos/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });

  return {
    videoId: response.data.videoId,
    uploadUrl: response.data.uploadUrl
  };
}

export async function createTestAvatar(page: Page, authToken: string) {
  // Create test avatar
  const api = axios.create({
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    headers: { Authorization: `Bearer ${authToken}` }
  });

  const response = await api.post('/api/avatar/create', {
    name: 'Test Avatar',
    style: 'professional',
    settings: {
      voice: 'natural',
      appearance: 'formal',
      background: 'office'
    }
  });

  return {
    avatarId: response.data.avatarId,
    settings: response.data.settings
  };
}

export async function mockTokenBalance(authToken: string, balance: number) {
  const api = axios.create({
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    headers: { Authorization: `Bearer ${authToken}` }
  });

  // Reset token balance for testing
  await api.post('/api/tokens/reset', { balance });
}

export async function waitForJobCompletion(page: Page, jobType: string, jobId: string, timeout = 30000) {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeout) {
    const status = await page.locator(`[data-testid="${jobType}-status-${jobId}"]`).textContent();
    if (status === 'Completed') {
      return true;
    }
    await page.waitForTimeout(1000);
  }
  
  throw new Error(`Job ${jobId} did not complete within ${timeout}ms`);
}

export async function setupTestData(page: Page, authToken: string) {
  // Create test video
  const { videoId } = await generateTestVideo(page, authToken);

  // Create test avatar
  const { avatarId } = await createTestAvatar(page, authToken);

  // Create test translation
  const api = axios.create({
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    headers: { Authorization: `Bearer ${authToken}` }
  });

  const translationResponse = await api.post('/api/videos/translate', {
    videoId,
    sourceLang: 'en',
    targetLang: 'es'
  });

  return {
    videoId,
    avatarId,
    translationId: translationResponse.data.jobId
  };
}

export async function cleanupTestData(authToken: string, resources: {
  videoId?: string;
  avatarId?: string;
  translationId?: string;
}) {
  const api = axios.create({
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    headers: { Authorization: `Bearer ${authToken}` }
  });

  const deletePromises = [];

  if (resources.videoId) {
    deletePromises.push(api.delete(`/api/videos/${resources.videoId}`));
  }
  if (resources.avatarId) {
    deletePromises.push(api.delete(`/api/avatars/${resources.avatarId}`));
  }
  if (resources.translationId) {
    deletePromises.push(api.delete(`/api/translations/${resources.translationId}`));
  }

  await Promise.all(deletePromises);
} 