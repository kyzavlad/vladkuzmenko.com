import { FullConfig } from '@playwright/test';
import { setupTestEnvironment } from './utils/test-setup';

async function globalSetup(config: FullConfig) {
  // Set up test environment
  const env = await setupTestEnvironment();
  
  // Store auth token for tests
  process.env.TEST_AUTH_TOKEN = env.authToken;
  process.env.TEST_USER_ID = env.userId;

  // Store cleanup function
  global.__CLEANUP__ = env.cleanup;
}

export default globalSetup; 