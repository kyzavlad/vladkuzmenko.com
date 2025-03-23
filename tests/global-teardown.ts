import { FullConfig } from '@playwright/test';

async function globalTeardown(config: FullConfig) {
  // Run cleanup function if it exists
  if (global.__CLEANUP__) {
    await global.__CLEANUP__();
  }

  // Clear environment variables
  delete process.env.TEST_AUTH_TOKEN;
  delete process.env.TEST_USER_ID;
}

export default globalTeardown; 