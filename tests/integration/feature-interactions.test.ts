import { test, expect } from '@playwright/test';
import { setupTestEnvironment, cleanupTestEnvironment } from '../utils/test-setup';
import { generateTestVideo, createTestAvatar, mockTokenBalance } from '../utils/test-helpers';

test.describe('Feature Interactions', () => {
  let authToken: string;
  let testVideoId: string;
  let testAvatarId: string;

  test.beforeAll(async () => {
    const setup = await setupTestEnvironment();
    authToken = setup.authToken;
    await mockTokenBalance(authToken, 1000); // Ensure sufficient tokens
  });

  test.afterAll(async () => {
    await cleanupTestEnvironment(authToken);
  });

  test('Video Editor to Translation Pipeline', async ({ page }) => {
    // 1. Upload and edit video
    const { videoId } = await generateTestVideo(page, authToken);
    testVideoId = videoId;
    
    // 2. Verify video processing
    await page.goto('/platform/videos/${videoId}');
    await expect(page.locator('[data-testid="video-status"]')).toHaveText('Processed');
    
    // 3. Initiate translation
    await page.click('[data-testid="translate-button"]');
    await page.selectOption('[data-testid="target-language"]', 'es');
    await page.click('[data-testid="start-translation"]');
    
    // 4. Verify translation job
    await expect(page.locator('[data-testid="translation-status"]')).toHaveText('In Progress');
    await page.waitForSelector('[data-testid="translation-complete"]', { timeout: 30000 });
  });

  test('Avatar Creation to Video Integration', async ({ page }) => {
    // 1. Create avatar
    const { avatarId } = await createTestAvatar(page, authToken);
    testAvatarId = avatarId;
    
    // 2. Verify avatar creation
    await page.goto('/platform/avatars/${avatarId}');
    await expect(page.locator('[data-testid="avatar-status"]')).toHaveText('Ready');
    
    // 3. Use avatar in video
    await page.goto('/platform/videos/${testVideoId}');
    await page.click('[data-testid="add-avatar"]');
    await page.click(`[data-avatar-id="${testAvatarId}"]`);
    
    // 4. Verify avatar integration
    await expect(page.locator('[data-testid="avatar-layer"]')).toBeVisible();
  });

  test('Clip Generation with Translation', async ({ page }) => {
    // 1. Generate clip from translated video
    await page.goto('/platform/videos/${testVideoId}');
    await page.click('[data-testid="generate-clip"]');
    await page.fill('[data-testid="clip-duration"]', '30');
    await page.click('[data-testid="start-generation"]');
    
    // 2. Verify clip generation
    await page.waitForSelector('[data-testid="clip-ready"]', { timeout: 30000 });
    
    // 3. Verify translated audio in clip
    await page.click('[data-testid="preview-clip"]');
    await expect(page.locator('[data-testid="audio-language"]')).toHaveText('Spanish');
  });

  test('Token Management Across Features', async ({ page }) => {
    // 1. Check initial balance
    await page.goto('/platform/tokens');
    const initialBalance = await page.locator('[data-testid="token-balance"]').innerText();
    
    // 2. Perform operations
    await generateTestVideo(page, authToken);
    await createTestAvatar(page, authToken);
    
    // 3. Verify token deduction
    await page.reload();
    const newBalance = await page.locator('[data-testid="token-balance"]').innerText();
    expect(parseInt(newBalance)).toBeLessThan(parseInt(initialBalance));
  });

  test('Real-time Updates Across Features', async ({ page }) => {
    // 1. Start multiple operations
    const videoPromise = generateTestVideo(page, authToken);
    const avatarPromise = createTestAvatar(page, authToken);
    
    // 2. Open activity feed
    await page.goto('/platform');
    
    // 3. Verify real-time updates
    await expect(page.locator('[data-testid="activity-feed"]')).toContainText('Video Processing');
    await expect(page.locator('[data-testid="activity-feed"]')).toContainText('Avatar Creation');
    
    // 4. Wait for completion
    await Promise.all([videoPromise, avatarPromise]);
    await page.reload();
    await expect(page.locator('[data-testid="activity-feed"]')).toContainText('Completed');
  });

  test('User Settings Persistence', async ({ page }) => {
    // 1. Update settings
    await page.goto('/platform/settings');
    await page.click('[data-testid="dark-mode"]');
    await page.selectOption('[data-testid="default-language"]', 'es');
    await page.click('[data-testid="save-settings"]');
    
    // 2. Verify persistence across features
    await page.goto('/platform/videos');
    await expect(page.locator('body')).toHaveClass(/dark/);
    await expect(page.locator('[data-testid="language-selector"]')).toHaveValue('es');
  });
}); 