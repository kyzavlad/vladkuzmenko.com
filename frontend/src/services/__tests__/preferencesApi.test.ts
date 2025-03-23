import { preferencesApi } from '../preferencesApi';
import {
  ThemeSettings,
  InterfaceSettings,
  NotificationSettings,
  PrivacySettings,
} from '../../types/preferences';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('Preferences API Service', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe('updateTheme', () => {
    const mockTheme: ThemeSettings = {
      mode: 'dark',
      primaryColor: '#805AD5',
      fontSize: 'large',
    };

    it('successfully updates theme settings', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockTheme),
      });

      const result = await preferencesApi.updateTheme(mockTheme);
      
      expect(result).toEqual(mockTheme);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/preferences/theme'),
        expect.objectContaining({
          method: 'PUT',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(mockTheme),
        })
      );
    });

    it('handles update error', async () => {
      const errorMessage = 'Failed to update theme';
      mockFetch.mockRejectedValueOnce(new Error(errorMessage));

      await expect(preferencesApi.updateTheme(mockTheme)).rejects.toThrow(errorMessage);
    });
  });

  describe('updateInterfaceSettings', () => {
    const mockSettings: InterfaceSettings = {
      sidebarCollapsed: true,
      enableAnimations: false,
      keyboardShortcuts: true,
    };

    it('successfully updates interface settings', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSettings),
      });

      const result = await preferencesApi.updateInterfaceSettings(mockSettings);
      
      expect(result).toEqual(mockSettings);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/preferences/interface'),
        expect.objectContaining({
          method: 'PUT',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(mockSettings),
        })
      );
    });

    it('handles validation error', async () => {
      const errorResponse = { message: 'Invalid settings' };
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve(errorResponse),
      });

      await expect(preferencesApi.updateInterfaceSettings(mockSettings))
        .rejects.toThrow(errorResponse.message);
    });
  });

  describe('updateNotificationSettings', () => {
    const mockSettings: NotificationSettings = {
      email: false,
      push: true,
      processingUpdates: true,
      marketingEmails: false,
    };

    it('successfully updates notification settings', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSettings),
      });

      const result = await preferencesApi.updateNotificationSettings(mockSettings);
      
      expect(result).toEqual(mockSettings);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/preferences/notifications'),
        expect.objectContaining({
          method: 'PUT',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(mockSettings),
        })
      );
    });

    it('handles unauthorized error', async () => {
      const errorResponse = { message: 'Unauthorized' };
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve(errorResponse),
      });

      await expect(preferencesApi.updateNotificationSettings(mockSettings))
        .rejects.toThrow(errorResponse.message);
    });
  });

  describe('updatePrivacySettings', () => {
    const mockSettings: PrivacySettings = {
      shareUsageData: false,
      autoSaveEnabled: true,
      storageRetentionDays: 60,
    };

    it('successfully updates privacy settings', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSettings),
      });

      const result = await preferencesApi.updatePrivacySettings(mockSettings);
      
      expect(result).toEqual(mockSettings);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/preferences/privacy'),
        expect.objectContaining({
          method: 'PUT',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(mockSettings),
        })
      );
    });

    it('handles server error', async () => {
      const errorResponse = { message: 'Internal server error' };
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve(errorResponse),
      });

      await expect(preferencesApi.updatePrivacySettings(mockSettings))
        .rejects.toThrow(errorResponse.message);
    });
  });
}); 