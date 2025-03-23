import { configureStore } from '@reduxjs/toolkit';
import preferencesReducer from '../../slices/preferencesSlice';
import {
  updateThemeThunk,
  updateInterfaceSettingsThunk,
  updateNotificationSettingsThunk,
  updatePrivacySettingsThunk,
} from '../preferencesThunks';
import { preferencesApi } from '../../services/preferencesApi';

// Mock API responses
const mockApiResponses = {
  theme: {
    mode: 'dark',
    primaryColor: '#805AD5',
    fontSize: 'large',
  },
  interface: {
    sidebarCollapsed: true,
    enableAnimations: false,
    keyboardShortcuts: true,
  },
  notifications: {
    email: false,
    push: true,
    processingUpdates: true,
    marketingEmails: false,
  },
  privacy: {
    shareUsageData: false,
    autoSaveEnabled: true,
    storageRetentionDays: 60,
  },
};

// Mock API service
jest.mock('../../services/preferencesApi', () => ({
  preferencesApi: {
    updateTheme: jest.fn(),
    updateInterfaceSettings: jest.fn(),
    updateNotificationSettings: jest.fn(),
    updatePrivacySettings: jest.fn(),
  },
}));

describe('User Preferences Thunks', () => {
  let store: ReturnType<typeof configureStore>;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        preferences: preferencesReducer,
      },
    });
    jest.clearAllMocks();
  });

  describe('updateThemeThunk', () => {
    it('successfully updates theme settings', async () => {
      (preferencesApi.updateTheme as jest.Mock).mockResolvedValueOnce(mockApiResponses.theme);

      const result = await store.dispatch(updateThemeThunk(mockApiResponses.theme));
      const state = store.getState().preferences;

      expect(result.payload).toEqual(mockApiResponses.theme);
      expect(state.theme).toEqual(mockApiResponses.theme);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('handles theme update error', async () => {
      const error = new Error('Failed to update theme');
      (preferencesApi.updateTheme as jest.Mock).mockRejectedValueOnce(error);

      const result = await store.dispatch(updateThemeThunk(mockApiResponses.theme));
      const state = store.getState().preferences;

      expect(result.error?.message).toBe(error.message);
      expect(state.loading).toBe(false);
      expect(state.error).toBe(error.message);
    });
  });

  describe('updateInterfaceSettingsThunk', () => {
    it('successfully updates interface settings', async () => {
      (preferencesApi.updateInterfaceSettings as jest.Mock).mockResolvedValueOnce(mockApiResponses.interface);

      const result = await store.dispatch(updateInterfaceSettingsThunk(mockApiResponses.interface));
      const state = store.getState().preferences;

      expect(result.payload).toEqual(mockApiResponses.interface);
      expect(state.interface).toEqual(mockApiResponses.interface);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('handles interface settings update error', async () => {
      const error = new Error('Failed to update interface settings');
      (preferencesApi.updateInterfaceSettings as jest.Mock).mockRejectedValueOnce(error);

      const result = await store.dispatch(updateInterfaceSettingsThunk(mockApiResponses.interface));
      const state = store.getState().preferences;

      expect(result.error?.message).toBe(error.message);
      expect(state.loading).toBe(false);
      expect(state.error).toBe(error.message);
    });
  });

  describe('updateNotificationSettingsThunk', () => {
    it('successfully updates notification settings', async () => {
      (preferencesApi.updateNotificationSettings as jest.Mock).mockResolvedValueOnce(mockApiResponses.notifications);

      const result = await store.dispatch(updateNotificationSettingsThunk(mockApiResponses.notifications));
      const state = store.getState().preferences;

      expect(result.payload).toEqual(mockApiResponses.notifications);
      expect(state.notifications).toEqual(mockApiResponses.notifications);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('handles notification settings update error', async () => {
      const error = new Error('Failed to update notification settings');
      (preferencesApi.updateNotificationSettings as jest.Mock).mockRejectedValueOnce(error);

      const result = await store.dispatch(updateNotificationSettingsThunk(mockApiResponses.notifications));
      const state = store.getState().preferences;

      expect(result.error?.message).toBe(error.message);
      expect(state.loading).toBe(false);
      expect(state.error).toBe(error.message);
    });
  });

  describe('updatePrivacySettingsThunk', () => {
    it('successfully updates privacy settings', async () => {
      (preferencesApi.updatePrivacySettings as jest.Mock).mockResolvedValueOnce(mockApiResponses.privacy);

      const result = await store.dispatch(updatePrivacySettingsThunk(mockApiResponses.privacy));
      const state = store.getState().preferences;

      expect(result.payload).toEqual(mockApiResponses.privacy);
      expect(state.privacy).toEqual(mockApiResponses.privacy);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('handles privacy settings update error', async () => {
      const error = new Error('Failed to update privacy settings');
      (preferencesApi.updatePrivacySettings as jest.Mock).mockRejectedValueOnce(error);

      const result = await store.dispatch(updatePrivacySettingsThunk(mockApiResponses.privacy));
      const state = store.getState().preferences;

      expect(result.error?.message).toBe(error.message);
      expect(state.loading).toBe(false);
      expect(state.error).toBe(error.message);
    });
  });
}); 