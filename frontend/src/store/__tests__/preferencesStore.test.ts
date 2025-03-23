import { configureStore } from '@reduxjs/toolkit';
import preferencesReducer, {
  updateTheme,
  updateInterfaceSettings,
  updateNotificationSettings,
  updatePrivacySettings,
  setLoading,
  setError,
  clearError,
  PreferencesState,
} from '../slices/preferencesSlice';
import {
  updateThemeThunk,
  updateInterfaceSettingsThunk,
  updateNotificationSettingsThunk,
  updatePrivacySettingsThunk,
} from '../thunks/preferencesThunks';
import { preferencesApi } from '../services/preferencesApi';

// Mock API service
jest.mock('../services/preferencesApi', () => ({
  preferencesApi: {
    updateTheme: jest.fn(),
    updateInterfaceSettings: jest.fn(),
    updateNotificationSettings: jest.fn(),
    updatePrivacySettings: jest.fn(),
  },
}));

describe('Preferences Store', () => {
  let store: ReturnType<typeof configureStore>;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        preferences: preferencesReducer,
      },
    });
    jest.clearAllMocks();
  });

  describe('Synchronous Actions', () => {
    it('should update theme', () => {
      const theme = {
        mode: 'dark',
        primaryColor: '#805AD5',
        fontSize: 'large',
      };
      store.dispatch(updateTheme(theme));
      
      expect(store.getState().preferences.theme).toEqual(theme);
    });

    it('should update interface settings', () => {
      const settings = {
        sidebarCollapsed: true,
        enableAnimations: false,
        keyboardShortcuts: true,
      };
      store.dispatch(updateInterfaceSettings(settings));
      
      expect(store.getState().preferences.interface).toEqual(settings);
    });

    it('should update notification settings', () => {
      const settings = {
        email: false,
        push: true,
        processingUpdates: true,
        marketingEmails: false,
      };
      store.dispatch(updateNotificationSettings(settings));
      
      expect(store.getState().preferences.notifications).toEqual(settings);
    });

    it('should update privacy settings', () => {
      const settings = {
        shareUsageData: false,
        autoSaveEnabled: true,
        storageRetentionDays: 60,
      };
      store.dispatch(updatePrivacySettings(settings));
      
      expect(store.getState().preferences.privacy).toEqual(settings);
    });

    it('should set loading state', () => {
      store.dispatch(setLoading(true));
      expect(store.getState().preferences.loading).toBe(true);
      
      store.dispatch(setLoading(false));
      expect(store.getState().preferences.loading).toBe(false);
    });

    it('should set and clear error', () => {
      const error = 'Test error';
      store.dispatch(setError(error));
      expect(store.getState().preferences.error).toBe(error);
      
      store.dispatch(clearError());
      expect(store.getState().preferences.error).toBeNull();
    });
  });

  describe('Async Actions', () => {
    describe('updateThemeThunk', () => {
      const theme = {
        mode: 'dark',
        primaryColor: '#805AD5',
        fontSize: 'large',
      };

      it('should update theme on successful API call', async () => {
        (preferencesApi.updateTheme as jest.Mock).mockResolvedValueOnce(theme);

        await store.dispatch(updateThemeThunk(theme));
        const state = store.getState().preferences;

        expect(state.theme).toEqual(theme);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
      });

      it('should handle update error', async () => {
        const error = new Error('Failed to update theme');
        (preferencesApi.updateTheme as jest.Mock).mockRejectedValueOnce(error);

        await store.dispatch(updateThemeThunk(theme));
        const state = store.getState().preferences;

        expect(state.loading).toBe(false);
        expect(state.error).toBe(error.message);
      });
    });

    describe('updateInterfaceSettingsThunk', () => {
      const settings = {
        sidebarCollapsed: true,
        enableAnimations: false,
        keyboardShortcuts: true,
      };

      it('should update interface settings on successful API call', async () => {
        (preferencesApi.updateInterfaceSettings as jest.Mock).mockResolvedValueOnce(settings);

        await store.dispatch(updateInterfaceSettingsThunk(settings));
        const state = store.getState().preferences;

        expect(state.interface).toEqual(settings);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
      });

      it('should handle update error', async () => {
        const error = new Error('Failed to update interface settings');
        (preferencesApi.updateInterfaceSettings as jest.Mock).mockRejectedValueOnce(error);

        await store.dispatch(updateInterfaceSettingsThunk(settings));
        const state = store.getState().preferences;

        expect(state.loading).toBe(false);
        expect(state.error).toBe(error.message);
      });
    });

    describe('updateNotificationSettingsThunk', () => {
      const settings = {
        email: false,
        push: true,
        processingUpdates: true,
        marketingEmails: false,
      };

      it('should update notification settings on successful API call', async () => {
        (preferencesApi.updateNotificationSettings as jest.Mock).mockResolvedValueOnce(settings);

        await store.dispatch(updateNotificationSettingsThunk(settings));
        const state = store.getState().preferences;

        expect(state.notifications).toEqual(settings);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
      });

      it('should handle update error', async () => {
        const error = new Error('Failed to update notification settings');
        (preferencesApi.updateNotificationSettings as jest.Mock).mockRejectedValueOnce(error);

        await store.dispatch(updateNotificationSettingsThunk(settings));
        const state = store.getState().preferences;

        expect(state.loading).toBe(false);
        expect(state.error).toBe(error.message);
      });
    });

    describe('updatePrivacySettingsThunk', () => {
      const settings = {
        shareUsageData: false,
        autoSaveEnabled: true,
        storageRetentionDays: 60,
      };

      it('should update privacy settings on successful API call', async () => {
        (preferencesApi.updatePrivacySettings as jest.Mock).mockResolvedValueOnce(settings);

        await store.dispatch(updatePrivacySettingsThunk(settings));
        const state = store.getState().preferences;

        expect(state.privacy).toEqual(settings);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
      });

      it('should handle update error', async () => {
        const error = new Error('Failed to update privacy settings');
        (preferencesApi.updatePrivacySettings as jest.Mock).mockRejectedValueOnce(error);

        await store.dispatch(updatePrivacySettingsThunk(settings));
        const state = store.getState().preferences;

        expect(state.loading).toBe(false);
        expect(state.error).toBe(error.message);
      });
    });
  });
}); 