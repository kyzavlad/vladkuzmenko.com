import { configureStore } from '@reduxjs/toolkit';
import preferencesReducer, {
  updateTheme,
  updateInterfaceSettings,
  updateNotificationSettings,
  updatePrivacySettings,
  PreferencesState,
} from '../preferencesSlice';

const mockInitialState: PreferencesState = {
  theme: {
    mode: 'light',
    primaryColor: '#3182CE',
    fontSize: 'medium',
  },
  interface: {
    sidebarCollapsed: false,
    enableAnimations: true,
    keyboardShortcuts: true,
  },
  notifications: {
    email: true,
    push: true,
    processingUpdates: true,
    marketingEmails: false,
  },
  privacy: {
    shareUsageData: true,
    autoSaveEnabled: true,
    storageRetentionDays: 30,
  },
  loading: false,
  error: null,
};

describe('Preferences Slice', () => {
  let store: ReturnType<typeof configureStore>;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        preferences: preferencesReducer,
      },
    });
  });

  it('should handle initial state', () => {
    expect(store.getState().preferences).toEqual(mockInitialState);
  });

  describe('updateTheme', () => {
    it('should update theme settings', async () => {
      const newTheme = {
        mode: 'dark',
        primaryColor: '#805AD5',
        fontSize: 'large',
      };

      await store.dispatch(updateTheme(newTheme));
      const state = store.getState().preferences;

      expect(state.theme).toEqual(newTheme);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('updateInterfaceSettings', () => {
    it('should update interface settings', async () => {
      const newSettings = {
        sidebarCollapsed: true,
        enableAnimations: false,
        keyboardShortcuts: true,
      };

      await store.dispatch(updateInterfaceSettings(newSettings));
      const state = store.getState().preferences;

      expect(state.interface).toEqual(newSettings);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('updateNotificationSettings', () => {
    it('should update notification settings', async () => {
      const newSettings = {
        email: false,
        push: true,
        processingUpdates: true,
        marketingEmails: true,
      };

      await store.dispatch(updateNotificationSettings(newSettings));
      const state = store.getState().preferences;

      expect(state.notifications).toEqual(newSettings);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('updatePrivacySettings', () => {
    it('should update privacy settings', async () => {
      const newSettings = {
        shareUsageData: false,
        autoSaveEnabled: true,
        storageRetentionDays: 60,
      };

      await store.dispatch(updatePrivacySettings(newSettings));
      const state = store.getState().preferences;

      expect(state.privacy).toEqual(newSettings);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('should handle update error', async () => {
      const errorMessage = 'Failed to update privacy settings';
      jest.spyOn(global, 'fetch').mockImplementationOnce(() =>
        Promise.reject(new Error(errorMessage))
      );

      await store.dispatch(updatePrivacySettings({
        shareUsageData: false,
        autoSaveEnabled: true,
        storageRetentionDays: 60,
      }));
      
      const state = store.getState().preferences;
      expect(state.loading).toBe(false);
      expect(state.error).toBe(errorMessage);
    });
  });
}); 