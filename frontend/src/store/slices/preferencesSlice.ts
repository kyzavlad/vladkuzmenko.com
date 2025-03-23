import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import axios from 'axios';

interface NotificationSettings {
  email: {
    processComplete: boolean;
    lowBalance: boolean;
    newsletter: boolean;
    productUpdates: boolean;
  };
  inApp: {
    processComplete: boolean;
    lowBalance: boolean;
    tips: boolean;
  };
}

interface PrivacySettings {
  shareUsageData: boolean;
  shareAnalytics: boolean;
  marketingCommunication: boolean;
}

interface ThemeSettings {
  mode: 'light' | 'dark' | 'system';
  primaryColor: string;
  fontSize: 'small' | 'medium' | 'large';
}

interface InterfaceSettings {
  defaultView: 'grid' | 'list';
  compactMode: boolean;
  showTutorials: boolean;
  enableKeyboardShortcuts: boolean;
  customShortcuts: Record<string, string>;
}

interface UserPreferences {
  theme: ThemeSettings;
  interface: InterfaceSettings;
  notifications: NotificationSettings;
  privacy: PrivacySettings;
  lastSynced: string | null;
}

interface PreferencesState extends UserPreferences {
  isLoading: boolean;
  error: string | null;
}

const initialState: PreferencesState = {
  theme: {
    mode: 'system',
    primaryColor: '#3182CE',
    fontSize: 'medium',
  },
  interface: {
    defaultView: 'grid',
    compactMode: false,
    showTutorials: true,
    enableKeyboardShortcuts: true,
    customShortcuts: {},
  },
  notifications: {
    email: {
      processComplete: true,
      lowBalance: true,
      newsletter: false,
      productUpdates: true,
    },
    inApp: {
      processComplete: true,
      lowBalance: true,
      tips: true,
    },
  },
  privacy: {
    shareUsageData: true,
    shareAnalytics: true,
    marketingCommunication: false,
  },
  lastSynced: null,
  isLoading: false,
  error: null,
};

export const fetchPreferences = createAsyncThunk(
  'preferences/fetch',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: { token: string } };
      const response = await axios.get('/api/user/preferences', {
        headers: { Authorization: `Bearer ${state.auth.token}` },
      });
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch preferences');
    }
  }
);

export const updatePreferences = createAsyncThunk(
  'preferences/update',
  async (preferences: Partial<UserPreferences>, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: { token: string } };
      const response = await axios.patch(
        '/api/user/preferences',
        preferences,
        { headers: { Authorization: `Bearer ${state.auth.token}` } }
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to update preferences');
    }
  }
);

const preferencesSlice = createSlice({
  name: 'preferences',
  initialState,
  reducers: {
    updateTheme: (state, action: PayloadAction<Partial<ThemeSettings>>) => {
      state.theme = { ...state.theme, ...action.payload };
    },
    updateInterface: (state, action: PayloadAction<Partial<InterfaceSettings>>) => {
      state.interface = { ...state.interface, ...action.payload };
    },
    updateNotifications: (state, action: PayloadAction<Partial<NotificationSettings>>) => {
      state.notifications = {
        email: { ...state.notifications.email, ...action.payload.email },
        inApp: { ...state.notifications.inApp, ...action.payload.inApp },
      };
    },
    updatePrivacy: (state, action: PayloadAction<Partial<PrivacySettings>>) => {
      state.privacy = { ...state.privacy, ...action.payload };
    },
    resetToDefaults: (state) => {
      return { ...initialState, lastSynced: state.lastSynced };
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch Preferences
      .addCase(fetchPreferences.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchPreferences.fulfilled, (state, action) => {
        state.isLoading = false;
        state.theme = action.payload.theme;
        state.interface = action.payload.interface;
        state.notifications = action.payload.notifications;
        state.privacy = action.payload.privacy;
        state.lastSynced = new Date().toISOString();
      })
      .addCase(fetchPreferences.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      // Update Preferences
      .addCase(updatePreferences.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(updatePreferences.fulfilled, (state, action) => {
        state.isLoading = false;
        state.lastSynced = new Date().toISOString();
      })
      .addCase(updatePreferences.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });
  },
});

export const {
  updateTheme,
  updateInterface,
  updateNotifications,
  updatePrivacy,
  resetToDefaults,
} = preferencesSlice.actions;

export default preferencesSlice.reducer; 