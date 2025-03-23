import { configureStore } from '@reduxjs/toolkit';
import tokenReducer from './slices/tokenSlice';
import preferencesReducer from './slices/preferencesSlice';

export const store = configureStore({
  reducer: {
    tokens: tokenReducer,
    preferences: preferencesReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types for serialization checks
        ignoredActions: ['preferences/update/fulfilled'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch; 