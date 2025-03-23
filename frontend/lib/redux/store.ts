import { configureStore } from '@reduxjs/toolkit';
import { setupListeners } from '@reduxjs/toolkit/query';
import authReducer from './slices/authSlice';
import videoReducer from './slices/videoSlice';
import avatarReducer from './slices/avatarSlice';
import translationReducer from './slices/translationSlice';
import { apiSlice } from './services/apiSlice';

export const store = configureStore({
  reducer: {
    auth: authReducer,
    video: videoReducer,
    avatar: avatarReducer,
    translation: translationReducer,
    [apiSlice.reducerPath]: apiSlice.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(apiSlice.middleware),
  devTools: process.env.NODE_ENV !== 'production',
});

setupListeners(store.dispatch);

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch; 