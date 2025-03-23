import { configureStore } from '@reduxjs/toolkit';

export const store = configureStore({
  reducer: {
    // Здесь будут редьюсеры
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch; 