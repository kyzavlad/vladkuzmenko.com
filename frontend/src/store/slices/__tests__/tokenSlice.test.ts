import { configureStore } from '@reduxjs/toolkit';
import tokenReducer, {
  fetchTokenBalance,
  fetchUsageHistory,
  purchaseTokens,
  redeemPromoCode,
  TokenState,
} from '../tokenSlice';

const mockInitialState: TokenState = {
  balance: 0,
  usageHistory: [],
  loading: false,
  error: null,
};

describe('Token Slice', () => {
  let store: ReturnType<typeof configureStore>;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        token: tokenReducer,
      },
    });
  });

  it('should handle initial state', () => {
    expect(store.getState().token).toEqual(mockInitialState);
  });

  describe('fetchTokenBalance', () => {
    it('should update balance on successful fetch', async () => {
      const mockBalance = 100;
      const mockResponse = { balance: mockBalance };
      
      jest.spyOn(global, 'fetch').mockImplementationOnce(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        } as Response)
      );

      await store.dispatch(fetchTokenBalance());
      const state = store.getState().token;

      expect(state.balance).toBe(mockBalance);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('should handle fetch error', async () => {
      const errorMessage = 'Failed to fetch balance';
      jest.spyOn(global, 'fetch').mockImplementationOnce(() =>
        Promise.reject(new Error(errorMessage))
      );

      await store.dispatch(fetchTokenBalance());
      const state = store.getState().token;

      expect(state.loading).toBe(false);
      expect(state.error).toBe(errorMessage);
    });
  });

  describe('purchaseTokens', () => {
    it('should update balance after successful purchase', async () => {
      const amount = 50;
      const mockResponse = { balance: mockInitialState.balance + amount };
      
      jest.spyOn(global, 'fetch').mockImplementationOnce(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        } as Response)
      );

      await store.dispatch(purchaseTokens({ amount }));
      const state = store.getState().token;

      expect(state.balance).toBe(mockResponse.balance);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('redeemPromoCode', () => {
    it('should update balance after successful promo code redemption', async () => {
      const code = 'PROMO123';
      const bonusTokens = 25;
      const mockResponse = { balance: mockInitialState.balance + bonusTokens };
      
      jest.spyOn(global, 'fetch').mockImplementationOnce(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        } as Response)
      );

      await store.dispatch(redeemPromoCode({ code }));
      const state = store.getState().token;

      expect(state.balance).toBe(mockResponse.balance);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('fetchUsageHistory', () => {
    it('should update usage history on successful fetch', async () => {
      const mockUsageHistory = [
        { id: '1', amount: 10, description: 'Video processing', timestamp: new Date().toISOString() },
      ];
      const mockResponse = { usageHistory: mockUsageHistory };
      
      jest.spyOn(global, 'fetch').mockImplementationOnce(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockResponse),
        } as Response)
      );

      await store.dispatch(fetchUsageHistory());
      const state = store.getState().token;

      expect(state.usageHistory).toEqual(mockUsageHistory);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });
  });
}); 