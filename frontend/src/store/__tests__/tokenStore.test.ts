import { configureStore } from '@reduxjs/toolkit';
import tokenReducer, {
  setBalance,
  addUsage,
  setLoading,
  setError,
  clearError,
  TokenState,
} from '../slices/tokenSlice';
import {
  fetchTokenBalance,
  purchaseTokens,
  redeemPromoCode,
  fetchUsageHistory,
} from '../thunks/tokenThunks';
import { tokenApi } from '../services/tokenApi';

// Mock API service
jest.mock('../services/tokenApi', () => ({
  tokenApi: {
    getBalance: jest.fn(),
    purchaseTokens: jest.fn(),
    redeemPromoCode: jest.fn(),
    getUsageHistory: jest.fn(),
  },
}));

describe('Token Store', () => {
  let store: ReturnType<typeof configureStore>;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        token: tokenReducer,
      },
    });
    jest.clearAllMocks();
  });

  describe('Synchronous Actions', () => {
    it('should set balance', () => {
      const balance = 100;
      store.dispatch(setBalance(balance));
      
      expect(store.getState().token.balance).toBe(balance);
    });

    it('should add usage record', () => {
      const usage = {
        id: '1',
        amount: -10,
        description: 'Video processing',
        timestamp: new Date().toISOString(),
      };
      store.dispatch(addUsage(usage));
      
      expect(store.getState().token.usageHistory).toContainEqual(usage);
    });

    it('should set loading state', () => {
      store.dispatch(setLoading(true));
      expect(store.getState().token.loading).toBe(true);
      
      store.dispatch(setLoading(false));
      expect(store.getState().token.loading).toBe(false);
    });

    it('should set and clear error', () => {
      const error = 'Test error';
      store.dispatch(setError(error));
      expect(store.getState().token.error).toBe(error);
      
      store.dispatch(clearError());
      expect(store.getState().token.error).toBeNull();
    });
  });

  describe('Async Actions', () => {
    describe('fetchTokenBalance', () => {
      it('should update state on successful fetch', async () => {
        const mockBalance = { balance: 100 };
        (tokenApi.getBalance as jest.Mock).mockResolvedValueOnce(mockBalance);

        await store.dispatch(fetchTokenBalance());
        const state = store.getState().token;

        expect(state.balance).toBe(mockBalance.balance);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
      });

      it('should handle fetch error', async () => {
        const error = new Error('Failed to fetch balance');
        (tokenApi.getBalance as jest.Mock).mockRejectedValueOnce(error);

        await store.dispatch(fetchTokenBalance());
        const state = store.getState().token;

        expect(state.loading).toBe(false);
        expect(state.error).toBe(error.message);
      });
    });

    describe('purchaseTokens', () => {
      const purchaseAmount = 50;

      it('should update balance on successful purchase', async () => {
        const mockResponse = { balance: 150, transactionId: 'tx123' };
        (tokenApi.purchaseTokens as jest.Mock).mockResolvedValueOnce(mockResponse);

        await store.dispatch(purchaseTokens({ amount: purchaseAmount }));
        const state = store.getState().token;

        expect(state.balance).toBe(mockResponse.balance);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
      });

      it('should handle purchase error', async () => {
        const error = new Error('Purchase failed');
        (tokenApi.purchaseTokens as jest.Mock).mockRejectedValueOnce(error);

        await store.dispatch(purchaseTokens({ amount: purchaseAmount }));
        const state = store.getState().token;

        expect(state.loading).toBe(false);
        expect(state.error).toBe(error.message);
      });
    });

    describe('redeemPromoCode', () => {
      const promoCode = 'BONUS50';

      it('should update balance on successful redemption', async () => {
        const mockResponse = { balance: 200, bonusAmount: 50 };
        (tokenApi.redeemPromoCode as jest.Mock).mockResolvedValueOnce(mockResponse);

        await store.dispatch(redeemPromoCode({ code: promoCode }));
        const state = store.getState().token;

        expect(state.balance).toBe(mockResponse.balance);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
      });

      it('should handle redemption error', async () => {
        const error = new Error('Invalid promo code');
        (tokenApi.redeemPromoCode as jest.Mock).mockRejectedValueOnce(error);

        await store.dispatch(redeemPromoCode({ code: promoCode }));
        const state = store.getState().token;

        expect(state.loading).toBe(false);
        expect(state.error).toBe(error.message);
      });
    });

    describe('fetchUsageHistory', () => {
      it('should update history on successful fetch', async () => {
        const mockHistory = {
          history: [
            {
              id: '1',
              amount: -10,
              description: 'Video processing',
              timestamp: new Date().toISOString(),
            },
          ],
        };
        (tokenApi.getUsageHistory as jest.Mock).mockResolvedValueOnce(mockHistory);

        await store.dispatch(fetchUsageHistory());
        const state = store.getState().token;

        expect(state.usageHistory).toEqual(mockHistory.history);
        expect(state.loading).toBe(false);
        expect(state.error).toBeNull();
      });

      it('should handle fetch error', async () => {
        const error = new Error('Failed to fetch history');
        (tokenApi.getUsageHistory as jest.Mock).mockRejectedValueOnce(error);

        await store.dispatch(fetchUsageHistory());
        const state = store.getState().token;

        expect(state.loading).toBe(false);
        expect(state.error).toBe(error.message);
      });
    });
  });
}); 