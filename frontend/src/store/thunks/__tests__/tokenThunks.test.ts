import { configureStore } from '@reduxjs/toolkit';
import tokenReducer from '../../slices/tokenSlice';
import {
  fetchTokenBalanceThunk,
  purchaseTokensThunk,
  redeemPromoCodeThunk,
  fetchUsageHistoryThunk,
} from '../tokenThunks';
import { tokenApi } from '../../services/tokenApi';

// Mock API responses
const mockApiResponses = {
  balance: { balance: 100 },
  purchase: { balance: 150, transactionId: 'tx123' },
  promoCode: { balance: 200, bonusAmount: 50 },
  usageHistory: {
    history: [
      {
        id: '1',
        amount: -10,
        description: 'Video processing',
        timestamp: '2024-01-20T12:00:00Z',
      },
    ],
  },
};

// Mock API service
jest.mock('../../services/tokenApi', () => ({
  tokenApi: {
    getBalance: jest.fn(),
    purchaseTokens: jest.fn(),
    redeemPromoCode: jest.fn(),
    getUsageHistory: jest.fn(),
  },
}));

describe('Token Management Thunks', () => {
  let store: ReturnType<typeof configureStore>;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        token: tokenReducer,
      },
    });
    jest.clearAllMocks();
  });

  describe('fetchTokenBalanceThunk', () => {
    it('successfully fetches token balance', async () => {
      (tokenApi.getBalance as jest.Mock).mockResolvedValueOnce(mockApiResponses.balance);

      const result = await store.dispatch(fetchTokenBalanceThunk());
      const state = store.getState().token;

      expect(result.payload).toEqual(mockApiResponses.balance);
      expect(state.balance).toBe(mockApiResponses.balance.balance);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('handles fetch balance error', async () => {
      const error = new Error('Failed to fetch balance');
      (tokenApi.getBalance as jest.Mock).mockRejectedValueOnce(error);

      const result = await store.dispatch(fetchTokenBalanceThunk());
      const state = store.getState().token;

      expect(result.error?.message).toBe(error.message);
      expect(state.loading).toBe(false);
      expect(state.error).toBe(error.message);
    });
  });

  describe('purchaseTokensThunk', () => {
    const purchaseAmount = 50;

    it('successfully purchases tokens', async () => {
      (tokenApi.purchaseTokens as jest.Mock).mockResolvedValueOnce(mockApiResponses.purchase);

      const result = await store.dispatch(purchaseTokensThunk({ amount: purchaseAmount }));
      const state = store.getState().token;

      expect(result.payload).toEqual(mockApiResponses.purchase);
      expect(state.balance).toBe(mockApiResponses.purchase.balance);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('handles purchase error', async () => {
      const error = new Error('Purchase failed');
      (tokenApi.purchaseTokens as jest.Mock).mockRejectedValueOnce(error);

      const result = await store.dispatch(purchaseTokensThunk({ amount: purchaseAmount }));
      const state = store.getState().token;

      expect(result.error?.message).toBe(error.message);
      expect(state.loading).toBe(false);
      expect(state.error).toBe(error.message);
    });
  });

  describe('redeemPromoCodeThunk', () => {
    const promoCode = 'BONUS50';

    it('successfully redeems promo code', async () => {
      (tokenApi.redeemPromoCode as jest.Mock).mockResolvedValueOnce(mockApiResponses.promoCode);

      const result = await store.dispatch(redeemPromoCodeThunk({ code: promoCode }));
      const state = store.getState().token;

      expect(result.payload).toEqual(mockApiResponses.promoCode);
      expect(state.balance).toBe(mockApiResponses.promoCode.balance);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('handles promo code redemption error', async () => {
      const error = new Error('Invalid promo code');
      (tokenApi.redeemPromoCode as jest.Mock).mockRejectedValueOnce(error);

      const result = await store.dispatch(redeemPromoCodeThunk({ code: promoCode }));
      const state = store.getState().token;

      expect(result.error?.message).toBe(error.message);
      expect(state.loading).toBe(false);
      expect(state.error).toBe(error.message);
    });
  });

  describe('fetchUsageHistoryThunk', () => {
    it('successfully fetches usage history', async () => {
      (tokenApi.getUsageHistory as jest.Mock).mockResolvedValueOnce(mockApiResponses.usageHistory);

      const result = await store.dispatch(fetchUsageHistoryThunk());
      const state = store.getState().token;

      expect(result.payload).toEqual(mockApiResponses.usageHistory);
      expect(state.usageHistory).toEqual(mockApiResponses.usageHistory.history);
      expect(state.loading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('handles fetch history error', async () => {
      const error = new Error('Failed to fetch history');
      (tokenApi.getUsageHistory as jest.Mock).mockRejectedValueOnce(error);

      const result = await store.dispatch(fetchUsageHistoryThunk());
      const state = store.getState().token;

      expect(result.error?.message).toBe(error.message);
      expect(state.loading).toBe(false);
      expect(state.error).toBe(error.message);
    });
  });
}); 