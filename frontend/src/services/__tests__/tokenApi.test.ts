import { tokenApi } from '../tokenApi';
import { TokenBalance, TokenPurchase, TokenUsage } from '../../types/token';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('Token API Service', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  describe('getBalance', () => {
    const mockBalance: TokenBalance = { balance: 100 };

    it('successfully fetches token balance', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockBalance),
      });

      const result = await tokenApi.getBalance();
      
      expect(result).toEqual(mockBalance);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/tokens/balance'),
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
    });

    it('handles fetch error', async () => {
      const errorMessage = 'Failed to fetch balance';
      mockFetch.mockRejectedValueOnce(new Error(errorMessage));

      await expect(tokenApi.getBalance()).rejects.toThrow(errorMessage);
    });

    it('handles API error response', async () => {
      const errorResponse = { message: 'Invalid request' };
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve(errorResponse),
      });

      await expect(tokenApi.getBalance()).rejects.toThrow(errorResponse.message);
    });
  });

  describe('purchaseTokens', () => {
    const mockPurchase: TokenPurchase = {
      amount: 50,
      paymentMethod: 'card',
      transactionId: 'tx123',
    };

    it('successfully purchases tokens', async () => {
      const mockResponse = { balance: 150, transactionId: mockPurchase.transactionId };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await tokenApi.purchaseTokens(mockPurchase);
      
      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/tokens/purchase'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(mockPurchase),
        })
      );
    });

    it('handles purchase error', async () => {
      const errorMessage = 'Payment failed';
      mockFetch.mockRejectedValueOnce(new Error(errorMessage));

      await expect(tokenApi.purchaseTokens(mockPurchase)).rejects.toThrow(errorMessage);
    });
  });

  describe('getUsageHistory', () => {
    const mockUsageHistory: TokenUsage[] = [
      {
        id: '1',
        amount: -10,
        description: 'Video processing',
        timestamp: '2024-01-20T12:00:00Z',
      },
    ];

    it('successfully fetches usage history', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ history: mockUsageHistory }),
      });

      const result = await tokenApi.getUsageHistory();
      
      expect(result).toEqual({ history: mockUsageHistory });
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/tokens/usage'),
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
    });

    it('handles pagination parameters', async () => {
      const page = 1;
      const limit = 10;
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ history: mockUsageHistory }),
      });

      await tokenApi.getUsageHistory({ page, limit });
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining(`/api/tokens/usage?page=${page}&limit=${limit}`),
        expect.any(Object)
      );
    });
  });

  describe('redeemPromoCode', () => {
    const promoCode = 'BONUS50';
    const mockResponse = { balance: 150, bonusAmount: 50 };

    it('successfully redeems promo code', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await tokenApi.redeemPromoCode(promoCode);
      
      expect(result).toEqual(mockResponse);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/tokens/redeem'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify({ code: promoCode }),
        })
      );
    });

    it('handles invalid promo code', async () => {
      const errorResponse = { message: 'Invalid promo code' };
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve(errorResponse),
      });

      await expect(tokenApi.redeemPromoCode(promoCode)).rejects.toThrow(errorResponse.message);
    });
  });
}); 