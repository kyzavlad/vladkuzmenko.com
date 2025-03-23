import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import axios from 'axios';

interface TokenUsage {
  featureId: string;
  featureName: string;
  tokensUsed: number;
  timestamp: string;
}

interface TokenPackage {
  id: string;
  name: string;
  tokenAmount: number;
  price: number;
  isSubscription: boolean;
  validityDays: number;
}

interface TokenState {
  balance: number;
  freeTokens: number;
  paidTokens: number;
  usageHistory: TokenUsage[];
  availablePackages: TokenPackage[];
  isLoading: boolean;
  error: string | null;
  lastUpdated: string | null;
}

const initialState: TokenState = {
  balance: 0,
  freeTokens: 0,
  paidTokens: 0,
  usageHistory: [],
  availablePackages: [],
  isLoading: false,
  error: null,
  lastUpdated: null,
};

export const fetchTokenBalance = createAsyncThunk(
  'tokens/fetchBalance',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: { token: string } };
      const response = await axios.get('/api/tokens/balance', {
        headers: { Authorization: `Bearer ${state.auth.token}` },
      });
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch token balance');
    }
  }
);

export const fetchUsageHistory = createAsyncThunk(
  'tokens/fetchUsageHistory',
  async (_, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: { token: string } };
      const response = await axios.get('/api/tokens/usage-history', {
        headers: { Authorization: `Bearer ${state.auth.token}` },
      });
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch usage history');
    }
  }
);

export const purchaseTokens = createAsyncThunk(
  'tokens/purchase',
  async (
    { packageId, paymentMethodId }: { packageId: string; paymentMethodId: string },
    { getState, rejectWithValue }
  ) => {
    try {
      const state = getState() as { auth: { token: string } };
      const response = await axios.post(
        '/api/tokens/purchase',
        { packageId, paymentMethodId },
        { headers: { Authorization: `Bearer ${state.auth.token}` } }
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Token purchase failed');
    }
  }
);

export const redeemPromoCode = createAsyncThunk(
  'tokens/redeemPromo',
  async (promoCode: string, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { auth: { token: string } };
      const response = await axios.post(
        '/api/tokens/redeem-promo',
        { promoCode },
        { headers: { Authorization: `Bearer ${state.auth.token}` } }
      );
      return response.data;
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to redeem promo code');
    }
  }
);

const tokenSlice = createSlice({
  name: 'tokens',
  initialState,
  reducers: {
    updateBalance: (state, action: PayloadAction<number>) => {
      state.balance = action.payload;
      state.lastUpdated = new Date().toISOString();
    },
    addUsage: (state, action: PayloadAction<TokenUsage>) => {
      state.usageHistory.unshift(action.payload);
      state.balance -= action.payload.tokensUsed;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch Balance
      .addCase(fetchTokenBalance.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(fetchTokenBalance.fulfilled, (state, action) => {
        state.isLoading = false;
        state.balance = action.payload.balance;
        state.freeTokens = action.payload.freeTokens;
        state.paidTokens = action.payload.paidTokens;
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(fetchTokenBalance.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      // Fetch Usage History
      .addCase(fetchUsageHistory.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(fetchUsageHistory.fulfilled, (state, action) => {
        state.isLoading = false;
        state.usageHistory = action.payload;
      })
      .addCase(fetchUsageHistory.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      // Purchase Tokens
      .addCase(purchaseTokens.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(purchaseTokens.fulfilled, (state, action) => {
        state.isLoading = false;
        state.balance = action.payload.newBalance;
        state.paidTokens = action.payload.paidTokens;
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(purchaseTokens.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      })
      // Redeem Promo Code
      .addCase(redeemPromoCode.pending, (state) => {
        state.isLoading = true;
      })
      .addCase(redeemPromoCode.fulfilled, (state, action) => {
        state.isLoading = false;
        state.balance = action.payload.newBalance;
        state.freeTokens = action.payload.freeTokens;
        state.lastUpdated = new Date().toISOString();
      })
      .addCase(redeemPromoCode.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.payload as string;
      });
  },
});

export const { updateBalance, addUsage, clearError } = tokenSlice.actions;
export default tokenSlice.reducer; 