import { UserTokens, TokenTransaction, FeatureCost } from '@/types/monetization';
import { FEATURE_COSTS, INITIAL_FREE_TOKENS } from '@/lib/config/monetization';

export class TokenService {
  private static instance: TokenService;
  private userTokens: Map<string, UserTokens> = new Map();

  private constructor() {}

  public static getInstance(): TokenService {
    if (!TokenService.instance) {
      TokenService.instance = new TokenService();
    }
    return TokenService.instance;
  }

  public async initializeUserTokens(userId: string): Promise<UserTokens> {
    const userTokens: UserTokens = {
      userId,
      balance: INITIAL_FREE_TOKENS,
      history: [
        {
          id: `initial-${Date.now()}`,
          userId,
          amount: INITIAL_FREE_TOKENS,
          type: 'earn',
          timestamp: new Date(),
          description: 'Initial free tokens',
        },
      ],
    };

    this.userTokens.set(userId, userTokens);
    return userTokens;
  }

  public async getTokenBalance(userId: string): Promise<number> {
    const userTokens = this.userTokens.get(userId);
    return userTokens?.balance || 0;
  }

  public async spendTokens(
    userId: string,
    featureId: string,
    amount: number
  ): Promise<boolean> {
    const userTokens = this.userTokens.get(userId);
    if (!userTokens || userTokens.balance < amount) {
      return false;
    }

    userTokens.balance -= amount;
    userTokens.history.push({
      id: `spend-${Date.now()}`,
      userId,
      amount: -amount,
      type: 'spend',
      featureId,
      timestamp: new Date(),
      description: `Spent ${amount} tokens on ${featureId}`,
    });

    return true;
  }

  public async earnTokens(
    userId: string,
    amount: number,
    description: string
  ): Promise<void> {
    const userTokens = this.userTokens.get(userId);
    if (!userTokens) {
      await this.initializeUserTokens(userId);
      return this.earnTokens(userId, amount, description);
    }

    userTokens.balance += amount;
    userTokens.history.push({
      id: `earn-${Date.now()}`,
      userId,
      amount,
      type: 'earn',
      timestamp: new Date(),
      description,
    });
  }

  public async getFeatureCost(featureId: string): Promise<number> {
    const feature = FEATURE_COSTS.find((f) => f.featureId === featureId);
    return feature?.tokenCost || 0;
  }

  public async canUseFeature(userId: string, featureId: string): Promise<boolean> {
    const cost = await this.getFeatureCost(featureId);
    const balance = await this.getTokenBalance(userId);
    return balance >= cost;
  }

  public async getTransactionHistory(userId: string): Promise<TokenTransaction[]> {
    const userTokens = this.userTokens.get(userId);
    return userTokens?.history || [];
  }

  public async getAllTransactions(): Promise<TokenTransaction[]> {
    const allTransactions: TokenTransaction[] = [];
    const userTokensArray = Array.from(this.userTokens.values());
    for (const userTokens of userTokensArray) {
      allTransactions.push(...userTokens.history);
    }
    return allTransactions;
  }

  public async purchaseTokens(
    userId: string,
    amount: number,
    price: number
  ): Promise<void> {
    const userTokens = this.userTokens.get(userId);
    if (!userTokens) {
      await this.initializeUserTokens(userId);
      return this.purchaseTokens(userId, amount, price);
    }

    userTokens.balance += amount;
    userTokens.history.push({
      id: `purchase-${Date.now()}`,
      userId,
      amount,
      type: 'purchase',
      timestamp: new Date(),
      description: `Purchased ${amount} tokens for $${price}`,
    });
  }

  public async applySubscriptionTokens(
    userId: string,
    amount: number
  ): Promise<void> {
    const userTokens = this.userTokens.get(userId);
    if (!userTokens) {
      await this.initializeUserTokens(userId);
      return this.applySubscriptionTokens(userId, amount);
    }

    userTokens.balance += amount;
    userTokens.history.push({
      id: `subscription-${Date.now()}`,
      userId,
      amount,
      type: 'subscription',
      timestamp: new Date(),
      description: `Received ${amount} tokens from subscription`,
    });
  }
} 