import { TokenTransaction, Subscription, EnterpriseSubscription } from '@/types/monetization';
import { SUBSCRIPTION_TIERS, TOKEN_PACKAGES, ENTERPRISE_PLANS } from '@/lib/config/monetization';
import { TokenService } from './tokenService';
import { SubscriptionService } from './subscriptionService';
import { EnterpriseService } from './enterpriseService';

export interface MonetizationMetrics {
  totalRevenue: number;
  activeSubscriptions: number;
  totalTokensIssued: number;
  totalTokensSpent: number;
  averageRevenuePerUser: number;
  conversionRate: number;
  churnRate: number;
  enterpriseRevenue: number;
  subscriptionRevenue: number;
  tokenPackageRevenue: number;
}

export interface UserMetrics {
  userId: string;
  totalSpent: number;
  subscriptionStatus: string;
  tokenBalance: number;
  tokenUsage: TokenUsageMetrics;
  featureUsage: FeatureUsageMetrics;
}

interface TokenUsageMetrics {
  totalEarned: number;
  totalSpent: number;
  currentBalance: number;
  usageByFeature: Map<string, number>;
}

interface FeatureUsageMetrics {
  totalFeaturesUsed: number;
  usageByFeature: Map<string, number>;
  lastUsedFeatures: string[];
}

export class MonetizationAnalyticsService {
  private static instance: MonetizationAnalyticsService;
  private tokenService: TokenService;
  private subscriptionService: SubscriptionService;
  private enterpriseService: EnterpriseService;
  private featureUsage: Map<string, Map<string, number>> = new Map();

  private constructor() {
    this.tokenService = TokenService.getInstance();
    this.subscriptionService = SubscriptionService.getInstance();
    this.enterpriseService = EnterpriseService.getInstance();
  }

  public static getInstance(): MonetizationAnalyticsService {
    if (!MonetizationAnalyticsService.instance) {
      MonetizationAnalyticsService.instance = new MonetizationAnalyticsService();
    }
    return MonetizationAnalyticsService.instance;
  }

  public async getOverallMetrics(): Promise<MonetizationMetrics> {
    const subscriptions = await this.subscriptionService.getAllSubscriptions();
    const enterpriseSubscriptions = await this.enterpriseService.getAllSubscriptions();
    const tokenTransactions = await this.tokenService.getAllTransactions();

    const metrics: MonetizationMetrics = {
      totalRevenue: 0,
      activeSubscriptions: 0,
      totalTokensIssued: 0,
      totalTokensSpent: 0,
      averageRevenuePerUser: 0,
      conversionRate: 0,
      churnRate: 0,
      enterpriseRevenue: 0,
      subscriptionRevenue: 0,
      tokenPackageRevenue: 0,
    };

    // Calculate subscription revenue
    for (const subscription of subscriptions) {
      const tier = SUBSCRIPTION_TIERS.find((t) => t.id === subscription.tierId);
      if (tier) {
        metrics.subscriptionRevenue += tier.price;
        if (subscription.status === 'active') {
          metrics.activeSubscriptions++;
        }
      }
    }

    // Calculate enterprise revenue
    for (const subscription of enterpriseSubscriptions) {
      const plan = ENTERPRISE_PLANS.find((p) => p.id === subscription.planId);
      if (plan) {
        metrics.enterpriseRevenue += plan.price;
      }
    }

    // Calculate token package revenue
    for (const transaction of tokenTransactions) {
      if (transaction.type === 'purchase') {
        const tokenPackage = TOKEN_PACKAGES.find(
          (p: { tokenAmount: number }) => p.tokenAmount === transaction.amount
        );
        if (tokenPackage) {
          metrics.tokenPackageRevenue += tokenPackage.price;
        }
      }
    }

    // Calculate total revenue
    metrics.totalRevenue =
      metrics.subscriptionRevenue +
      metrics.enterpriseRevenue +
      metrics.tokenPackageRevenue;

    // Calculate token metrics
    for (const transaction of tokenTransactions) {
      if (transaction.type === 'earn' || transaction.type === 'purchase') {
        metrics.totalTokensIssued += transaction.amount;
      } else if (transaction.type === 'spend') {
        metrics.totalTokensSpent += Math.abs(transaction.amount);
      }
    }

    // Calculate other metrics
    const totalUsers = subscriptions.length + enterpriseSubscriptions.length;
    metrics.averageRevenuePerUser =
      totalUsers > 0 ? metrics.totalRevenue / totalUsers : 0;

    // Calculate conversion rate (users who made a purchase / total users)
    const usersWithPurchases = new Set(
      tokenTransactions
        .filter((t: TokenTransaction) => t.type === 'purchase')
        .map((t: TokenTransaction) => t.userId)
    ).size;
    metrics.conversionRate =
      totalUsers > 0 ? (usersWithPurchases / totalUsers) * 100 : 0;

    // Calculate churn rate (cancelled subscriptions / total subscriptions)
    const cancelledSubscriptions = subscriptions.filter(
      (s) => s.status === 'cancelled'
    ).length;
    metrics.churnRate =
      subscriptions.length > 0
        ? (cancelledSubscriptions / subscriptions.length) * 100
        : 0;

    return metrics;
  }

  public async getUserMetrics(userId: string): Promise<UserMetrics> {
    const tokenTransactions = await this.tokenService.getTransactionHistory(userId);
    const subscription = await this.subscriptionService.getSubscription(userId);
    const tokenBalance = await this.tokenService.getTokenBalance(userId);

    const metrics: UserMetrics = {
      userId,
      totalSpent: 0,
      subscriptionStatus: subscription?.status || 'none',
      tokenBalance,
      tokenUsage: {
        totalEarned: 0,
        totalSpent: 0,
        currentBalance: tokenBalance,
        usageByFeature: new Map(),
      },
      featureUsage: {
        totalFeaturesUsed: 0,
        usageByFeature: new Map(),
        lastUsedFeatures: [],
      },
    };

    // Calculate token usage metrics
    for (const transaction of tokenTransactions) {
      if (transaction.type === 'earn' || transaction.type === 'purchase') {
        metrics.tokenUsage.totalEarned += transaction.amount;
      } else if (transaction.type === 'spend') {
        metrics.tokenUsage.totalSpent += Math.abs(transaction.amount);
        if (transaction.featureId) {
          const currentUsage =
            metrics.tokenUsage.usageByFeature.get(transaction.featureId) || 0;
          metrics.tokenUsage.usageByFeature.set(
            transaction.featureId,
            currentUsage + Math.abs(transaction.amount)
          );
        }
      }
    }

    // Calculate feature usage metrics
    const featureUsage = this.featureUsage.get(userId) || new Map();
    metrics.featureUsage.totalFeaturesUsed = featureUsage.size;
    metrics.featureUsage.usageByFeature = featureUsage;

    // Get last used features
    metrics.featureUsage.lastUsedFeatures = Array.from(featureUsage.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([feature]) => feature);

    // Calculate total spent
    if (subscription) {
      const tier = SUBSCRIPTION_TIERS.find((t) => t.id === subscription.tierId);
      if (tier) {
        metrics.totalSpent += tier.price;
      }
    }

    for (const transaction of tokenTransactions) {
      if (transaction.type === 'purchase') {
        const tokenPackage = TOKEN_PACKAGES.find(
          (p) => p.tokenAmount === transaction.amount
        );
        if (tokenPackage) {
          metrics.totalSpent += tokenPackage.price;
        }
      }
    }

    return metrics;
  }

  public async trackFeatureUsage(
    userId: string,
    featureId: string
  ): Promise<void> {
    const userFeatures = this.featureUsage.get(userId) || new Map();
    const currentUsage = userFeatures.get(featureId) || 0;
    userFeatures.set(featureId, currentUsage + 1);
    this.featureUsage.set(userId, userFeatures);
  }

  public async getFeatureUsageMetrics(
    featureId: string
  ): Promise<{
    totalUsage: number;
    uniqueUsers: number;
    averageUsagePerUser: number;
  }> {
    let totalUsage = 0;
    let uniqueUsers = 0;

    const featureUsageArray = Array.from(this.featureUsage.entries());
    for (const [userId, features] of featureUsageArray) {
      const usage = features.get(featureId) || 0;
      if (usage > 0) {
        totalUsage += usage;
        uniqueUsers++;
      }
    }

    return {
      totalUsage,
      uniqueUsers,
      averageUsagePerUser: uniqueUsers > 0 ? totalUsage / uniqueUsers : 0,
    };
  }
} 