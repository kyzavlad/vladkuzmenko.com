import { Subscription, SubscriptionTier, PaymentMethod } from '@/types/monetization';
import { SUBSCRIPTION_TIERS } from '@/lib/config/monetization';
import { TokenService } from './tokenService';

export class SubscriptionService {
  private static instance: SubscriptionService;
  private subscriptions: Map<string, Subscription> = new Map();
  private tokenService: TokenService;

  private constructor() {
    this.tokenService = TokenService.getInstance();
  }

  public static getInstance(): SubscriptionService {
    if (!SubscriptionService.instance) {
      SubscriptionService.instance = new SubscriptionService();
    }
    return SubscriptionService.instance;
  }

  public async createSubscription(
    userId: string,
    tierId: string,
    paymentMethodId: string,
    interval: 'monthly' | 'yearly'
  ): Promise<Subscription> {
    const tier = SUBSCRIPTION_TIERS.find((t) => t.id === tierId);
    if (!tier) {
      throw new Error('Invalid subscription tier');
    }

    const price = this.calculatePrice(tier, interval);
    const subscription: Subscription = {
      id: `sub-${Date.now()}`,
      userId,
      tierId,
      status: 'active',
      startDate: new Date(),
      endDate: this.calculateEndDate(new Date(), interval),
      autoRenew: true,
      paymentMethodId,
      interval,
    };

    this.subscriptions.set(userId, subscription);

    // Apply subscription tokens
    if (tier.tokenAmount > 0) {
      await this.tokenService.applySubscriptionTokens(userId, tier.tokenAmount);
    }

    return subscription;
  }

  public async cancelSubscription(userId: string): Promise<void> {
    const subscription = this.subscriptions.get(userId);
    if (subscription) {
      subscription.status = 'cancelled';
      subscription.autoRenew = false;
    }
  }

  public async upgradeSubscription(
    userId: string,
    newTierId: string
  ): Promise<Subscription> {
    const currentSubscription = this.subscriptions.get(userId);
    if (!currentSubscription) {
      throw new Error('No active subscription found');
    }

    const newTier = SUBSCRIPTION_TIERS.find((t) => t.id === newTierId);
    if (!newTier) {
      throw new Error('Invalid subscription tier');
    }

    // Calculate prorated amount if needed
    const proratedAmount = this.calculateProratedAmount(
      currentSubscription,
      newTier
    );

    // Update subscription
    currentSubscription.tierId = newTierId;
    currentSubscription.endDate = this.calculateEndDate(
      currentSubscription.startDate,
      currentSubscription.interval
    );

    // Apply new tier tokens
    if (newTier.tokenAmount > 0) {
      await this.tokenService.applySubscriptionTokens(userId, newTier.tokenAmount);
    }

    return currentSubscription;
  }

  public async getSubscription(userId: string): Promise<Subscription | null> {
    return this.subscriptions.get(userId) || null;
  }

  public async getAllSubscriptions(): Promise<Subscription[]> {
    return Array.from(this.subscriptions.values());
  }

  public async updatePaymentMethod(
    userId: string,
    newPaymentMethodId: string
  ): Promise<void> {
    const subscription = this.subscriptions.get(userId);
    if (subscription) {
      subscription.paymentMethodId = newPaymentMethodId;
    }
  }

  private calculatePrice(tier: SubscriptionTier, interval: 'monthly' | 'yearly'): number {
    if (interval === 'yearly') {
      return tier.price * 12 * 0.8; // 20% discount for yearly
    }
    return tier.price;
  }

  private calculateEndDate(startDate: Date, interval: 'monthly' | 'yearly'): Date {
    const endDate = new Date(startDate);
    if (interval === 'yearly') {
      endDate.setFullYear(endDate.getFullYear() + 1);
    } else {
      endDate.setMonth(endDate.getMonth() + 1);
    }
    return endDate;
  }

  private calculateProratedAmount(
    currentSubscription: Subscription,
    newTier: SubscriptionTier
  ): number {
    const currentTier = SUBSCRIPTION_TIERS.find(
      (t) => t.id === currentSubscription.tierId
    );
    if (!currentTier) return 0;

    const daysRemaining =
      (currentSubscription.endDate.getTime() - new Date().getTime()) /
      (1000 * 60 * 60 * 24);
    const totalDays = this.calculateTotalDays(currentSubscription);

    const currentValue = (currentTier.price * daysRemaining) / totalDays;
    const newValue = (newTier.price * daysRemaining) / totalDays;

    return Math.max(0, newValue - currentValue);
  }

  private calculateTotalDays(subscription: Subscription): number {
    const startDate = subscription.startDate;
    const endDate = subscription.endDate;
    return (endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24);
  }
} 