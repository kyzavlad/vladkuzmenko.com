import { EnterprisePlan } from '@/types/monetization';
import { ENTERPRISE_PLANS } from '@/lib/config/monetization';
import { SubscriptionService } from './subscriptionService';
import { TokenService } from './tokenService';

export class EnterpriseService {
  private static instance: EnterpriseService;
  private subscriptionService: SubscriptionService;
  private tokenService: TokenService;
  private enterpriseSubscriptions: Map<string, EnterpriseSubscription> = new Map();

  private constructor() {
    this.subscriptionService = SubscriptionService.getInstance();
    this.tokenService = TokenService.getInstance();
  }

  public static getInstance(): EnterpriseService {
    if (!EnterpriseService.instance) {
      EnterpriseService.instance = new EnterpriseService();
    }
    return EnterpriseService.instance;
  }

  public async createEnterpriseSubscription(
    organizationId: string,
    planId: string,
    adminUserId: string,
    users: string[]
  ): Promise<EnterpriseSubscription> {
    const plan = ENTERPRISE_PLANS.find((p) => p.id === planId);
    if (!plan) {
      throw new Error('Invalid enterprise plan');
    }

    if (users.length > plan.maxUsers) {
      throw new Error('Number of users exceeds plan limit');
    }

    const subscription: EnterpriseSubscription = {
      id: `ent-${Date.now()}`,
      organizationId,
      planId,
      adminUserId,
      users,
      status: 'active',
      startDate: new Date(),
      endDate: this.calculateEndDate(new Date(), plan.interval),
      autoRenew: true,
    };

    this.enterpriseSubscriptions.set(organizationId, subscription);

    // Initialize tokens for all users
    for (const userId of users) {
      await this.tokenService.initializeUserTokens(userId);
    }

    return subscription;
  }

  public async addUserToEnterprise(
    organizationId: string,
    userId: string
  ): Promise<void> {
    const subscription = this.enterpriseSubscriptions.get(organizationId);
    if (!subscription) {
      throw new Error('Enterprise subscription not found');
    }

    const plan = ENTERPRISE_PLANS.find((p) => p.id === subscription.planId);
    if (!plan) {
      throw new Error('Invalid enterprise plan');
    }

    if (subscription.users.length >= plan.maxUsers) {
      throw new Error('Maximum number of users reached for this plan');
    }

    subscription.users.push(userId);
    await this.tokenService.initializeUserTokens(userId);
  }

  public async removeUserFromEnterprise(
    organizationId: string,
    userId: string
  ): Promise<void> {
    const subscription = this.enterpriseSubscriptions.get(organizationId);
    if (!subscription) {
      throw new Error('Enterprise subscription not found');
    }

    subscription.users = subscription.users.filter((id) => id !== userId);
  }

  public async upgradeEnterprisePlan(
    organizationId: string,
    newPlanId: string
  ): Promise<EnterpriseSubscription> {
    const subscription = this.enterpriseSubscriptions.get(organizationId);
    if (!subscription) {
      throw new Error('Enterprise subscription not found');
    }

    const newPlan = ENTERPRISE_PLANS.find((p) => p.id === newPlanId);
    if (!newPlan) {
      throw new Error('Invalid enterprise plan');
    }

    if (subscription.users.length > newPlan.maxUsers) {
      throw new Error('Current number of users exceeds new plan limit');
    }

    subscription.planId = newPlanId;
    subscription.endDate = this.calculateEndDate(
      subscription.startDate,
      newPlan.interval
    );

    return subscription;
  }

  public async getEnterpriseSubscription(
    organizationId: string
  ): Promise<EnterpriseSubscription | null> {
    return this.enterpriseSubscriptions.get(organizationId) || null;
  }

  public async getAllSubscriptions(): Promise<EnterpriseSubscription[]> {
    return Array.from(this.enterpriseSubscriptions.values());
  }

  public async getEnterpriseUsers(
    organizationId: string
  ): Promise<string[]> {
    const subscription = this.enterpriseSubscriptions.get(organizationId);
    return subscription?.users || [];
  }

  public async cancelEnterpriseSubscription(
    organizationId: string
  ): Promise<void> {
    const subscription = this.enterpriseSubscriptions.get(organizationId);
    if (subscription) {
      subscription.status = 'cancelled';
      subscription.autoRenew = false;
    }
  }

  private calculateEndDate(
    startDate: Date,
    interval: 'monthly' | 'yearly'
  ): Date {
    const endDate = new Date(startDate);
    if (interval === 'yearly') {
      endDate.setFullYear(endDate.getFullYear() + 1);
    } else {
      endDate.setMonth(endDate.getMonth() + 1);
    }
    return endDate;
  }

  public async getEnterpriseFeatures(
    organizationId: string
  ): Promise<string[]> {
    const subscription = this.enterpriseSubscriptions.get(organizationId);
    if (!subscription) {
      return [];
    }

    const plan = ENTERPRISE_PLANS.find((p) => p.id === subscription.planId);
    return plan?.features || [];
  }

  public async hasFeatureAccess(
    organizationId: string,
    feature: string
  ): Promise<boolean> {
    const features = await this.getEnterpriseFeatures(organizationId);
    return features.includes(feature);
  }
}

interface EnterpriseSubscription {
  id: string;
  organizationId: string;
  planId: string;
  adminUserId: string;
  users: string[];
  status: 'active' | 'cancelled' | 'expired';
  startDate: Date;
  endDate: Date;
  autoRenew: boolean;
} 