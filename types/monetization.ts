export interface TokenPackage {
  id: string;
  name: string;
  tokenAmount: number;
  price: number;
  currency: string;
  isPopular?: boolean;
}

export interface SubscriptionTier {
  id: string;
  name: string;
  price: number;
  currency: string;
  interval: 'monthly' | 'yearly';
  tokenAmount: number;
  features: string[];
  isPopular?: boolean;
}

export interface FeatureCost {
  featureId: string;
  name: string;
  tokenCost: number;
  description: string;
  isPremium: boolean;
}

export interface UserTokens {
  userId: string;
  balance: number;
  history: TokenTransaction[];
}

export interface TokenTransaction {
  id: string;
  userId: string;
  amount: number;
  type: 'earn' | 'spend' | 'purchase' | 'subscription';
  featureId?: string;
  timestamp: Date;
  description: string;
}

export interface PaymentMethod {
  id: string;
  userId: string;
  type: 'credit_card' | 'paypal' | 'apple_pay' | 'google_pay';
  lastFour?: string;
  expiryDate?: string;
  isDefault: boolean;
}

export interface Subscription {
  id: string;
  userId: string;
  tierId: string;
  status: 'active' | 'cancelled' | 'expired' | 'pending';
  startDate: Date;
  endDate: Date;
  autoRenew: boolean;
  paymentMethodId: string;
  interval: 'monthly' | 'yearly';
}

export interface SpecialOffer {
  id: string;
  name: string;
  description: string;
  type: 'discount' | 'bonus_tokens' | 'bundle';
  value: number;
  startDate: Date;
  endDate: Date;
  conditions?: string[];
  isActive: boolean;
}

export interface EnterprisePlan {
  id: string;
  name: string;
  price: number;
  currency: string;
  interval: 'monthly' | 'yearly';
  maxUsers: number;
  features: string[];
  customFeatures?: string[];
  apiAccess: boolean;
  whiteLabel: boolean;
}

export interface EnterpriseSubscription {
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