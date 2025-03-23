import { TokenPackage, SubscriptionTier, FeatureCost, SpecialOffer, EnterprisePlan } from '@/types/monetization';

export const INITIAL_FREE_TOKENS = 60;

export const TOKEN_PACKAGES: TokenPackage[] = [
  {
    id: 'token-100',
    name: 'Starter Pack',
    tokenAmount: 100,
    price: 4.99,
    currency: 'USD',
  },
  {
    id: 'token-500',
    name: 'Popular',
    tokenAmount: 500,
    price: 19.99,
    currency: 'USD',
    isPopular: true,
  },
  {
    id: 'token-1000',
    name: 'Value Pack',
    tokenAmount: 1000,
    price: 34.99,
    currency: 'USD',
  },
  {
    id: 'token-2000',
    name: 'Pro Pack',
    tokenAmount: 2000,
    price: 59.99,
    currency: 'USD',
  },
];

export const SUBSCRIPTION_TIERS: SubscriptionTier[] = [
  {
    id: 'free',
    name: 'Free',
    price: 0,
    currency: 'USD',
    interval: 'monthly',
    tokenAmount: INITIAL_FREE_TOKENS,
    features: [
      'Basic workout tracking',
      'Limited AI form feedback',
      'Basic progress analytics',
      'Standard workout library',
    ],
  },
  {
    id: 'essentials',
    name: 'Fitness Essentials',
    price: 9.99,
    currency: 'USD',
    interval: 'monthly',
    tokenAmount: 500,
    features: [
      'Advanced workout tracking',
      'Unlimited AI form feedback',
      'Detailed progress analytics',
      'Premium workout library',
      'Basic meal planning',
      'Priority support',
    ],
    isPopular: true,
  },
  {
    id: 'pro',
    name: 'Fitness Pro',
    price: 19.99,
    currency: 'USD',
    interval: 'monthly',
    tokenAmount: 1500,
    features: [
      'Everything in Essentials',
      'Custom workout generation',
      'Advanced meal planning',
      'Video analysis',
      'Personalized recommendations',
      'Priority support',
      'Early access to new features',
    ],
  },
  {
    id: 'elite',
    name: 'Fitness Elite',
    price: 29.99,
    currency: 'USD',
    interval: 'monthly',
    tokenAmount: -1, // Unlimited
    features: [
      'Everything in Pro',
      'Unlimited tokens',
      '1-on-1 coaching sessions',
      'Custom program creation',
      'Exclusive content',
      'VIP support',
      'Beta testing access',
    ],
  },
];

export const FEATURE_COSTS: FeatureCost[] = [
  {
    featureId: 'food-analysis',
    name: 'Real-time Food Analysis',
    tokenCost: 2,
    description: 'Get instant nutritional analysis of your meals',
    isPremium: true,
  },
  {
    featureId: 'workout-generation',
    name: 'Custom Workout Generation',
    tokenCost: 10,
    description: 'AI-powered personalized workout plans',
    isPremium: true,
  },
  {
    featureId: 'progress-analytics',
    name: 'Progress Analytics',
    tokenCost: 5,
    description: 'Detailed progress tracking and insights',
    isPremium: true,
  },
  {
    featureId: 'form-feedback',
    name: 'AI Form Feedback',
    tokenCost: 3,
    description: 'Real-time exercise form analysis',
    isPremium: true,
  },
  {
    featureId: 'meal-plan',
    name: 'Meal Plan Generation',
    tokenCost: 15,
    description: 'Personalized meal planning',
    isPremium: true,
  },
];

export const SPECIAL_OFFERS: SpecialOffer[] = [
  {
    id: 'welcome-bonus',
    name: 'Welcome Bonus',
    description: 'Get 100 bonus tokens on your first purchase',
    type: 'bonus_tokens',
    value: 100,
    startDate: new Date('2024-01-01'),
    endDate: new Date('2024-12-31'),
    conditions: ['First-time purchase only'],
    isActive: true,
  },
  {
    id: 'annual-discount',
    name: 'Annual Plan Discount',
    description: 'Save 20% on annual subscriptions',
    type: 'discount',
    value: 20,
    startDate: new Date('2024-01-01'),
    endDate: new Date('2024-12-31'),
    isActive: true,
  },
];

export const ENTERPRISE_PLANS: EnterprisePlan[] = [
  {
    id: 'family',
    name: 'Family Plan',
    price: 49.99,
    currency: 'USD',
    interval: 'monthly',
    maxUsers: 6,
    features: [
      'Everything in Fitness Pro',
      'Family progress tracking',
      'Shared workout library',
      'Family meal planning',
      'Group challenges',
    ],
    apiAccess: false,
    whiteLabel: false,
  },
  {
    id: 'corporate',
    name: 'Corporate Wellness',
    price: 199.99,
    currency: 'USD',
    interval: 'monthly',
    maxUsers: 50,
    features: [
      'Everything in Family Plan',
      'Corporate dashboard',
      'Team challenges',
      'Wellness reporting',
      'HR integration',
      'Custom branding',
    ],
    apiAccess: true,
    whiteLabel: true,
  },
  {
    id: 'gym-partner',
    name: 'Gym Partner Program',
    price: 499.99,
    currency: 'USD',
    interval: 'monthly',
    maxUsers: 200,
    features: [
      'Everything in Corporate Plan',
      'Trainer management',
      'Class scheduling',
      'Member management',
      'Revenue sharing',
      'Custom integrations',
    ],
    apiAccess: true,
    whiteLabel: true,
  },
]; 