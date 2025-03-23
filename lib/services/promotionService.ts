import { SpecialOffer } from '@/types/monetization';
import { SPECIAL_OFFERS } from '@/lib/config/monetization';
import { TokenService } from './tokenService';

export class PromotionService {
  private static instance: PromotionService;
  private tokenService: TokenService;
  private usedOffers: Map<string, Set<string>> = new Map();

  private constructor() {
    this.tokenService = TokenService.getInstance();
  }

  public static getInstance(): PromotionService {
    if (!PromotionService.instance) {
      PromotionService.instance = new PromotionService();
    }
    return PromotionService.instance;
  }

  public async getActiveOffers(): Promise<SpecialOffer[]> {
    const now = new Date();
    return SPECIAL_OFFERS.filter(
      (offer) =>
        offer.isActive &&
        offer.startDate <= now &&
        offer.endDate >= now
    );
  }

  public async applyOffer(
    userId: string,
    offerId: string
  ): Promise<boolean> {
    const offer = SPECIAL_OFFERS.find((o) => o.id === offerId);
    if (!offer) {
      throw new Error('Invalid offer');
    }

    // Check if offer is active
    if (!this.isOfferActive(offer)) {
      return false;
    }

    // Check if user has already used this offer
    if (this.hasUserUsedOffer(userId, offerId)) {
      return false;
    }

    // Apply the offer based on its type
    switch (offer.type) {
      case 'bonus_tokens':
        await this.tokenService.earnTokens(
          userId,
          offer.value,
          `Bonus tokens from promotion: ${offer.name}`
        );
        break;
      case 'discount':
        // Handle discount in the payment service
        break;
      case 'bundle':
        // Handle bundle offers
        break;
    }

    // Mark offer as used
    this.markOfferAsUsed(userId, offerId);
    return true;
  }

  public async checkOfferEligibility(
    userId: string,
    offerId: string
  ): Promise<boolean> {
    const offer = SPECIAL_OFFERS.find((o) => o.id === offerId);
    if (!offer) {
      return false;
    }

    // Check if offer is active
    if (!this.isOfferActive(offer)) {
      return false;
    }

    // Check if user has already used this offer
    if (this.hasUserUsedOffer(userId, offerId)) {
      return false;
    }

    // Check if user meets any additional conditions
    if (offer.conditions) {
      return this.checkConditions(userId, offer.conditions);
    }

    return true;
  }

  private isOfferActive(offer: SpecialOffer): boolean {
    const now = new Date();
    return (
      offer.isActive &&
      offer.startDate <= now &&
      offer.endDate >= now
    );
  }

  private hasUserUsedOffer(userId: string, offerId: string): boolean {
    const userOffers = this.usedOffers.get(userId);
    return userOffers?.has(offerId) || false;
  }

  private markOfferAsUsed(userId: string, offerId: string): void {
    const userOffers = this.usedOffers.get(userId) || new Set();
    userOffers.add(offerId);
    this.usedOffers.set(userId, userOffers);
  }

  private async checkConditions(
    userId: string,
    conditions: string[]
  ): Promise<boolean> {
    // Implement condition checking logic here
    // For example, checking user's subscription status, purchase history, etc.
    return true;
  }

  public async getReferralBonus(userId: string): Promise<number> {
    // Implement referral bonus logic here
    // For example, checking the number of successful referrals
    return 50; // Example: 50 tokens per successful referral
  }

  public async applyReferralBonus(
    referrerId: string,
    referredId: string
  ): Promise<void> {
    const bonus = await this.getReferralBonus(referrerId);
    await this.tokenService.earnTokens(
      referrerId,
      bonus,
      `Referral bonus for user ${referredId}`
    );
  }

  public async getLoyaltyRewards(userId: string): Promise<number> {
    // Implement loyalty rewards logic here
    // For example, based on subscription duration, engagement, etc.
    return 100; // Example: 100 tokens for long-term loyalty
  }

  public async applyLoyaltyRewards(userId: string): Promise<void> {
    const rewards = await this.getLoyaltyRewards(userId);
    await this.tokenService.earnTokens(
      userId,
      rewards,
      'Loyalty rewards'
    );
  }
} 