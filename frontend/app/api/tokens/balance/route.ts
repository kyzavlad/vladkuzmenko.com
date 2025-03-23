import { NextResponse } from 'next/server';

// In-memory store of user token balances (in production, this would be in a database)
const userTokenBalances = new Map<string, number>();

// In-memory store of token usage history (in production, this would be in a database)
interface TokenUsage {
  timestamp: number;
  amount: number;
  feature: string;
  description: string;
}

const userTokenUsage = new Map<string, TokenUsage[]>();

export async function GET(request: Request) {
  try {
    // In production, get the user ID from the authenticated session
    const userId = 'test-user';
    
    const balance = userTokenBalances.get(userId) || 0;
    const usage = userTokenUsage.get(userId) || [];
    
    // Calculate usage statistics
    const now = Date.now();
    const thirtyDaysAgo = now - (30 * 24 * 60 * 60 * 1000);
    
    const recentUsage = usage.filter(entry => entry.timestamp >= thirtyDaysAgo);
    const totalUsed = recentUsage.reduce((sum, entry) => sum + entry.amount, 0);
    
    // Group usage by feature
    const featureUsage = recentUsage.reduce((acc, entry) => {
      acc[entry.feature] = (acc[entry.feature] || 0) + entry.amount;
      return acc;
    }, {} as Record<string, number>);

    return NextResponse.json({
      balance,
      totalUsed,
      featureUsage,
      recentUsage: recentUsage.slice(0, 10), // Return last 10 entries
      hasLowBalance: balance < 100, // Alert if balance is below 100 tokens
    });
  } catch (error) {
    console.error('Error fetching token balance:', error);
    return NextResponse.json(
      { error: 'Failed to fetch token balance' },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    const { amount, feature, description } = await request.json();
    
    // In production, get the user ID from the authenticated session
    const userId = 'test-user';
    
    const currentBalance = userTokenBalances.get(userId) || 0;
    
    // Check if user has enough tokens
    if (currentBalance < amount) {
      return NextResponse.json(
        { error: 'Insufficient token balance' },
        { status: 400 }
      );
    }
    
    // Update balance
    const newBalance = currentBalance - amount;
    userTokenBalances.set(userId, newBalance);
    
    // Record usage
    const usage: TokenUsage = {
      timestamp: Date.now(),
      amount,
      feature,
      description,
    };
    
    const userUsage = userTokenUsage.get(userId) || [];
    userUsage.unshift(usage);
    userTokenUsage.set(userId, userUsage);
    
    return NextResponse.json({
      success: true,
      balance: newBalance,
      usage,
    });
  } catch (error) {
    console.error('Error updating token balance:', error);
    return NextResponse.json(
      { error: 'Failed to update token balance' },
      { status: 500 }
    );
  }
} 