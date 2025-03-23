import { NextResponse } from 'next/server';
import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2023-10-16',
});

// In-memory store of token packages (in production, this would be in a database)
const tokenPackages = new Map([
  ['basic', { id: 'basic', name: 'Basic', tokens: 1000, price: 10 }],
  ['pro', { id: 'pro', name: 'Pro', tokens: 5000, price: 45, popular: true, savings: 10 }],
  ['enterprise', { id: 'enterprise', name: 'Enterprise', tokens: 10000, price: 80, savings: 20 }],
]);

// In-memory store of user token balances (in production, this would be in a database)
const userTokenBalances = new Map<string, number>();

export async function POST(request: Request) {
  try {
    const { packageId, paymentMethodId } = await request.json();

    // Get the token package
    const tokenPackage = tokenPackages.get(packageId);
    if (!tokenPackage) {
      return NextResponse.json(
        { error: 'Invalid package selected' },
        { status: 400 }
      );
    }

    // Create a payment intent with Stripe
    const paymentIntent = await stripe.paymentIntents.create({
      amount: tokenPackage.price * 100, // Convert to cents
      currency: 'usd',
      payment_method: paymentMethodId,
      confirm: true,
      return_url: `${process.env.NEXT_PUBLIC_BASE_URL}/tokens/confirmation`,
    });

    if (paymentIntent.status === 'succeeded') {
      // In a real application, you would:
      // 1. Store the transaction in your database
      // 2. Update the user's token balance in your database
      // 3. Create an audit log entry
      // 4. Send a confirmation email
      // 5. Trigger any necessary webhooks

      // For this example, we'll just update the in-memory balance
      const userId = 'test-user'; // In production, get this from the authenticated session
      const currentBalance = userTokenBalances.get(userId) || 0;
      userTokenBalances.set(userId, currentBalance + tokenPackage.tokens);

      return NextResponse.json({
        success: true,
        tokens: tokenPackage.tokens,
        balance: currentBalance + tokenPackage.tokens,
        transactionId: paymentIntent.id,
      });
    } else {
      return NextResponse.json(
        { error: 'Payment failed' },
        { status: 400 }
      );
    }
  } catch (error) {
    console.error('Token purchase error:', error);
    return NextResponse.json(
      { error: 'An error occurred while processing your payment' },
      { status: 500 }
    );
  }
}

export async function GET() {
  return NextResponse.json({
    packages: Array.from(tokenPackages.values()),
  });
} 