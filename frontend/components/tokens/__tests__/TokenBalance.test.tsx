import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { TokenBalance } from '../TokenBalance';
import { ChakraProvider } from '@chakra-ui/react';

describe('TokenBalance', () => {
  const defaultProps = {
    balance: 5000,
    totalAllocation: 10000,
    lowBalanceThreshold: 1000,
    onPurchase: jest.fn(),
  };

  const renderComponent = (props = {}) => {
    return render(
      <ChakraProvider>
        <TokenBalance {...defaultProps} {...props} />
      </ChakraProvider>
    );
  };

  it('renders the component with correct balance', () => {
    renderComponent();
    expect(screen.getByText('5,000')).toBeInTheDocument();
    expect(screen.getByText('10,000 tokens')).toBeInTheDocument();
  });

  it('shows low balance warning when balance is below threshold', () => {
    renderComponent({ balance: 500 });
    expect(screen.getByText(/your token balance is running low/i)).toBeInTheDocument();
  });

  it('does not show low balance warning when balance is above threshold', () => {
    renderComponent();
    expect(screen.queryByText(/your token balance is running low/i)).not.toBeInTheDocument();
  });

  it('calls onPurchase when purchase button is clicked', () => {
    renderComponent();
    const purchaseButton = screen.getByText('Purchase Tokens');
    fireEvent.click(purchaseButton);
    expect(defaultProps.onPurchase).toHaveBeenCalled();
  });

  it('displays correct used tokens amount', () => {
    renderComponent();
    expect(screen.getByText('5,000 tokens')).toBeInTheDocument(); // Used tokens (total - balance)
  });

  it('uses correct color for different balance levels', () => {
    // Test low balance (red)
    renderComponent({ balance: 1000, totalAllocation: 10000 });
    const progressLow = document.querySelector('.chakra-progress');
    expect(progressLow).toHaveStyle({ color: 'var(--chakra-colors-red-500)' });

    // Test medium balance (orange)
    renderComponent({ balance: 3000, totalAllocation: 10000 });
    const progressMedium = document.querySelector('.chakra-progress');
    expect(progressMedium).toHaveStyle({ color: 'var(--chakra-colors-orange-500)' });

    // Test high balance (primary)
    renderComponent({ balance: 8000, totalAllocation: 10000 });
    const progressHigh = document.querySelector('.chakra-progress');
    expect(progressHigh).toHaveStyle({ color: 'var(--chakra-colors-primary-500)' });
  });
}); 