describe('Token Purchase Flow', () => {
  beforeEach(() => {
    cy.visit('/tokens');
    // Mock Stripe.js
    cy.window().then((win) => {
      win.Stripe = () => ({
        elements: () => ({
          create: () => ({
            mount: cy.stub(),
            destroy: cy.stub(),
          }),
        }),
        createPaymentMethod: () =>
          Promise.resolve({ paymentMethod: { id: 'test_payment_method' } }),
      });
    });
  });

  it('displays available token packages', () => {
    cy.get('[data-testid="token-package"]').should('have.length.at.least', 1);
    cy.contains('Basic').should('be.visible');
    cy.contains('Pro').should('be.visible');
    cy.contains('Enterprise').should('be.visible');
  });

  it('shows most popular package highlighted', () => {
    cy.get('[data-testid="token-package"]')
      .contains('Pro')
      .parent()
      .should('have.class', 'popular');
  });

  it('opens purchase modal when clicking purchase button', () => {
    cy.contains('Purchase').first().click();
    cy.get('[role="dialog"]').should('be.visible');
    cy.contains('Complete Purchase').should('be.visible');
  });

  it('completes purchase flow successfully', () => {
    // Intercept API calls
    cy.intercept('POST', '/api/tokens/purchase', {
      statusCode: 200,
      body: {
        success: true,
        tokens: 1000,
        balance: 1000,
        transactionId: 'test_tx',
      },
    }).as('purchaseRequest');

    // Start purchase flow
    cy.contains('Purchase').first().click();
    cy.get('[role="dialog"]').should('be.visible');

    // Fill payment details (mock)
    cy.get('form').submit();

    // Verify API call and success state
    cy.wait('@purchaseRequest');
    cy.get('[role="dialog"]').should('not.exist');
    cy.contains('Purchase successful').should('be.visible');
  });

  it('handles payment errors gracefully', () => {
    // Intercept API calls with error
    cy.intercept('POST', '/api/tokens/purchase', {
      statusCode: 400,
      body: { error: 'Payment failed' },
    }).as('failedPurchase');

    // Start purchase flow
    cy.contains('Purchase').first().click();
    cy.get('[role="dialog"]').should('be.visible');

    // Submit form
    cy.get('form').submit();

    // Verify error handling
    cy.wait('@failedPurchase');
    cy.contains('Payment failed').should('be.visible');
    cy.get('[role="dialog"]').should('be.visible');
  });

  it('updates token balance after successful purchase', () => {
    // Get initial balance
    cy.get('[data-testid="token-balance"]').invoke('text').as('initialBalance');

    // Complete purchase
    cy.contains('Purchase').first().click();
    cy.get('form').submit();

    // Verify balance update
    cy.get('[data-testid="token-balance"]')
      .invoke('text')
      .then((newBalance) => {
        cy.get('@initialBalance').then((initialBalance) => {
          expect(Number(newBalance)).to.be.greaterThan(Number(initialBalance));
        });
      });
  });
}); 