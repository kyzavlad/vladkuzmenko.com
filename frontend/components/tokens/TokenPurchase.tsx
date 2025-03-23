import React, { useState } from 'react';
import {
  Box,
  Button,
  Grid,
  HStack,
  Text,
  VStack,
  useToken,
  Badge,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  useDisclosure,
} from '@chakra-ui/react';
import { FiCheck, FiZap, FiCreditCard } from 'react-icons/fi';
import { loadStripe } from '@stripe/stripe-js';
import {
  Elements,
  CardElement,
  useStripe,
  useElements,
} from '@stripe/react-stripe-js';

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_KEY!);

interface TokenPackage {
  id: string;
  name: string;
  tokens: number;
  price: number;
  popular?: boolean;
  savings?: number;
}

interface TokenPurchaseProps {
  packages: TokenPackage[];
  onPurchaseComplete: (tokens: number) => void;
}

function CheckoutForm({
  selectedPackage,
  onSuccess,
  onClose,
}: {
  selectedPackage: TokenPackage;
  onSuccess: (tokens: number) => void;
  onClose: () => void;
}) {
  const stripe = useStripe();
  const elements = useElements();
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!stripe || !elements) return;

    setIsProcessing(true);
    setError(null);

    try {
      const { error: stripeError, paymentMethod } = await stripe.createPaymentMethod({
        type: 'card',
        card: elements.getElement(CardElement)!,
      });

      if (stripeError) {
        setError(stripeError.message);
        return;
      }

      // Call your backend to process the payment
      const response = await fetch('/api/tokens/purchase', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          packageId: selectedPackage.id,
          paymentMethodId: paymentMethod.id,
        }),
      });

      const result = await response.json();

      if (result.error) {
        setError(result.error);
      } else {
        onSuccess(selectedPackage.tokens);
        onClose();
      }
    } catch (err) {
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <VStack spacing={6}>
        <Box w="full">
          <Text mb={2} fontSize="sm" fontWeight="medium">
            Card Details
          </Text>
          <Box
            borderWidth={1}
            borderColor="neutral.700"
            borderRadius="md"
            p={4}
          >
            <CardElement
              options={{
                style: {
                  base: {
                    fontSize: '16px',
                    color: '#fff',
                    '::placeholder': {
                      color: '#718096',
                    },
                  },
                },
              }}
            />
          </Box>
        </Box>

        {error && (
          <Text color="red.500" fontSize="sm">
            {error}
          </Text>
        )}

        <Button
          type="submit"
          colorScheme="primary"
          size="lg"
          w="full"
          leftIcon={<FiCreditCard />}
          isLoading={isProcessing}
          loadingText="Processing..."
        >
          Pay ${selectedPackage.price}
        </Button>
      </VStack>
    </form>
  );
}

export function TokenPurchase({ packages, onPurchaseComplete }: TokenPurchaseProps) {
  const [selectedPackage, setSelectedPackage] = useState<TokenPackage | null>(null);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [primary500] = useToken('colors', ['primary.500']);

  const handlePackageSelect = (pkg: TokenPackage) => {
    setSelectedPackage(pkg);
    onOpen();
  };

  return (
    <>
      <Grid
        templateColumns={{ base: '1fr', md: 'repeat(2, 1fr)', lg: 'repeat(3, 1fr)' }}
        gap={6}
      >
        {packages.map((pkg) => (
          <Box
            key={pkg.id}
            bg="neutral.800"
            borderRadius="xl"
            p={6}
            position="relative"
            borderWidth={2}
            borderColor={pkg.popular ? primary500 : 'transparent'}
          >
            {pkg.popular && (
              <Badge
                colorScheme="primary"
                position="absolute"
                top={4}
                right={4}
              >
                Most Popular
              </Badge>
            )}

            <VStack spacing={4} align="stretch">
              <Text fontSize="xl" fontWeight="semibold">
                {pkg.name}
              </Text>

              <HStack>
                <Text fontSize="3xl" fontWeight="bold">
                  ${pkg.price}
                </Text>
                {pkg.savings && (
                  <Badge colorScheme="green">Save {pkg.savings}%</Badge>
                )}
              </HStack>

              <Text color="neutral.400">
                {pkg.tokens.toLocaleString()} tokens
              </Text>

              <Button
                leftIcon={<FiZap />}
                colorScheme={pkg.popular ? 'primary' : 'neutral'}
                size="lg"
                onClick={() => handlePackageSelect(pkg)}
              >
                Purchase
              </Button>
            </VStack>
          </Box>
        ))}
      </Grid>

      <Modal isOpen={isOpen} onClose={onClose} size="md">
        <ModalOverlay />
        <ModalContent bg="neutral.900">
          <ModalHeader>Complete Purchase</ModalHeader>
          <ModalCloseButton />
          <ModalBody pb={6}>
            {selectedPackage && (
              <Elements stripe={stripePromise}>
                <CheckoutForm
                  selectedPackage={selectedPackage}
                  onSuccess={onPurchaseComplete}
                  onClose={onClose}
                />
              </Elements>
            )}
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
} 