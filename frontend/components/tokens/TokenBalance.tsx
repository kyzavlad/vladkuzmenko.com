import React from 'react';
import {
  Box,
  CircularProgress,
  CircularProgressLabel,
  HStack,
  Text,
  VStack,
  Button,
  useToken,
  Alert,
  AlertIcon,
} from '@chakra-ui/react';
import { FiZap, FiAlertTriangle } from 'react-icons/fi';

interface TokenBalanceProps {
  balance: number;
  totalAllocation: number;
  lowBalanceThreshold: number;
  onPurchase: () => void;
}

export function TokenBalance({
  balance,
  totalAllocation,
  lowBalanceThreshold,
  onPurchase,
}: TokenBalanceProps) {
  const [primary500] = useToken('colors', ['primary.500']);
  const percentage = (balance / totalAllocation) * 100;
  const isLowBalance = balance <= lowBalanceThreshold;

  const getStatusColor = (percent: number): string => {
    if (percent <= 20) return 'red.500';
    if (percent <= 40) return 'orange.500';
    return primary500;
  };

  return (
    <Box
      bg="neutral.800"
      borderRadius="xl"
      p={6}
      position="relative"
      overflow="hidden"
    >
      <VStack spacing={6} align="stretch">
        <HStack justify="space-between">
          <Text fontSize="xl" fontWeight="semibold">
            Token Balance
          </Text>
          <Button
            leftIcon={<FiZap />}
            colorScheme="primary"
            size="sm"
            onClick={onPurchase}
          >
            Purchase Tokens
          </Button>
        </HStack>

        <HStack spacing={8} align="center">
          <CircularProgress
            value={percentage}
            size="120px"
            thickness="8px"
            color={getStatusColor(percentage)}
          >
            <CircularProgressLabel>
              <VStack spacing={0}>
                <Text fontSize="xl" fontWeight="bold">
                  {balance.toLocaleString()}
                </Text>
                <Text fontSize="xs" color="neutral.400">
                  tokens
                </Text>
              </VStack>
            </CircularProgressLabel>
          </CircularProgress>

          <VStack align="start" flex={1} spacing={4}>
            <Box>
              <Text fontSize="sm" color="neutral.400">
                Total Allocation
              </Text>
              <Text fontSize="lg" fontWeight="semibold">
                {totalAllocation.toLocaleString()} tokens
              </Text>
            </Box>

            <Box>
              <Text fontSize="sm" color="neutral.400">
                Used Tokens
              </Text>
              <Text fontSize="lg" fontWeight="semibold">
                {(totalAllocation - balance).toLocaleString()} tokens
              </Text>
            </Box>
          </VStack>
        </HStack>

        {isLowBalance && (
          <Alert status="warning" borderRadius="md">
            <AlertIcon as={FiAlertTriangle} />
            <Text fontSize="sm">
              Your token balance is running low. Consider purchasing more tokens to
              avoid service interruption.
            </Text>
          </Alert>
        )}
      </VStack>
    </Box>
  );
} 