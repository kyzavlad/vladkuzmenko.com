import React from 'react';
import {
  Box,
  HStack,
  Text,
  VStack,
  Select,
  useToken,
} from '@chakra-ui/react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from 'recharts';

interface TokenUsageData {
  date: string;
  translations: number;
  clips: number;
  avatars: number;
  total: number;
}

interface FeatureUsage {
  feature: string;
  tokens: number;
  percentage: number;
}

interface TokenUsageProps {
  timeRange: string;
  usageData: TokenUsageData[];
  featureBreakdown: FeatureUsage[];
  onTimeRangeChange: (range: string) => void;
}

export function TokenUsage({
  timeRange,
  usageData,
  featureBreakdown,
  onTimeRangeChange,
}: TokenUsageProps) {
  const [primary500, neutral400] = useToken('colors', ['primary.500', 'neutral.400']);

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
            Token Usage
          </Text>
          <Select
            value={timeRange}
            onChange={(e) => onTimeRangeChange(e.target.value)}
            size="sm"
            w="150px"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
            <option value="1y">Last year</option>
          </Select>
        </HStack>

        <Box h="300px">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={usageData}>
              <defs>
                <linearGradient id="totalGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={primary500} stopOpacity={0.1} />
                  <stop offset="95%" stopColor={primary500} stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="date"
                stroke={neutral400}
                tick={{ fill: neutral400 }}
              />
              <YAxis
                stroke={neutral400}
                tick={{ fill: neutral400 }}
                tickFormatter={(value) => `${value}k`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(26, 32, 44, 0.9)',
                  border: 'none',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: neutral400 }}
              />
              <Area
                type="monotone"
                dataKey="total"
                stroke={primary500}
                fill="url(#totalGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </Box>

        <Box h="200px">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={featureBreakdown}>
              <XAxis
                dataKey="feature"
                stroke={neutral400}
                tick={{ fill: neutral400 }}
              />
              <YAxis
                stroke={neutral400}
                tick={{ fill: neutral400 }}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(26, 32, 44, 0.9)',
                  border: 'none',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: neutral400 }}
              />
              <Legend />
              <Bar
                dataKey="percentage"
                fill={primary500}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </Box>

        <VStack spacing={4}>
          {featureBreakdown.map((feature) => (
            <HStack key={feature.feature} w="full" justify="space-between">
              <Text color="neutral.400">{feature.feature}</Text>
              <Text fontWeight="semibold">
                {feature.tokens.toLocaleString()} tokens ({feature.percentage}%)
              </Text>
            </HStack>
          ))}
        </VStack>
      </VStack>
    </Box>
  );
} 