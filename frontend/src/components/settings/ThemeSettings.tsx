import React from 'react';
import {
  VStack,
  FormControl,
  FormLabel,
  Select,
  Input,
  Button,
  useColorMode,
  HStack,
  Box,
  Text,
} from '@chakra-ui/react';
import { ThemeSettings as ThemeSettingsType } from '../../store/slices/preferencesSlice';

interface ThemeSettingsProps {
  settings: ThemeSettingsType;
  onUpdate: (settings: Partial<ThemeSettingsType>) => void;
  isLoading: boolean;
}

const ThemeSettings: React.FC<ThemeSettingsProps> = ({
  settings,
  onUpdate,
  isLoading,
}) => {
  const { colorMode, toggleColorMode } = useColorMode();
  const [localSettings, setLocalSettings] = React.useState(settings);

  React.useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  const handleChange = (
    field: keyof ThemeSettingsType,
    value: string
  ) => {
    const newSettings = { ...localSettings, [field]: value };
    setLocalSettings(newSettings);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onUpdate(localSettings);
  };

  const handleReset = () => {
    setLocalSettings(settings);
  };

  return (
    <form onSubmit={handleSubmit}>
      <VStack spacing={6} align="stretch">
        <Box>
          <Text fontSize="lg" fontWeight="medium" mb={4}>
            Theme Preferences
          </Text>
          <Text fontSize="sm" color="gray.600" mb={6}>
            Customize the appearance of your interface
          </Text>
        </Box>

        <FormControl>
          <FormLabel>Color Mode</FormLabel>
          <Select
            value={localSettings.mode}
            onChange={(e) => handleChange('mode', e.target.value)}
            disabled={isLoading}
          >
            <option value="light">Light</option>
            <option value="dark">Dark</option>
            <option value="system">System</option>
          </Select>
        </FormControl>

        <FormControl>
          <FormLabel>Primary Color</FormLabel>
          <Input
            type="color"
            value={localSettings.primaryColor}
            onChange={(e) => handleChange('primaryColor', e.target.value)}
            disabled={isLoading}
          />
        </FormControl>

        <FormControl>
          <FormLabel>Font Size</FormLabel>
          <Select
            value={localSettings.fontSize}
            onChange={(e) => handleChange('fontSize', e.target.value)}
            disabled={isLoading}
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
          </Select>
        </FormControl>

        <Box>
          <Text fontSize="sm" color="gray.600" mb={2}>
            Preview your changes in real-time
          </Text>
          <Box
            p={4}
            borderRadius="md"
            bg={colorMode === 'light' ? 'gray.50' : 'gray.700'}
            border="1px solid"
            borderColor={colorMode === 'light' ? 'gray.200' : 'gray.600'}
          >
            <Text
              fontSize={
                localSettings.fontSize === 'small'
                  ? 'sm'
                  : localSettings.fontSize === 'large'
                  ? 'lg'
                  : 'md'
              }
              color={localSettings.primaryColor}
            >
              Sample text with your selected preferences
            </Text>
          </Box>
        </Box>

        <HStack spacing={4} justify="flex-end">
          <Button
            variant="outline"
            onClick={handleReset}
            isDisabled={isLoading}
          >
            Reset
          </Button>
          <Button
            type="submit"
            colorScheme="blue"
            isLoading={isLoading}
          >
            Save Changes
          </Button>
        </HStack>
      </VStack>
    </form>
  );
};

export default ThemeSettings; 