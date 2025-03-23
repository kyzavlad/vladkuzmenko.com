import React from 'react';
import {
  VStack,
  FormControl,
  FormLabel,
  FormHelperText,
  Switch,
  Button,
  HStack,
  Box,
  Text,
  Alert,
  AlertIcon,
  AlertTitle,
  AlertDescription,
  useToast,
} from '@chakra-ui/react';
import { PrivacySettings as PrivacySettingsType } from '../../store/slices/preferencesSlice';

interface PrivacySettingsProps {
  settings: PrivacySettingsType;
  onUpdate: (settings: Partial<PrivacySettingsType>) => void;
  isLoading: boolean;
}

const PrivacySettings: React.FC<PrivacySettingsProps> = ({
  settings,
  onUpdate,
  isLoading,
}) => {
  const [localSettings, setLocalSettings] = React.useState(settings);
  const [hasChanges, setHasChanges] = React.useState(false);
  const toast = useToast();

  React.useEffect(() => {
    setLocalSettings(settings);
    setHasChanges(false);
  }, [settings]);

  const handleChange = (field: keyof PrivacySettingsType, value: boolean) => {
    const newSettings = { ...localSettings, [field]: value };
    setLocalSettings(newSettings);
    setHasChanges(true);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await onUpdate(localSettings);
      toast({
        title: 'Privacy settings updated',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
      setHasChanges(false);
    } catch (error) {
      toast({
        title: 'Error updating settings',
        description: error instanceof Error ? error.message : 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleReset = () => {
    setLocalSettings(settings);
    setHasChanges(false);
    toast({
      title: 'Settings reset',
      status: 'info',
      duration: 2000,
    });
  };

  return (
    <form onSubmit={handleSubmit}>
      <VStack spacing={6} align="stretch">
        <Box>
          <Text fontSize="lg" fontWeight="medium" mb={4}>
            Privacy & Security Settings
          </Text>
          <Text fontSize="sm" color="gray.600" mb={6}>
            Control how your data is used and shared
          </Text>
        </Box>

        {hasChanges && (
          <Alert status="info" borderRadius="md">
            <AlertIcon />
            <AlertTitle mr={2}>Unsaved changes</AlertTitle>
            <AlertDescription>
              You have made changes to your privacy settings that haven't been saved
            </AlertDescription>
          </Alert>
        )}

        <FormControl>
          <FormLabel>Usage Data Collection</FormLabel>
          <Switch
            isChecked={localSettings.shareUsageData}
            onChange={(e) => handleChange('shareUsageData', e.target.checked)}
            isDisabled={isLoading}
          />
          <FormHelperText>
            Allow us to collect anonymous usage data to improve our services
          </FormHelperText>
        </FormControl>

        <FormControl>
          <FormLabel>Analytics</FormLabel>
          <Switch
            isChecked={localSettings.shareAnalytics}
            onChange={(e) => handleChange('shareAnalytics', e.target.checked)}
            isDisabled={isLoading}
          />
          <FormHelperText>
            Share analytics data to help us understand how you use our platform
          </FormHelperText>
        </FormControl>

        <FormControl>
          <FormLabel>Marketing Communications</FormLabel>
          <Switch
            isChecked={localSettings.marketingCommunication}
            onChange={(e) =>
              handleChange('marketingCommunication', e.target.checked)
            }
            isDisabled={isLoading}
          />
          <FormHelperText>
            Receive marketing communications about new features and updates
          </FormHelperText>
        </FormControl>

        <Box>
          <Text fontSize="sm" color="gray.600">
            Note: Your privacy is important to us. We never share your personal
            information with third parties without your explicit consent.
          </Text>
        </Box>

        <HStack spacing={4} justify="flex-end">
          <Button
            variant="outline"
            onClick={handleReset}
            isDisabled={isLoading || !hasChanges}
          >
            Reset
          </Button>
          <Button
            type="submit"
            colorScheme="blue"
            isLoading={isLoading}
            isDisabled={!hasChanges}
          >
            Save Changes
          </Button>
        </HStack>
      </VStack>
    </form>
  );
};

export default PrivacySettings; 