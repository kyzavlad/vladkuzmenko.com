import React from 'react';
import {
  VStack,
  FormControl,
  FormLabel,
  Switch,
  Button,
  HStack,
  Box,
  Text,
  Divider,
  useToast,
} from '@chakra-ui/react';
import { NotificationSettings as NotificationSettingsType } from '../../store/slices/preferencesSlice';

interface NotificationSettingsProps {
  settings: NotificationSettingsType;
  onUpdate: (settings: Partial<NotificationSettingsType>) => void;
  isLoading: boolean;
}

const NotificationSettings: React.FC<NotificationSettingsProps> = ({
  settings,
  onUpdate,
  isLoading,
}) => {
  const [localSettings, setLocalSettings] = React.useState(settings);
  const toast = useToast();

  React.useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  const handleEmailChange = (field: keyof NotificationSettingsType['email'], value: boolean) => {
    const newSettings = {
      ...localSettings,
      email: {
        ...localSettings.email,
        [field]: value,
      },
    };
    setLocalSettings(newSettings);
  };

  const handleInAppChange = (field: keyof NotificationSettingsType['inApp'], value: boolean) => {
    const newSettings = {
      ...localSettings,
      inApp: {
        ...localSettings.inApp,
        [field]: value,
      },
    };
    setLocalSettings(newSettings);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await onUpdate(localSettings);
      toast({
        title: 'Notification preferences updated',
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      toast({
        title: 'Error updating preferences',
        description: error instanceof Error ? error.message : 'An error occurred',
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  const handleReset = () => {
    setLocalSettings(settings);
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
            Notification Preferences
          </Text>
          <Text fontSize="sm" color="gray.600" mb={6}>
            Choose how and when you want to be notified
          </Text>
        </Box>

        <Box>
          <Text fontSize="md" fontWeight="medium" mb={4}>
            Email Notifications
          </Text>
          <VStack spacing={4} align="stretch">
            <FormControl display="flex" alignItems="center">
              <FormLabel mb="0">Processing Complete</FormLabel>
              <Switch
                isChecked={localSettings.email.processComplete}
                onChange={(e) =>
                  handleEmailChange('processComplete', e.target.checked)
                }
                isDisabled={isLoading}
              />
            </FormControl>

            <FormControl display="flex" alignItems="center">
              <FormLabel mb="0">Low Balance Alert</FormLabel>
              <Switch
                isChecked={localSettings.email.lowBalance}
                onChange={(e) =>
                  handleEmailChange('lowBalance', e.target.checked)
                }
                isDisabled={isLoading}
              />
            </FormControl>

            <FormControl display="flex" alignItems="center">
              <FormLabel mb="0">Newsletter</FormLabel>
              <Switch
                isChecked={localSettings.email.newsletter}
                onChange={(e) =>
                  handleEmailChange('newsletter', e.target.checked)
                }
                isDisabled={isLoading}
              />
            </FormControl>

            <FormControl display="flex" alignItems="center">
              <FormLabel mb="0">Product Updates</FormLabel>
              <Switch
                isChecked={localSettings.email.productUpdates}
                onChange={(e) =>
                  handleEmailChange('productUpdates', e.target.checked)
                }
                isDisabled={isLoading}
              />
            </FormControl>
          </VStack>
        </Box>

        <Divider />

        <Box>
          <Text fontSize="md" fontWeight="medium" mb={4}>
            In-App Notifications
          </Text>
          <VStack spacing={4} align="stretch">
            <FormControl display="flex" alignItems="center">
              <FormLabel mb="0">Processing Complete</FormLabel>
              <Switch
                isChecked={localSettings.inApp.processComplete}
                onChange={(e) =>
                  handleInAppChange('processComplete', e.target.checked)
                }
                isDisabled={isLoading}
              />
            </FormControl>

            <FormControl display="flex" alignItems="center">
              <FormLabel mb="0">Low Balance Alert</FormLabel>
              <Switch
                isChecked={localSettings.inApp.lowBalance}
                onChange={(e) =>
                  handleInAppChange('lowBalance', e.target.checked)
                }
                isDisabled={isLoading}
              />
            </FormControl>

            <FormControl display="flex" alignItems="center">
              <FormLabel mb="0">Tips & Suggestions</FormLabel>
              <Switch
                isChecked={localSettings.inApp.tips}
                onChange={(e) => handleInAppChange('tips', e.target.checked)}
                isDisabled={isLoading}
              />
            </FormControl>
          </VStack>
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

export default NotificationSettings; 