import React, { useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  Box,
  VStack,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  useToast,
  Heading,
  Text,
} from '@chakra-ui/react';
import { RootState, AppDispatch } from '../../store';
import {
  updateTheme,
  updateInterface,
  updateNotifications,
  updatePrivacy,
  fetchPreferences,
} from '../../store/slices/preferencesSlice';
import ThemeSettings from './ThemeSettings';
import InterfaceSettings from './InterfaceSettings';
import NotificationSettings from './NotificationSettings';
import PrivacySettings from './PrivacySettings';

const UserSettings: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const toast = useToast();
  const preferences = useSelector((state: RootState) => state.preferences);
  const { isLoading, error } = preferences;

  useEffect(() => {
    dispatch(fetchPreferences())
      .unwrap()
      .catch((error) => {
        toast({
          title: 'Error loading preferences',
          description: error,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      });
  }, [dispatch, toast]);

  const handleThemeUpdate = (themeSettings: any) => {
    dispatch(updateTheme(themeSettings))
      .then(() => {
        toast({
          title: 'Theme updated',
          status: 'success',
          duration: 2000,
        });
      })
      .catch((error) => {
        toast({
          title: 'Error updating theme',
          description: error.message,
          status: 'error',
          duration: 5000,
        });
      });
  };

  const handleInterfaceUpdate = (interfaceSettings: any) => {
    dispatch(updateInterface(interfaceSettings))
      .then(() => {
        toast({
          title: 'Interface preferences updated',
          status: 'success',
          duration: 2000,
        });
      })
      .catch((error) => {
        toast({
          title: 'Error updating interface preferences',
          description: error.message,
          status: 'error',
          duration: 5000,
        });
      });
  };

  const handleNotificationsUpdate = (notificationSettings: any) => {
    dispatch(updateNotifications(notificationSettings))
      .then(() => {
        toast({
          title: 'Notification preferences updated',
          status: 'success',
          duration: 2000,
        });
      })
      .catch((error) => {
        toast({
          title: 'Error updating notification preferences',
          description: error.message,
          status: 'error',
          duration: 5000,
        });
      });
  };

  const handlePrivacyUpdate = (privacySettings: any) => {
    dispatch(updatePrivacy(privacySettings))
      .then(() => {
        toast({
          title: 'Privacy settings updated',
          status: 'success',
          duration: 2000,
        });
      })
      .catch((error) => {
        toast({
          title: 'Error updating privacy settings',
          description: error.message,
          status: 'error',
          duration: 5000,
        });
      });
  };

  if (error) {
    return (
      <Box p={4}>
        <Text color="red.500">Error: {error}</Text>
      </Box>
    );
  }

  return (
    <Box p={4} maxW="container.lg" mx="auto">
      <VStack spacing={6} align="stretch">
        <Heading size="lg">User Settings</Heading>
        <Text color="gray.600">
          Customize your experience by adjusting your preferences below
        </Text>

        <Tabs isLazy>
          <TabList>
            <Tab>Theme</Tab>
            <Tab>Interface</Tab>
            <Tab>Notifications</Tab>
            <Tab>Privacy</Tab>
          </TabList>

          <TabPanels>
            <TabPanel>
              <ThemeSettings
                settings={preferences.theme}
                onUpdate={handleThemeUpdate}
                isLoading={isLoading}
              />
            </TabPanel>
            <TabPanel>
              <InterfaceSettings
                settings={preferences.interface}
                onUpdate={handleInterfaceUpdate}
                isLoading={isLoading}
              />
            </TabPanel>
            <TabPanel>
              <NotificationSettings
                settings={preferences.notifications}
                onUpdate={handleNotificationsUpdate}
                isLoading={isLoading}
              />
            </TabPanel>
            <TabPanel>
              <PrivacySettings
                settings={preferences.privacy}
                onUpdate={handlePrivacyUpdate}
                isLoading={isLoading}
              />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </VStack>
    </Box>
  );
};

export default UserSettings; 