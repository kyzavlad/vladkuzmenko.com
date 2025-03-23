import React from 'react';
import {
  Box,
  Flex,
  HStack,
  IconButton,
  Button,
  Menu,
  MenuButton,
  MenuList,
  MenuItem,
  useColorModeValue,
  useDisclosure,
  useColorMode,
  Avatar,
  Text,
  Badge,
  Drawer,
  DrawerOverlay,
  DrawerContent,
  DrawerHeader,
  DrawerBody,
  VStack,
} from '@chakra-ui/react';
import { useSelector, useDispatch } from 'react-redux';
import {
  FiMenu,
  FiVideo,
  FiScissors,
  FiFolder,
  FiSettings,
  FiUser,
  FiLogOut,
  FiMoon,
  FiSun,
  FiDollarSign,
} from 'react-icons/fi';
import { Link, useLocation } from 'react-router-dom';
import { RootState } from '../../store';
import { logout } from '../../store/slices/authSlice';
import SearchBar from '../search/SearchBar';

interface NavigationItem {
  name: string;
  path: string;
  icon: React.ReactElement;
  badge?: string;
}

const Navigation: React.FC = () => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const { colorMode, toggleColorMode } = useColorMode();
  const dispatch = useDispatch();
  const location = useLocation();
  const { user, isAuthenticated } = useSelector((state: RootState) => state.auth);
  const { balance } = useSelector((state: RootState) => state.tokens);

  const navigationItems: NavigationItem[] = [
    { name: 'Video Editor', path: '/editor', icon: <FiVideo /> },
    { name: 'Clip Generator', path: '/clips', icon: <FiScissors /> },
    { name: 'Asset Library', path: '/assets', icon: <FiFolder /> },
    { name: 'Settings', path: '/settings', icon: <FiSettings /> },
  ];

  const bgColor = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');

  const handleLogout = () => {
    dispatch(logout());
  };

  const NavLink: React.FC<{ item: NavigationItem }> = ({ item }) => {
    const isActive = location.pathname === item.path;
    const activeBg = useColorModeValue('blue.50', 'blue.900');
    const hoverBg = useColorModeValue('gray.100', 'gray.700');

    return (
      <Button
        as={Link}
        to={item.path}
        leftIcon={item.icon}
        variant={isActive ? 'solid' : 'ghost'}
        justifyContent="flex-start"
        w="full"
        bg={isActive ? activeBg : 'transparent'}
        _hover={{ bg: isActive ? activeBg : hoverBg }}
      >
        {item.name}
        {item.badge && (
          <Badge ml={2} colorScheme="blue">
            {item.badge}
          </Badge>
        )}
      </Button>
    );
  };

  const DesktopNav = () => (
    <HStack spacing={4}>
      {navigationItems.map((item) => (
        <NavLink key={item.path} item={item} />
      ))}
    </HStack>
  );

  const MobileNav = () => (
    <VStack align="stretch" spacing={4}>
      {navigationItems.map((item) => (
        <NavLink key={item.path} item={item} />
      ))}
    </VStack>
  );

  return (
    <Box
      as="nav"
      position="fixed"
      w="full"
      bg={bgColor}
      borderBottom="1px"
      borderColor={borderColor}
      zIndex={1000}
    >
      <Flex
        h="16"
        alignItems="center"
        justifyContent="space-between"
        maxW="container.xl"
        mx="auto"
        px={4}
      >
        {/* Mobile menu button */}
        <IconButton
          display={{ base: 'flex', md: 'none' }}
          onClick={onOpen}
          variant="ghost"
          aria-label="Open menu"
          icon={<FiMenu />}
        />

        {/* Logo */}
        <Text
          fontSize="xl"
          fontWeight="bold"
          as={Link}
          to="/"
          _hover={{ textDecoration: 'none' }}
        >
          AI Video Platform
        </Text>

        {/* Desktop Navigation */}
        <HStack spacing={4} display={{ base: 'none', md: 'flex' }}>
          <DesktopNav />
        </HStack>

        {/* Search Bar */}
        <Box flex={1} mx={8} display={{ base: 'none', md: 'block' }}>
          <SearchBar />
        </Box>

        {/* Right Section */}
        <HStack spacing={4}>
          {/* Token Balance */}
          {isAuthenticated && (
            <Button
              leftIcon={<FiDollarSign />}
              variant="ghost"
              as={Link}
              to="/tokens"
            >
              {balance} Tokens
            </Button>
          )}

          {/* Color Mode Toggle */}
          <IconButton
            aria-label="Toggle color mode"
            icon={colorMode === 'light' ? <FiMoon /> : <FiSun />}
            onClick={toggleColorMode}
            variant="ghost"
          />

          {/* User Menu */}
          {isAuthenticated ? (
            <Menu>
              <MenuButton
                as={Button}
                variant="ghost"
                rightIcon={<FiUser />}
              >
                <Avatar size="sm" name={user?.name} src={user?.avatar} />
              </MenuButton>
              <MenuList>
                <MenuItem as={Link} to="/profile" icon={<FiUser />}>
                  Profile
                </MenuItem>
                <MenuItem as={Link} to="/settings" icon={<FiSettings />}>
                  Settings
                </MenuItem>
                <MenuItem onClick={handleLogout} icon={<FiLogOut />}>
                  Logout
                </MenuItem>
              </MenuList>
            </Menu>
          ) : (
            <Button as={Link} to="/login" colorScheme="blue">
              Sign In
            </Button>
          )}
        </HStack>
      </Flex>

      {/* Mobile Navigation Drawer */}
      <Drawer isOpen={isOpen} placement="left" onClose={onClose}>
        <DrawerOverlay />
        <DrawerContent>
          <DrawerHeader borderBottomWidth="1px">Menu</DrawerHeader>
          <DrawerBody>
            <VStack spacing={4} align="stretch">
              <SearchBar />
              <MobileNav />
            </VStack>
          </DrawerBody>
        </DrawerContent>
      </Drawer>
    </Box>
  );
};

export default Navigation; 