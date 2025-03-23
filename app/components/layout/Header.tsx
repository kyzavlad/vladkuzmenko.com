import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { Box, Flex, Button, IconButton, useDisclosure, Stack, Text } from '@chakra-ui/react';
import { HamburgerIcon, CloseIcon, ChevronDownIcon } from '@chakra-ui/icons';

interface NavItemProps {
  href: string;
  children: React.ReactNode;
  hasDropdown?: boolean;
}

const Header: React.FC = () => {
  const { isOpen, onToggle } = useDisclosure();
  const [productDropdownOpen, setProductDropdownOpen] = React.useState(false);
  const router = useRouter();

  const NavItem: React.FC<NavItemProps> = ({ href, children, hasDropdown = false }) => (
    <Box
      position="relative"
      onMouseEnter={() => hasDropdown && setProductDropdownOpen(true)}
      onMouseLeave={() => hasDropdown && setProductDropdownOpen(false)}
    >
      <Link href={href} passHref>
        <Text
          px={4}
          py={2}
          fontSize="sm"
          fontWeight="medium"
          color="white"
          _hover={{ color: '#3a86ff' }}
          display="flex"
          alignItems="center"
        >
          {children}
          {hasDropdown && <ChevronDownIcon ml={1} />}
        </Text>
      </Link>
      {hasDropdown && productDropdownOpen && (
        <Box
          position="absolute"
          top="100%"
          left={0}
          bg="#1a1a1a"
          border="1px solid #333"
          borderRadius="md"
          p={2}
          minW="200px"
          zIndex={10}
        >
          <Link href="/platform/dashboard" passHref>
            <Text
              p={2}
              fontSize="sm"
              color="white"
              _hover={{ bg: '#333', color: '#3a86ff' }}
            >
              AI Video Platform
            </Text>
          </Link>
        </Box>
      )}
    </Box>
  );

  return (
    <Box bg="#1a1a1a" px={4} position="fixed" w="100%" top={0} zIndex={1000}>
      <Flex h={16} alignItems="center" justifyContent="space-between">
        <Flex alignItems="center">
          <Link href="/" passHref>
            <Box mr={8}>
              <img src="/platform/logo.svg" alt="bolt.new" height={32} width={120} />
            </Box>
          </Link>

          <Stack
            direction="row"
            spacing={8}
            alignItems="center"
            display={{ base: 'none', md: 'flex' }}
          >
            <NavItem href="/">Home</NavItem>
            <NavItem href="#" hasDropdown>Product</NavItem>
            <NavItem href="/company">Company</NavItem>
            <NavItem href="/docs">Docs</NavItem>
          </Stack>
        </Flex>

        <Flex alignItems="center">
          <Stack
            direction="row"
            spacing={4}
            display={{ base: 'none', md: 'flex' }}
          >
            <Button
              variant="ghost"
              color="white"
              _hover={{ bg: '#333' }}
              onClick={() => router.push('/platform/dashboard')}
            >
              Dashboard
            </Button>
            <Button
              bg="#3a86ff"
              color="white"
              _hover={{ bg: '#2563eb' }}
              onClick={() => router.push('/platform/dashboard')}
            >
              Get Started
            </Button>
          </Stack>

          <IconButton
            display={{ base: 'flex', md: 'none' }}
            onClick={onToggle}
            icon={isOpen ? <CloseIcon /> : <HamburgerIcon />}
            variant="ghost"
            color="white"
            aria-label="Toggle navigation"
            ml={4}
          />
        </Flex>
      </Flex>

      {/* Mobile menu */}
      {isOpen && (
        <Box pb={4} display={{ md: 'none' }}>
          <Stack spacing={4}>
            <Link href="/" passHref>
              <Text p={2} color="white">Home</Text>
            </Link>
            <Text
              p={2}
              color="white"
              onClick={() => setProductDropdownOpen(!productDropdownOpen)}
              cursor="pointer"
            >
              Product <ChevronDownIcon />
            </Text>
            {productDropdownOpen && (
              <Link href="/platform/dashboard" passHref>
                <Text pl={4} py={2} color="white">AI Video Platform</Text>
              </Link>
            )}
            <Link href="/company" passHref>
              <Text p={2} color="white">Company</Text>
            </Link>
            <Link href="/docs" passHref>
              <Text p={2} color="white">Docs</Text>
            </Link>
            <Button
              w="full"
              bg="#3a86ff"
              color="white"
              _hover={{ bg: '#2563eb' }}
              onClick={() => router.push('/platform/dashboard')}
            >
              Get Started
            </Button>
          </Stack>
        </Box>
      )}
    </Box>
  );
};

export default Header; 