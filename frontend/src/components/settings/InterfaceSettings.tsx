import React from 'react';
import {
  VStack,
  FormControl,
  FormLabel,
  Switch,
  Select,
  Button,
  HStack,
  Box,
  Text,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  IconButton,
  useDisclosure,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Input,
} from '@chakra-ui/react';
import { FiEdit2, FiTrash2 } from 'react-icons/fi';
import { InterfaceSettings as InterfaceSettingsType } from '../../store/slices/preferencesSlice';

interface InterfaceSettingsProps {
  settings: InterfaceSettingsType;
  onUpdate: (settings: Partial<InterfaceSettingsType>) => void;
  isLoading: boolean;
}

interface ShortcutModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (action: string, shortcut: string) => void;
  initialAction?: string;
  initialShortcut?: string;
}

const ShortcutModal: React.FC<ShortcutModalProps> = ({
  isOpen,
  onClose,
  onSave,
  initialAction = '',
  initialShortcut = '',
}) => {
  const [action, setAction] = React.useState(initialAction);
  const [shortcut, setShortcut] = React.useState(initialShortcut);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(action, shortcut);
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <ModalOverlay />
      <ModalContent>
        <form onSubmit={handleSubmit}>
          <ModalHeader>
            {initialAction ? 'Edit Shortcut' : 'Add New Shortcut'}
          </ModalHeader>
          <ModalBody>
            <VStack spacing={4}>
              <FormControl isRequired>
                <FormLabel>Action</FormLabel>
                <Input
                  value={action}
                  onChange={(e) => setAction(e.target.value)}
                  placeholder="e.g., Save File"
                />
              </FormControl>
              <FormControl isRequired>
                <FormLabel>Shortcut</FormLabel>
                <Input
                  value={shortcut}
                  onChange={(e) => setShortcut(e.target.value)}
                  placeholder="e.g., Ctrl+S"
                />
              </FormControl>
            </VStack>
          </ModalBody>
          <ModalFooter>
            <Button variant="ghost" mr={3} onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" colorScheme="blue">
              Save
            </Button>
          </ModalFooter>
        </form>
      </ModalContent>
    </Modal>
  );
};

const InterfaceSettings: React.FC<InterfaceSettingsProps> = ({
  settings,
  onUpdate,
  isLoading,
}) => {
  const [localSettings, setLocalSettings] = React.useState(settings);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [editingShortcut, setEditingShortcut] = React.useState<{
    action: string;
    shortcut: string;
  } | null>(null);

  React.useEffect(() => {
    setLocalSettings(settings);
  }, [settings]);

  const handleChange = (
    field: keyof InterfaceSettingsType,
    value: any
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

  const handleShortcutSave = (action: string, shortcut: string) => {
    const newShortcuts = {
      ...localSettings.customShortcuts,
      [action]: shortcut,
    };
    handleChange('customShortcuts', newShortcuts);
  };

  const handleShortcutDelete = (action: string) => {
    const newShortcuts = { ...localSettings.customShortcuts };
    delete newShortcuts[action];
    handleChange('customShortcuts', newShortcuts);
  };

  const handleShortcutEdit = (action: string, shortcut: string) => {
    setEditingShortcut({ action, shortcut });
    onOpen();
  };

  return (
    <form onSubmit={handleSubmit}>
      <VStack spacing={6} align="stretch">
        <Box>
          <Text fontSize="lg" fontWeight="medium" mb={4}>
            Interface Preferences
          </Text>
          <Text fontSize="sm" color="gray.600" mb={6}>
            Customize how you interact with the platform
          </Text>
        </Box>

        <FormControl display="flex" alignItems="center">
          <FormLabel mb="0">Default Grid View</FormLabel>
          <Switch
            isChecked={localSettings.defaultView === 'grid'}
            onChange={(e) =>
              handleChange('defaultView', e.target.checked ? 'grid' : 'list')
            }
            isDisabled={isLoading}
          />
        </FormControl>

        <FormControl display="flex" alignItems="center">
          <FormLabel mb="0">Compact Mode</FormLabel>
          <Switch
            isChecked={localSettings.compactMode}
            onChange={(e) => handleChange('compactMode', e.target.checked)}
            isDisabled={isLoading}
          />
        </FormControl>

        <FormControl display="flex" alignItems="center">
          <FormLabel mb="0">Show Tutorials</FormLabel>
          <Switch
            isChecked={localSettings.showTutorials}
            onChange={(e) => handleChange('showTutorials', e.target.checked)}
            isDisabled={isLoading}
          />
        </FormControl>

        <FormControl display="flex" alignItems="center">
          <FormLabel mb="0">Enable Keyboard Shortcuts</FormLabel>
          <Switch
            isChecked={localSettings.enableKeyboardShortcuts}
            onChange={(e) =>
              handleChange('enableKeyboardShortcuts', e.target.checked)
            }
            isDisabled={isLoading}
          />
        </FormControl>

        <Box>
          <HStack justify="space-between" mb={4}>
            <Text fontSize="md" fontWeight="medium">
              Custom Shortcuts
            </Text>
            <Button
              size="sm"
              onClick={() => {
                setEditingShortcut(null);
                onOpen();
              }}
              isDisabled={!localSettings.enableKeyboardShortcuts}
            >
              Add Shortcut
            </Button>
          </HStack>

          <Table size="sm">
            <Thead>
              <Tr>
                <Th>Action</Th>
                <Th>Shortcut</Th>
                <Th width="100px">Actions</Th>
              </Tr>
            </Thead>
            <Tbody>
              {Object.entries(localSettings.customShortcuts).map(
                ([action, shortcut]) => (
                  <Tr key={action}>
                    <Td>{action}</Td>
                    <Td>
                      <code>{shortcut}</code>
                    </Td>
                    <Td>
                      <HStack spacing={2}>
                        <IconButton
                          aria-label="Edit shortcut"
                          icon={<FiEdit2 />}
                          size="sm"
                          variant="ghost"
                          onClick={() => handleShortcutEdit(action, shortcut)}
                        />
                        <IconButton
                          aria-label="Delete shortcut"
                          icon={<FiTrash2 />}
                          size="sm"
                          variant="ghost"
                          colorScheme="red"
                          onClick={() => handleShortcutDelete(action)}
                        />
                      </HStack>
                    </Td>
                  </Tr>
                )
              )}
            </Tbody>
          </Table>
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

      <ShortcutModal
        isOpen={isOpen}
        onClose={onClose}
        onSave={handleShortcutSave}
        initialAction={editingShortcut?.action}
        initialShortcut={editingShortcut?.shortcut}
      />
    </form>
  );
};

export default InterfaceSettings; 