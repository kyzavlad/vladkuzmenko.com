import React from 'react';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { render } from '../../../utils/test-utils';
import { SearchBar } from '../SearchBar';
import { useDispatch, useSelector } from 'react-redux';
import { searchContent, clearResults } from '../../../store/slices/searchSlice';
import userEvent from '@testing-library/user-event';

// Mock Redux hooks
jest.mock('react-redux', () => ({
  useDispatch: jest.fn(),
  useSelector: jest.fn(),
}));

const mockSearchResults = [
  {
    id: '1',
    type: 'video',
    title: 'Test Video',
    description: 'A test video description',
    path: '/videos/test',
    timestamp: '2024-01-20T12:00:00Z',
  },
];

describe('SearchBar Redux Integration', () => {
  const mockDispatch = jest.fn();
  
  beforeEach(() => {
    (useDispatch as jest.Mock).mockReturnValue(mockDispatch);
    (useSelector as jest.Mock).mockImplementation((selector) =>
      selector({
        search: {
          results: [],
          loading: false,
          error: null,
        },
      })
    );
    jest.clearAllMocks();
  });

  it('dispatches search action on input change', async () => {
    render(<SearchBar />);
    const searchInput = screen.getByPlaceholderText('Search content...');
    
    await userEvent.type(searchInput, 'test');
    
    await waitFor(() => {
      expect(mockDispatch).toHaveBeenCalledWith(
        expect.objectContaining({
          type: searchContent.pending.type,
          meta: expect.objectContaining({
            arg: 'test',
          }),
        })
      );
    });
  });

  it('displays search results from Redux state', async () => {
    (useSelector as jest.Mock).mockImplementation((selector) =>
      selector({
        search: {
          results: mockSearchResults,
          loading: false,
          error: null,
        },
      })
    );

    render(<SearchBar />);
    const searchInput = screen.getByPlaceholderText('Search content...');
    await userEvent.type(searchInput, 'test');

    expect(screen.getByText('Test Video')).toBeInTheDocument();
    expect(screen.getByText('A test video description')).toBeInTheDocument();
  });

  it('shows loading state from Redux', async () => {
    (useSelector as jest.Mock).mockImplementation((selector) =>
      selector({
        search: {
          results: [],
          loading: true,
          error: null,
        },
      })
    );

    render(<SearchBar />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('displays error state from Redux', async () => {
    const errorMessage = 'Search failed';
    (useSelector as jest.Mock).mockImplementation((selector) =>
      selector({
        search: {
          results: [],
          loading: false,
          error: errorMessage,
        },
      })
    );

    render(<SearchBar />);
    expect(screen.getByText(`Error: ${errorMessage}`)).toBeInTheDocument();
  });

  it('dispatches clear results action when input is cleared', async () => {
    render(<SearchBar />);
    const searchInput = screen.getByPlaceholderText('Search content...');
    
    await userEvent.type(searchInput, 'test');
    await userEvent.clear(searchInput);

    expect(mockDispatch).toHaveBeenCalledWith(clearResults());
  });

  it('debounces search dispatches', async () => {
    jest.useFakeTimers();
    render(<SearchBar />);
    const searchInput = screen.getByPlaceholderText('Search content...');

    await userEvent.type(searchInput, 'test query');
    
    // Fast typing shouldn't trigger multiple dispatches
    expect(mockDispatch).not.toHaveBeenCalled();

    // After debounce delay
    jest.runAllTimers();

    expect(mockDispatch).toHaveBeenCalledTimes(1);
    expect(mockDispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: searchContent.pending.type,
        meta: expect.objectContaining({
          arg: 'test query',
        }),
      })
    );

    jest.useRealTimers();
  });

  it('handles keyboard navigation of results', async () => {
    (useSelector as jest.Mock).mockImplementation((selector) =>
      selector({
        search: {
          results: mockSearchResults,
          loading: false,
          error: null,
        },
      })
    );

    render(<SearchBar />);
    const searchInput = screen.getByPlaceholderText('Search content...');
    
    await userEvent.type(searchInput, 'test');
    
    // Press arrow down to highlight first result
    await userEvent.keyboard('{ArrowDown}');
    expect(screen.getByText('Test Video').parentElement).toHaveAttribute('aria-selected', 'true');

    // Press Enter to select
    await userEvent.keyboard('{Enter}');
    expect(mockDispatch).toHaveBeenCalledWith(
      expect.objectContaining({
        type: expect.stringContaining('selectResult'),
        payload: mockSearchResults[0],
      })
    );
  });
}); 