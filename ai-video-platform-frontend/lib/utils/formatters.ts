/**
 * Format a date string to a human-readable format
 * @param dateString - The date string to format
 * @param options - Format options
 * @returns Formatted date string
 */
export function formatDate(
  dateString: string | Date,
  options: Intl.DateTimeFormatOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  }
): string {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat('en-US', options).format(date);
}

/**
 * Format a number to a human-readable file size
 * @param bytes - The file size in bytes
 * @param decimals - The number of decimal places to include
 * @returns Formatted file size string (e.g., "2.5 MB")
 */
export function formatFileSize(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

/**
 * Format a duration in seconds to a human-readable time string
 * @param seconds - Duration in seconds
 * @param includeMilliseconds - Whether to include milliseconds in the output
 * @returns Formatted time string (e.g., "01:23:45")
 */
export function formatDuration(seconds: number, includeMilliseconds: boolean = false): string {
  if (seconds < 0) seconds = 0;

  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 1000);

  const hDisplay = h > 0 ? `${h.toString().padStart(2, '0')}:` : '';
  const mDisplay = `${m.toString().padStart(2, '0')}:`;
  const sDisplay = s.toString().padStart(2, '0');
  const msDisplay = includeMilliseconds ? `.${ms.toString().padStart(3, '0')}` : '';

  return `${hDisplay}${mDisplay}${sDisplay}${msDisplay}`;
}

/**
 * Convert a YouTube URL to an embed URL
 * @param url - The YouTube URL to convert
 * @returns YouTube embed URL
 */
export function getYouTubeEmbedUrl(url: string): string | null {
  if (!url) return null;
  
  // Extract video ID from various YouTube URL formats
  const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
  const match = url.match(regExp);
  
  return match && match[2].length === 11
    ? `https://www.youtube.com/embed/${match[2]}`
    : null;
}

/**
 * Format a percentage value
 * @param value - The percentage value (0-100)
 * @param decimals - The number of decimal places
 * @returns Formatted percentage string
 */
export function formatPercentage(value: number, decimals: number = 0): string {
  return `${value.toFixed(decimals)}%`;
}

/**
 * Truncate a string to a specified length
 * @param str - The string to truncate
 * @param maxLength - The maximum length
 * @param suffix - The suffix to add to truncated strings
 * @returns Truncated string
 */
export function truncateString(str: string, maxLength: number = 50, suffix: string = '...'): string {
  if (!str) return '';
  if (str.length <= maxLength) return str;
  
  return `${str.substring(0, maxLength)}${suffix}`;
}

export function formatRelativeTime(date: string): string {
  const now = new Date();
  const then = new Date(date);
  const seconds = Math.floor((now.getTime() - then.getTime()) / 1000);

  if (seconds < 60) {
    return 'just now';
  }

  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }

  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h ago`;
  }

  const days = Math.floor(hours / 24);
  if (days < 7) {
    return `${days}d ago`;
  }

  const weeks = Math.floor(days / 7);
  if (weeks < 4) {
    return `${weeks}w ago`;
  }

  const months = Math.floor(days / 30);
  if (months < 12) {
    return `${months}mo ago`;
  }

  const years = Math.floor(days / 365);
  return `${years}y ago`;
} 