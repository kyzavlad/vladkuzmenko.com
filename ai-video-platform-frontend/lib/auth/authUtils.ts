import Cookies from 'js-cookie';
import { jwtDecode } from 'jwt-decode';

// Token-related constants
export const TOKEN_KEY = 'token';
export const TOKEN_EXPIRY_DAYS = 7;

/**
 * Interface for the decoded JWT token
 */
export interface DecodedToken {
  sub: string; // User ID
  email: string;
  name: string;
  role: string;
  exp: number; // Expiration timestamp
  iat: number; // Issued at timestamp
}

/**
 * Get the authentication token from cookies
 */
export function getToken(): string | null {
  return Cookies.get(TOKEN_KEY) || null;
}

/**
 * Set the authentication token in cookies
 */
export function setToken(token: string): void {
  Cookies.set(TOKEN_KEY, token, { expires: TOKEN_EXPIRY_DAYS });
}

/**
 * Remove the authentication token from cookies
 */
export function removeToken(): void {
  Cookies.remove(TOKEN_KEY);
}

/**
 * Check if a user is authenticated
 */
export function isAuthenticated(): boolean {
  const token = getToken();
  
  if (!token) {
    return false;
  }
  
  try {
    const decodedToken = jwtDecode<DecodedToken>(token);
    return decodedToken.exp * 1000 > Date.now();
  } catch (error) {
    removeToken();
    return false;
  }
}

/**
 * Get user information from the token
 */
export function getUserFromToken(): DecodedToken | null {
  const token = getToken();
  
  if (!token) {
    return null;
  }
  
  try {
    return jwtDecode<DecodedToken>(token);
  } catch (error) {
    return null;
  }
}

/**
 * Check if token is about to expire (within the specified time)
 * @param minutesThreshold - Minutes threshold before expiration
 */
export function isTokenExpiringSoon(minutesThreshold: number = 15): boolean {
  const token = getToken();
  
  if (!token) {
    return false;
  }
  
  try {
    const decodedToken = jwtDecode<DecodedToken>(token);
    const expirationTime = decodedToken.exp * 1000;
    const currentTime = Date.now();
    const thresholdMs = minutesThreshold * 60 * 1000;
    
    return expirationTime - currentTime < thresholdMs;
  } catch (error) {
    return false;
  }
}

/**
 * Check if the current user has the required role
 * @param requiredRole - The role required to access a resource
 */
export function hasRole(requiredRole: string): boolean {
  const user = getUserFromToken();
  
  if (!user) {
    return false;
  }
  
  if (requiredRole === 'admin' && user.role !== 'admin') {
    return false;
  }
  
  return true;
} 