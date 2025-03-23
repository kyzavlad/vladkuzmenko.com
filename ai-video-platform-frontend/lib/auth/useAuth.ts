import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

export interface User {
  id: string;
  email: string;
  name: string;
}

export function useAuth() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const token = localStorage.getItem('token');
        if (!token) {
          setUser(null);
          setLoading(false);
          return;
        }

        // Здесь должен быть реальный запрос к API для проверки токена
        // Сейчас просто симулируем успешную авторизацию
        const mockUser = {
          id: '1',
          email: 'user@example.com',
          name: 'Test User'
        };
        
        setUser(mockUser);
      } catch (error) {
        console.error('Auth check failed:', error);
        localStorage.removeItem('token');
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  const logout = () => {
    localStorage.removeItem('token');
    setUser(null);
    router.push('/login');
  };

  return { user, loading, logout };
} 