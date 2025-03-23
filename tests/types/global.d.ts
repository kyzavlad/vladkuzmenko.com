declare global {
  var __CLEANUP__: () => Promise<void>;
  
  namespace NodeJS {
    interface ProcessEnv {
      TEST_AUTH_TOKEN: string;
      TEST_USER_ID: string;
      CI?: string;
      BASE_URL?: string;
      NODE_ENV: 'development' | 'production' | 'test';
    }
  }
}

export {}; 