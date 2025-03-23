# Deployment Guide: AI Video Platform on bolt.new

This guide provides step-by-step instructions for deploying the AI video editing platform to bolt.new with the `/platform` route.

## 1. Bolt.new Integration Guide

### Initial Setup

1. Create a new project in bolt.new:
   ```bash
   # Install bolt CLI
   npm install -g @bolt/cli

   # Login to bolt.new
   bolt login

   # Create new project
   bolt init ai-video-platform
   ```

2. Connect your repository:
   - Option 1: GitHub Integration
     ```bash
     bolt link --repo github.com/yourusername/ai-video-platform
     ```
   - Option 2: Manual Upload
     ```bash
     bolt deploy --dir ./
     ```

3. Configure project settings:
   ```bash
   # Set project type
   bolt config set type next

   # Set Node.js version
   bolt config set nodeVersion 18.x

   # Configure build settings
   bolt config set buildCommand "npm run build"
   bolt config set outputDir ".next"
   ```

### Environment Setup

1. Set required environment variables:
   ```bash
   # Production environment variables
   bolt env set NEXT_PUBLIC_BASE_URL "https://your-domain.bolt.new"
   bolt env set NEXT_PUBLIC_STRIPE_KEY "pk_live_xxx"
   bolt env set STRIPE_SECRET_KEY "sk_live_xxx"
   bolt env set API_BASE_URL "https://api.your-domain.com"

   # Optional environment variables
   bolt env set ENABLE_ANALYTICS "true"
   bolt env set LOG_LEVEL "error"
   ```

2. Configure secrets:
   ```bash
   # Add sensitive configuration
   bolt secrets add STRIPE_WEBHOOK_SECRET "whsec_xxx"
   bolt secrets add JWT_SECRET "your-jwt-secret"
   ```

## 2. Deployment Configuration

### Build Settings

1. Verify build configuration in `bolt.config.js`:
   ```javascript
   module.exports = {
     project: {
       buildCommand: 'npm run build',
       outputDir: '.next',
       nodeVersion: '18.x',
     }
   }
   ```

2. Configure route handling:
   ```javascript
   module.exports = {
     routes: {
       '/': {
         redirect: '/platform',
         permanent: true,
       },
       '/platform/*': {
         source: '/platform/:path*',
         destination: '/:path*',
       }
     }
   }
   ```

3. Create custom error pages:
   - Create `pages/404.tsx` for custom 404 page
   - Create `pages/500.tsx` for server error page
   - Create `pages/_error.tsx` for general error handling

## 3. Integration with Netlify

### Netlify Deployment Settings

1. Configure build hooks:
   ```bash
   # Create build hook
   bolt hooks create production "https://api.netlify.com/build_hooks/xxx"

   # Test build hook
   curl -X POST -d {} https://api.netlify.com/build_hooks/xxx
   ```

2. Set up custom domain:
   ```bash
   # Add custom domain
   bolt domain add your-domain.com

   # Configure DNS
   bolt domain dns your-domain.com
   ```

3. Configure SSL:
   ```bash
   # Enable automatic SSL
   bolt ssl auto your-domain.com
   ```

### Performance Optimization

1. Configure cache headers:
   ```javascript
   module.exports = {
     cache: {
       patterns: [
         {
           match: '/static/**/*',
           headers: {
             'Cache-Control': 'public, max-age=31536000, immutable'
           }
         }
       ]
     }
   }
   ```

2. Enable asset optimization:
   ```javascript
   module.exports = {
     optimization: {
       minify: true,
       compress: true,
       splitChunks: true,
       treeshake: true
     }
   }
   ```

## 4. Post-Deployment

### Verification Steps

1. Test all routes:
   ```bash
   # Run E2E tests
   npm run cypress:run

   # Test critical paths manually
   - Visit /platform
   - Test authentication flow
   - Verify API connections
   - Test file uploads
   ```

2. Verify API connections:
   ```bash
   # Check API health
   curl https://your-domain.bolt.new/api/health

   # Test token endpoints
   curl https://your-domain.bolt.new/api/tokens/balance
   ```

### Monitoring Setup

1. Configure error tracking:
   ```javascript
   module.exports = {
     monitoring: {
       errorTracking: true,
       performanceMonitoring: true
     }
   }
   ```

2. Set up analytics:
   ```bash
   # Enable analytics
   bolt monitoring enable

   # Configure custom events
   bolt monitoring events add "token_purchase"
   bolt monitoring events add "video_upload"
   ```

## 5. CI/CD Pipeline

### Automated Testing

1. Configure test runs:
   ```yaml
   # .github/workflows/test.yml
   name: Tests
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - run: npm ci
         - run: npm test
         - run: npm run cypress:run
   ```

### Preview Deployments

1. Enable preview deployments:
   ```javascript
   module.exports = {
     ci: {
       prPreview: true,
       autoDeployBranch: 'main'
     }
   }
   ```

2. Configure preview environment:
   ```bash
   # Set preview variables
   bolt env set -e preview NEXT_PUBLIC_BASE_URL "https://preview.your-domain.bolt.new"
   ```

### Production Deployment

1. Set up deployment workflow:
   ```bash
   # Deploy to production
   bolt deploy production

   # Monitor deployment
   bolt deployment status
   ```

2. Configure rollback procedure:
   ```bash
   # List deployments
   bolt deployments list

   # Rollback if needed
   bolt rollback [deployment-id]
   ```

## Feature Verification and Integration

### Pre-deployment Verification

1. Run the feature verification script:
   ```bash
   # Set up environment variables
   export AUTH_TOKEN="your_test_token"
   export BASE_URL="http://localhost:3000"

   # Run verification
   node scripts/verify-features.js
   ```

2. Verify WebSocket connections:
   ```bash
   # Test WebSocket endpoint
   wscat -c "ws://localhost:3000/ws" -H "Authorization: Bearer $AUTH_TOKEN"
   ```

3. Check feature interactions:
   ```bash
   # Run integration tests
   npm run test:integration

   # Run E2E tests
   npm run cypress:run
   ```

### Post-deployment Verification

1. Verify core functionality:
   ```bash
   # Set production URL
   export BASE_URL="https://your-domain.bolt.new"

   # Run verification against production
   NODE_ENV=production node scripts/verify-features.js
   ```

2. Monitor system health:
   ```bash
   # Check system metrics
   bolt monitoring status

   # View error logs
   bolt logs --type error
   ```

3. Performance validation:
   ```bash
   # Run Lighthouse audit
   npx lighthouse $BASE_URL/platform --output=json

   # Check API response times
   bolt monitoring latency
   ```

### Integration Points Checklist

- [ ] Authentication flow works with token management
- [ ] WebSocket connections maintain stability under load
- [ ] Video processing pipeline handles concurrent jobs
- [ ] Translation services interact correctly with avatar generation
- [ ] Clip generator respects token limits
- [ ] Real-time updates propagate through WebSocket
- [ ] User settings persist across sessions
- [ ] Payment processing integrates with token system

### Troubleshooting Integration Issues

1. WebSocket Connection Issues:
   - Check WebSocket server status
   - Verify authentication headers
   - Monitor connection stability

2. Token Management Issues:
   - Validate token balance updates
   - Check transaction atomicity
   - Monitor concurrent token operations

3. Feature Interaction Issues:
   - Review job queue processing
   - Check feature dependencies
   - Monitor system resources

4. Performance Issues:
   - Analyze response times
   - Check resource utilization
   - Monitor memory usage

## Troubleshooting

### Common Issues

1. Build failures:
   - Check Node.js version compatibility
   - Verify all dependencies are installed
   - Check for environment variable issues

2. Route issues:
   - Verify route configuration in `bolt.config.js`
   - Check redirect rules
   - Verify API proxy settings

3. Performance issues:
   - Review cache configuration
   - Check asset optimization settings
   - Monitor API response times

### Support Resources

- Bolt.new Documentation: https://docs.bolt.new
- Community Forums: https://community.bolt.new
- Support Email: support@bolt.new

## Security Considerations

1. Enable security headers:
   ```javascript
   module.exports = {
     headers: {
       '/*': {
         'X-Frame-Options': 'DENY',
         'X-XSS-Protection': '1; mode=block'
       }
     }
   }
   ```

2. Configure rate limiting:
   ```javascript
   module.exports = {
     rateLimit: {
       window: 60000,
       max: 100
     }
   }
   ```

Remember to regularly update dependencies and monitor security advisories for potential vulnerabilities. 