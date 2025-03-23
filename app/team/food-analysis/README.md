# AI Food Analysis Feature

This module provides AI-powered food recognition and nutritional analysis for the fitness platform. Users can analyze their meals using their camera, uploaded images, or from their gallery of previous food photos.

## Features

- **Real-time Camera Analysis**: Capture food photos directly with device camera
- **Image Upload**: Upload existing food images for analysis
- **Gallery Access**: Access and reanalyze previous food images
- **Detailed Nutritional Analysis**: Get comprehensive breakdown of calories, macronutrients, and food detection
- **Portion Adjustment**: Adjust portion sizes and see calorie/macro changes in real-time
- **Meal Logging**: Save analyzed foods to meal log
- **Token System**: Efficient token usage for AI analysis

## Components

### Main Page Structure
- `app/team/food-analysis/page.tsx`: Main page with tab navigation between camera, upload, and gallery modes

### UI Components
- `food-camera-analysis.tsx`: Camera interface with live preview and capture
- `food-upload-analysis.tsx`: Upload component for existing images
- `food-gallery-analysis.tsx`: Gallery view of saved/recent images
- `food-results-display.tsx`: Results display with nutritional breakdown

## Technology

- Next.js with React (client components)
- Radix UI components (Tabs, Progress, Slider)
- Tailwind CSS for styling
- MediaStream API for camera access
- Canvas API for image capture

## Usage

Navigate to `/team/food-analysis` in the application to access the feature. The interface provides three methods to analyze food:

1. **Camera**: Use device camera to capture and analyze food in real-time
2. **Upload**: Upload existing food images from your device
3. **Gallery**: Browse and reanalyze previously analyzed food images

After analysis, users can:
- View detailed nutritional breakdown
- Adjust portion sizes
- Save to meal log
- View AI suggestions for better nutrition

## Token System

The AI analysis operates on a token-based system to manage API usage:
- Each analysis uses approximately 15 tokens
- Users can see their token usage and remaining tokens
- The basic plan includes 100 tokens per month

## Future Enhancements

- Integration with meal planning features
- Restaurant menu scanning
- Barcode scanning for packaged foods
- Meal comparison
- Recipe suggestions based on analyzed ingredients 