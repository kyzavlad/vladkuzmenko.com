# AI Video Platform Frontend

This is the frontend application for the AI Video Platform, featuring an advanced video upload and configuration interface for generating optimized video clips.

## Features

- **Video Upload Interface**
  - Batch video selection and upload
  - Platform-specific optimization presets
  - Content category selection
  - Generate clips option

- **Configuration Panel**
  - Duration range selection (5-120 seconds)
  - Face tracking toggle
  - Silence removal toggle
  - Moment detection toggle
  - Target platform optimization
  - Clip count limitation
  - Output quality selection

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn

## Installation

1. Clone the repository
2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
3. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

## Development

To start the development server:

```bash
npm start
# or
yarn start
```

The application will be available at `http://localhost:3000`.

## Building for Production

To create a production build:

```bash
npm run build
# or
yarn build
```

The build artifacts will be stored in the `build/` directory.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── VideoUpload.tsx
│   │   └── ConfigurationPanel.tsx
│   ├── utils/
│   │   └── VideoPresets.ts
│   ├── App.tsx
│   └── index.tsx
├── public/
└── package.json
```

## Dependencies

- React 18
- Chakra UI
- React Dropzone
- TypeScript
- React Icons

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 