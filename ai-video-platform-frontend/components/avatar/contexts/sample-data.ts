import { Background, CameraAngle, LightingSetup, Prop } from './generation-context';

export const SAMPLE_BACKGROUNDS: Background[] = [
  {
    id: 'office',
    name: 'Modern Office',
    thumbnail: '/backgrounds/office.jpg',
    description: 'A clean, modern office space with natural lighting',
    type: 'environment',
    category: 'indoor',
    videoUrl: '/backgrounds/office.mp4'
  },
  {
    id: 'studio',
    name: 'Professional Studio',
    thumbnail: '/backgrounds/studio.jpg',
    description: 'A professional video production studio',
    type: 'environment',
    category: 'indoor',
    videoUrl: '/backgrounds/studio.mp4'
  },
  {
    id: 'outdoor',
    name: 'Outdoor Scene',
    thumbnail: '/backgrounds/outdoor.jpg',
    description: 'A natural outdoor environment',
    type: 'environment',
    category: 'outdoor',
    videoUrl: '/backgrounds/outdoor.mp4'
  }
];

export const SAMPLE_CAMERA_ANGLES: CameraAngle[] = [
  {
    id: 'front',
    name: 'Front View',
    thumbnail: '/camera-angles/front.jpg',
    description: 'Direct front view of the avatar',
    zoom: 100,
    position: { x: 0, y: 0 }
  },
  {
    id: 'three-quarter',
    name: 'Three-Quarter View',
    thumbnail: '/camera-angles/three-quarter.jpg',
    description: 'Slightly angled view for more depth',
    zoom: 100,
    position: { x: 0, y: 0 }
  },
  {
    id: 'profile',
    name: 'Profile View',
    thumbnail: '/camera-angles/profile.jpg',
    description: 'Side view of the avatar',
    zoom: 100,
    position: { x: 0, y: 0 }
  }
];

export const SAMPLE_LIGHTING_SETUPS: LightingSetup[] = [
  {
    id: 'natural',
    name: 'Natural Lighting',
    thumbnail: '/lighting/natural.jpg',
    description: 'Soft, natural lighting setup',
    brightness: 80,
    contrast: 60,
    temperature: 5500,
    direction: 'front'
  },
  {
    id: 'studio',
    name: 'Studio Lighting',
    thumbnail: '/lighting/studio.jpg',
    description: 'Professional studio lighting setup',
    brightness: 90,
    contrast: 70,
    temperature: 6500,
    direction: 'front'
  },
  {
    id: 'dramatic',
    name: 'Dramatic Lighting',
    thumbnail: '/lighting/dramatic.jpg',
    description: 'Dramatic lighting for emphasis',
    brightness: 70,
    contrast: 80,
    temperature: 4500,
    direction: 'front'
  }
];

export const SAMPLE_PROPS: Prop[] = [
  {
    id: 'desk',
    name: 'Desk',
    thumbnail: '/props/desk.jpg',
    description: 'Modern office desk',
    category: 'furniture',
    position: { x: 0, y: 0, z: 0, rotation: 0, scale: 1 }
  },
  {
    id: 'chair',
    name: 'Chair',
    thumbnail: '/props/chair.jpg',
    description: 'Office chair',
    category: 'furniture',
    position: { x: 0, y: 0, z: 0, rotation: 0, scale: 1 }
  },
  {
    id: 'screen',
    name: 'Screen',
    thumbnail: '/props/screen.jpg',
    description: 'Computer screen',
    category: 'electronics',
    position: { x: 0, y: 0, z: 0, rotation: 0, scale: 1 }
  }
]; 