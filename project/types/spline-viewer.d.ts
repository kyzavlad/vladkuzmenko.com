declare namespace JSX {
  interface IntrinsicElements {
    'spline-viewer': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
      url: string;
      loading?: 'lazy' | 'eager';
      'events-target'?: string;
    };
  }
}