{`"use client";

import * as React from 'react';
import { GripVertical } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ResizableProps extends React.HTMLAttributes<HTMLDivElement> {
  onResize?: (width: number) => void;
  minWidth?: number;
  maxWidth?: number;
  defaultWidth?: number;
}

const Resizable = React.forwardRef<HTMLDivElement, ResizableProps>(
  ({ className, children, onResize, minWidth = 200, maxWidth = 800, defaultWidth = 300, ...props }, ref) => {
    const [width, setWidth] = React.useState(defaultWidth);
    const [isResizing, setIsResizing] = React.useState(false);
    const resizeRef = React.useRef<HTMLDivElement>(null);

    React.useEffect(() => {
      const handleMouseMove = (e: MouseEvent) => {
        if (!isResizing) return;
        
        const newWidth = Math.min(Math.max(e.clientX, minWidth), maxWidth);
        setWidth(newWidth);
        onResize?.(newWidth);
      };

      const handleMouseUp = () => {
        setIsResizing(false);
      };

      if (isResizing) {
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
      }

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }, [isResizing, minWidth, maxWidth, onResize]);

    return (
      <div
        ref={ref}
        className={cn('relative flex', className)}
        style={{ width: \`\${width}px\` }}
        {...props}
      >
        {children}
        <div
          ref={resizeRef}
          className="absolute right-0 top-0 h-full w-2 cursor-col-resize"
          onMouseDown={() => setIsResizing(true)}
        >
          <GripVertical className="h-4 w-4 absolute top-1/2 right-0 -translate-y-1/2 text-muted-foreground/60" />
        </div>
      </div>
    );
  }
);

Resizable.displayName = 'Resizable';

export { Resizable };`}