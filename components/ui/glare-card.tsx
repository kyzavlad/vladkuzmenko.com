import React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface GlareCardProps {
  children: React.ReactNode;
  className?: string;
  [key: string]: any; // Для поддержки дополнительных props
}

// Создаем компонент без использования forwardRef для обхода ошибки типизации
export const GlareCard: React.FC<GlareCardProps> = ({ 
  children, 
  className,
  ...props 
}) => {
  return (
    <motion.div
      className={cn(
        "relative overflow-hidden rounded-xl border border-white/10 bg-black p-8",
        className
      )}
      {...props}
    >
      {/* Glare effect */}
      <motion.div
        className="pointer-events-none absolute -inset-px opacity-0 transition duration-300"
        style={{
          background:
            "radial-gradient(600px circle at var(--mouse-x) var(--mouse-y), rgba(255,255,255,0.06), transparent 40%)",
        }}
      />
      
      {children}
    </motion.div>
  );
};

GlareCard.displayName = "GlareCard";