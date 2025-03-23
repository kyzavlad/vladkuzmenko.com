"use client";

import { Button } from "@/components/ui/button";
import { CheckCircle2, CreditCard } from "lucide-react";
import { motion } from "framer-motion";
import { GlowEffect } from "@/components/ui/glow-effect";

interface PaymentConfirmationProps {
  planName: string;
  planPrice: string;
  onClose: () => void;
  cardData?: {
    cardNumber: string;
    cardHolder: string;
    expiry: string;
  };
}

export function PaymentConfirmation({
  planName,
  planPrice,
  onClose,
  cardData
}: PaymentConfirmationProps) {
  const formatCardNumber = (number: string) => {
    return `•••• •••• •••• ${number.slice(-4)}`;
  };

  return (
    <div className="flex flex-col items-center justify-center py-6">
      <motion.div
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{
          type: "spring",
          stiffness: 260,
          damping: 20
        }}
        className="mb-8"
      >
        {/* Credit Card Design */}
        <div className="relative w-[340px] h-[200px] rounded-2xl overflow-hidden perspective-1000">
          <GlowEffect
            colors={["#0894FF", "#C959DD", "#FF2E54", "#FF9004"]}
            mode="static"
            blur="medium"
          />
          <div className="relative h-full w-full rounded-2xl bg-black/80 backdrop-blur-xl p-6 text-white">
            {/* Chip and Wireless Icons */}
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-9 bg-yellow-400/90 rounded-md" />
              <svg 
                viewBox="0 0 24 24" 
                className="w-6 h-6 rotate-90 text-white/80"
                fill="none"
                stroke="currentColor"
              >
                <path 
                  d="M8.288 12c0-2.045.344-4.297 1.03-6.177M12 12c0-3.037.566-6.04 1.677-8.5M15.712 12c0-2.045-.344-4.297-1.03-6.177" 
                  strokeWidth="2" 
                  strokeLinecap="round"
                />
              </svg>
            </div>

            {/* Card Number */}
            <div className="font-mono text-xl tracking-wider mb-4">
              {cardData ? formatCardNumber(cardData.cardNumber) : "•••• •••• •••• ••••"}
            </div>

            {/* Card Holder and Expiry */}
            <div className="flex justify-between items-end mt-auto">
              <div>
                <div className="text-xs text-white/60 mb-1">Card Holder</div>
                <div className="font-medium tracking-wide">
                  {cardData?.cardHolder || "CARD HOLDER"}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-white/60 mb-1">Expires</div>
                <div className="font-medium">
                  {cardData?.expiry || "MM/YY"}
                </div>
              </div>
            </div>

            {/* Card Brand Logo */}
            <div className="absolute top-6 right-6">
              <CreditCard className="w-8 h-8 text-white/90" />
            </div>
          </div>
        </div>
      </motion.div>
      
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="text-center mb-8"
      >
        <div className="flex items-center justify-center mb-4">
          <div className="rounded-full bg-emerald-100 p-2 dark:bg-emerald-900/30">
            <CheckCircle2 className="h-6 w-6 text-emerald-600 dark:text-emerald-400" />
          </div>
        </div>
        <h3 className="text-xl font-semibold mb-2">Payment Successful!</h3>
        <p className="text-muted-foreground">
          Thank you for choosing our {planName} plan. Your subscription is now active.
        </p>
        <div className="mt-4 text-sm text-muted-foreground">
          <p>We're excited to embark on this journey together.</p>
          <p>Let's create something amazing!</p>
        </div>
      </motion.div>
      
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="w-full"
      >
        <Button 
          onClick={onClose}
          className="w-full"
          variant="outline"
        >
          Close
        </Button>
      </motion.div>
    </div>
  );
}