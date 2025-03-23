"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import { CreditCard, Lock } from "lucide-react";

interface PaymentFormProps {
  onSubmit: (data: PaymentFormData) => void;
  isSubmitting?: boolean;
}

export interface PaymentFormData {
  cardNumber: string;
  expiry: string;
  cvc: string;
  cardHolder: string;
}

export function PaymentForm({ onSubmit, isSubmitting }: PaymentFormProps) {
  const [cardNumber, setCardNumber] = useState("");
  const [expiry, setExpiry] = useState("");
  const [cvc, setCvc] = useState("");
  const [cardHolder, setCardHolder] = useState("");
  const [errors, setErrors] = useState<Partial<PaymentFormData>>({});
  const [focused, setFocused] = useState<keyof PaymentFormData | null>(null);

  const expiryRef = useRef<HTMLInputElement>(null);
  const cvcRef = useRef<HTMLInputElement>(null);

  // Format card number with spaces
  const formatCardNumber = (value: string) => {
    const digits = value.replace(/\D/g, "");
    const groups = digits.match(/.{1,4}/g) || [];
    return groups.join(" ").substr(0, 19); // Allow 16-19 digits
  };

  // Format expiry date with slash
  const formatExpiry = (value: string) => {
    const digits = value.replace(/\D/g, "");
    if (digits.length >= 2) {
      return `${digits.slice(0, 2)}/${digits.slice(2, 4)}`;
    }
    return digits;
  };

  // Validate expiry date
  const validateExpiry = (value: string) => {
    const [month, year] = value.split("/");
    const currentYear = new Date().getFullYear() % 100;
    const currentMonth = new Date().getMonth() + 1;

    if (!month || !year) return false;
    const numMonth = parseInt(month);
    const numYear = parseInt(year);

    if (numMonth < 1 || numMonth > 12) return false;
    if (numYear < currentYear) return false;
    if (numYear === currentYear && numMonth < currentMonth) return false;

    return true;
  };

  // Handle card number input
  const handleCardNumberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const formatted = formatCardNumber(e.target.value);
    setCardNumber(formatted);

    // Auto-advance to expiry field
    if (formatted.length >= 19) {
      expiryRef.current?.focus();
    }

    setErrors(prev => ({ ...prev, cardNumber: undefined }));
  };

  // Handle expiry input
  const handleExpiryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const formatted = formatExpiry(e.target.value);
    setExpiry(formatted);

    // Auto-advance to CVC field
    if (formatted.length === 5) {
      cvcRef.current?.focus();
    }

    setErrors(prev => ({ ...prev, expiry: undefined }));
  };

  // Handle CVC input
  const handleCvcChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value.replace(/\D/g, "").substr(0, 4);
    setCvc(value);
    setErrors(prev => ({ ...prev, cvc: undefined }));
  };

  // Handle form submission
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const newErrors: Partial<PaymentFormData> = {};

    // Validate card number
    if (cardNumber.replace(/\s/g, "").length < 16) {
      newErrors.cardNumber = "Invalid card number";
    }

    // Validate expiry
    if (!validateExpiry(expiry)) {
      newErrors.expiry = "Invalid expiry date";
    }

    // Validate CVC
    if (cvc.length < 3) {
      newErrors.cvc = "Invalid CVC";
    }

    // Validate card holder
    if (!cardHolder.trim()) {
      newErrors.cardHolder = "Required";
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    onSubmit({ cardNumber, expiry, cvc, cardHolder });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Card Number */}
      <div className="space-y-2">
        <Label htmlFor="cardNumber">Card Number</Label>
        <div className="relative">
          <Input
            id="cardNumber"
            type="text"
            value={cardNumber}
            onChange={handleCardNumberChange}
            onFocus={() => setFocused("cardNumber")}
            onBlur={() => setFocused(null)}
            placeholder="4242 4242 4242 4242"
            className={cn(
              "pl-10",
              errors.cardNumber && "border-red-500 focus-visible:ring-red-500"
            )}
            autoComplete="cc-number"
          />
          <CreditCard className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <AnimatePresence>
            {focused === "cardNumber" && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="absolute right-3 top-1/2 -translate-y-1/2"
              >
                <Lock className="h-4 w-4 text-muted-foreground" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        {errors.cardNumber && (
          <p className="text-sm text-red-500">{errors.cardNumber}</p>
        )}
      </div>

      {/* Card Holder */}
      <div className="space-y-2">
        <Label htmlFor="cardHolder">Card Holder</Label>
        <Input
          id="cardHolder"
          type="text"
          value={cardHolder}
          onChange={(e) => setCardHolder(e.target.value)}
          onFocus={() => setFocused("cardHolder")}
          onBlur={() => setFocused(null)}
          placeholder="John Doe"
          className={cn(
            errors.cardHolder && "border-red-500 focus-visible:ring-red-500"
          )}
          autoComplete="cc-name"
        />
        {errors.cardHolder && (
          <p className="text-sm text-red-500">{errors.cardHolder}</p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Expiry Date */}
        <div className="space-y-2">
          <Label htmlFor="expiry">Expiry Date</Label>
          <Input
            id="expiry"
            ref={expiryRef}
            type="text"
            value={expiry}
            onChange={handleExpiryChange}
            onFocus={() => setFocused("expiry")}
            onBlur={() => setFocused(null)}
            placeholder="MM/YY"
            className={cn(
              errors.expiry && "border-red-500 focus-visible:ring-red-500"
            )}
            autoComplete="cc-exp"
          />
          {errors.expiry && (
            <p className="text-sm text-red-500">{errors.expiry}</p>
          )}
        </div>

        {/* CVC */}
        <div className="space-y-2">
          <Label htmlFor="cvc">CVC</Label>
          <Input
            id="cvc"
            ref={cvcRef}
            type="text"
            value={cvc}
            onChange={handleCvcChange}
            onFocus={() => setFocused("cvc")}
            onBlur={() => setFocused(null)}
            placeholder="123"
            className={cn(
              errors.cvc && "border-red-500 focus-visible:ring-red-500"
            )}
            autoComplete="cc-csc"
          />
          {errors.cvc && <p className="text-sm text-red-500">{errors.cvc}</p>}
        </div>
      </div>

      <Button
        type="submit"
        className="w-full"
        disabled={isSubmitting}
      >
        {isSubmitting ? (
          <>
            <span className="loader-sm mr-2"></span>
            Processing...
          </>
        ) : (
          "Pay Now"
        )}
      </Button>
    </form>
  );
}