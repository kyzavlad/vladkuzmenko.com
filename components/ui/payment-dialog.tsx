"use client";

import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { Lock, CreditCard } from "lucide-react";

interface PaymentDialogProps {
  planName: string;
  planPrice: string;
  planPeriod: string;
  buttonText: string;
  buttonVariant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link";
  fullWidth?: boolean;
  className?: string;
}

export function PaymentDialog({
  planName,
  planPrice,
  planPeriod,
  buttonText,
  buttonVariant = "default",
  fullWidth = false,
  className
}: PaymentDialogProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate payment processing
    setTimeout(() => {
      setIsSubmitting(false);
      // Store access token
      localStorage.setItem("platform-access", "true");
      // Refresh the page to update access state
      window.location.reload();
    }, 2000);
  };

  const TriggerComponent = fullWidth ? 'div' : Button;

  return (
    <Dialog>
      <DialogTrigger asChild>
        <TriggerComponent 
          className={cn(
            fullWidth ? "cursor-pointer" : "",
            className
          )}
          {...(!fullWidth ? { variant: buttonVariant } : {})}
        >
          {fullWidth ? (
            <Button 
              variant={buttonVariant}
              className="w-full"
            >
              {buttonText}
            </Button>
          ) : (
            buttonText
          )}
        </TriggerComponent>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Payment Details</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-6 py-4">
          <div className="rounded-lg border p-4 bg-muted/50">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium">{planName}</span>
              <span className="text-muted-foreground">{planPeriod}</span>
            </div>
            <div className="text-2xl font-bold">${planPrice}</div>
          </div>

          <div className="space-y-4">
            <div>
              <Label htmlFor="cardName">Name on Card</Label>
              <Input id="cardName" placeholder="John Doe" required />
            </div>
            <div>
              <Label htmlFor="cardNumber">Card Number</Label>
              <Input id="cardNumber" placeholder="4242 4242 4242 4242" required />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="expiry">Expiry Date</Label>
                <Input id="expiry" placeholder="MM/YY" required />
              </div>
              <div>
                <Label htmlFor="cvc">CVC</Label>
                <Input id="cvc" placeholder="123" required />
              </div>
            </div>
          </div>

          <Button 
            type="submit" 
            className="w-full"
            disabled={isSubmitting}
          >
            <CreditCard className="w-4 h-4 mr-2" />
            {isSubmitting ? "Processing..." : `Pay $${planPrice}`}
          </Button>

          <div className="flex items-center gap-2 text-sm text-muted-foreground justify-center">
            <Lock className="w-4 h-4" />
            <span>Payments are secure and encrypted</span>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}