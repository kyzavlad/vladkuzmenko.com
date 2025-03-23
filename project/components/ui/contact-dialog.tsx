"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { RainbowButton } from "@/components/ui/rainbow-button";
import { ArrowRight } from "lucide-react";

interface ContactDialogProps {
  triggerText: string;
  children?: React.ReactNode;
  variant?: "default" | "outline" | "secondary" | "destructive" | "ghost" | "link";
  className?: string;
  isRainbowButton?: boolean;
}

export function ContactDialog({ 
  triggerText, 
  children, 
  variant = "default", 
  className = "",
  isRainbowButton = false
}: ContactDialogProps) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate API call
    setTimeout(() => {
      setIsSubmitting(false);
      setIsSuccess(true);
      
      // Reset form after success
      setTimeout(() => {
        setIsSuccess(false);
      }, 3000);
    }, 1500);
  };
  
  return (
    <Dialog>
      <DialogTrigger asChild>
        {children ? (
          children
        ) : isRainbowButton ? (
          <RainbowButton className={className}>
            <span className="flex items-center">
              {triggerText}
              <ArrowRight className="ml-2 h-4 w-4" />
            </span>
          </RainbowButton>
        ) : (
          <Button variant={variant} className={className}>{triggerText}</Button>
        )}
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Contact Us</DialogTitle>
          <DialogDescription>
            Fill out the form below and our team will get back to you within 24 hours.
          </DialogDescription>
        </DialogHeader>
        
        {isSuccess ? (
          <div className="py-6 text-center">
            <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-green-100 text-green-500">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="h-6 w-6">
                <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
              </svg>
            </div>
            <h3 className="mb-2 text-lg font-medium">Message Sent!</h3>
            <p className="text-muted-foreground">Thank you for reaching out. We'll be in touch soon.</p>
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="name" className="text-right">
                  Name
                </Label>
                <Input id="name" className="col-span-3" required />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="email" className="text-right">
                  Email
                </Label>
                <Input id="email" type="email" className="col-span-3" required />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="company" className="text-right">
                  Company
                </Label>
                <Input id="company" className="col-span-3" />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="message" className="text-right">
                  Message
                </Label>
                <Textarea id="message" className="col-span-3" required />
              </div>
            </div>
            <DialogFooter>
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting ? (
                  <>
                    <span className="loader-sm mr-2"></span>
                    Sending...
                  </>
                ) : (
                  "Send Message"
                )}
              </Button>
            </DialogFooter>
          </form>
        )}
      </DialogContent>
    </Dialog>
  );
}