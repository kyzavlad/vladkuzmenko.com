"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { PaymentDialog } from "@/components/ui/payment-dialog";
import { motion } from "framer-motion";
import { Chrome, Github, Facebook, Twitter } from "lucide-react";
import { cn } from "@/lib/utils";
import { useLocalStorage } from "usehooks-ts";

export function PlatformAccessSection() {
  const [hasAccess, setHasAccess] = useLocalStorage("platform-access", false);

  // Функция для обработки успешного платежа
  const handlePaymentSuccess = () => {
    setHasAccess(true);
  };

  return (
    <section className="w-full py-20 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="relative overflow-hidden rounded-2xl border border-white/10 shadow-lg">
          {/* Glow effect */}
          <div className="absolute inset-0 pointer-events-none">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-purple-500/10 animate-aurora"></div>
            <div className="absolute -inset-px bg-neutral-900/90 [mask-image:linear-gradient(black,transparent)]"></div>
          </div>
          
          {/* Blur Overlay (until payment is completed) */}
          {!hasAccess && (
            <div className="absolute inset-0 backdrop-blur-xl bg-black/60 z-20 flex flex-col items-center justify-center">
              <h3 className="text-2xl md:text-3xl font-bold text-white mb-6">
                Access the Warriors Platform
              </h3>
              <p className="text-lg text-gray-300 mb-6 max-w-xl text-center">
                Join our elite community of ambitious men dedicated to growth and success.
              </p>
              
              {/* Используем обычный PaymentDialog без onSuccess */}
              <div onClick={handlePaymentSuccess}>
                <PaymentDialog
                  planName="Warriors Platform Access"
                  planPrice="199"
                  planPeriod="per month"
                  buttonText="Get Access - $199/month"
                  buttonVariant="default"
                  className="inline-block"
                />
              </div>
            </div>
          )}
          
          {/* Platform Content */}
          <div className={cn("relative p-8 md:p-12 transition-all", !hasAccess && "filter blur-md pointer-events-none")}>
            <div className="text-center mb-8">
              <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
                Welcome to the Warriors Platform
              </h2>
              <p className="text-xl text-gray-300">
                Connect with your preferred account to get started
              </p>
            </div>
            <div className="max-w-md mx-auto space-y-4">
              <Button className="w-full bg-[#DB4437] text-white hover:bg-[#DB4437]/90">
                <Chrome className="mr-2 h-5 w-5" />
                Continue with Google
              </Button>
              <Button className="w-full bg-[#1877f2] text-white hover:bg-[#1877f2]/90">
                <Facebook className="mr-2 h-5 w-5" />
                Continue with Facebook
              </Button>
              <Button className="w-full bg-[#14171a] text-white hover:bg-[#14171a]/90">
                <Twitter className="mr-2 h-5 w-5" />
                Continue with X
              </Button>
              <Button className="w-full bg-[#333333] text-white hover:bg-[#333333]/90">
                <Github className="mr-2 h-5 w-5" />
                Continue with GitHub
              </Button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}