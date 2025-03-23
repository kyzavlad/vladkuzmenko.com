"use client";

import * as React from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Instagram, Moon, Send, Sun, Twitter, Youtube, MessageCircle, MapPin } from "lucide-react"

export function Footerdemo() {
  const [isDarkMode, setIsDarkMode] = React.useState(true)
  const [isChatOpen, setIsChatOpen] = React.useState(false)

  React.useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark")
    } else {
      document.documentElement.classList.remove("dark")
    }
  }, [isDarkMode])

  const scrollToSection = (sectionId: string, e: React.MouseEvent) => {
    e.preventDefault();
    const section = document.getElementById(sectionId);
    if (section) {
      section.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <footer className="w-full py-8 md:py-12 border-t">
      <div className="container px-4 md:px-6">
        <div className="grid gap-12 md:grid-cols-2 lg:grid-cols-4">
          <div className="relative">
            <h2 className="text-3xl font-bold tracking-tighter text-foreground">Stay Connected</h2>
            <p className="mt-4 text-base md:text-lg text-muted-foreground mb-6">
              Join our newsletter for the latest updates and exclusive offers.
            </p>
            <form className="relative">
              <Input
                type="email"
                placeholder="Enter your email"
                className="pr-12 backdrop-blur-sm"
              />
              <Button
                type="submit"
                size="icon"
                className="absolute right-1 top-1 h-8 w-8 rounded-full bg-primary text-primary-foreground transition-transform hover:scale-105"
              >
                <Send className="h-4 w-4" />
                <span className="sr-only">Subscribe</span>
              </Button>
            </form>
            <div className="absolute -right-4 top-0 h-24 w-24 rounded-full bg-primary/10 blur-2xl" />
          </div>
          <div>
            <h3 className="mb-4 text-lg font-semibold">Quick Links</h3>
            <nav className="space-y-2 text-sm">
              <a 
                href="#hero-section" 
                className="block transition-colors hover:text-primary"
                onClick={(e) => scrollToSection('hero-section', e)}
              >
                Home
              </a>
              <a 
                href="#features-section" 
                className="block transition-colors hover:text-primary"
                onClick={(e) => scrollToSection('features-section', e)}
              >
                Features
              </a>
              <a 
                href="#board-section" 
                className="block transition-colors hover:text-primary"
                onClick={(e) => scrollToSection('board-section', e)}
              >
                Services
              </a>
              <a 
                href="#pricing-section" 
                className="block transition-colors hover:text-primary"
                onClick={(e) => scrollToSection('pricing-section', e)}
              >
                Pricing
              </a>
              <a 
                href="#blog-section" 
                className="block transition-colors hover:text-primary"
                onClick={(e) => scrollToSection('blog-section', e)}
              >
                Blog
              </a>
            </nav>
          </div>
          <div>
            <h3 className="mb-4 text-lg font-extrabold">Platform Sections</h3>
            <nav className="space-y-2 text-sm">
              <a 
                href="#audio-section" 
                className="block transition-colors hover:text-primary"
                onClick={(e) => scrollToSection('audio-section', e)}
              >
                Voice Assistant
              </a>
              <a 
                href="#testimonials-section" 
                className="block transition-colors hover:text-primary"
                onClick={(e) => scrollToSection('testimonials-section', e)}
              >
                Testimonials
              </a>
              <a 
                href="#map-section" 
                className="block transition-colors hover:text-primary"
                onClick={(e) => scrollToSection('map-section', e)}
              >
                Global Network
              </a>
              <a 
                href="#saas-launch-section" 
                className="block transition-colors hover:text-primary"
                onClick={(e) => scrollToSection('saas-launch-section', e)}
              >
                Coming Soon
              </a>
              <a 
                href="#cta-section" 
                className="block transition-colors hover:text-primary"
                onClick={(e) => scrollToSection('cta-section', e)}
              >
                Get Started
              </a>
            </nav>
          </div>
          <div className="relative">
            <h3 className="mb-4 text-lg font-extrabold">Contact us</h3>
            <div className="flex items-center gap-2 mb-4 text-white">
              <MapPin className="h-4 w-4" />
              <p className="text-sm">400 5th Ave, New York, NY 10018, United States</p>
            </div>
            <p className="text-sm text-white mb-4">
              Email: <a href="mailto:ai@vladkuzmenko.com" className="hover:text-primary transition-colors">ai@vladkuzmenko.com</a>
            </p>
            <div className="mb-6 flex space-x-4">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" className="rounded-full" asChild>
                      <a href="https://www.instagram.com/vladkuzmenkosxy/" target="_blank" rel="noopener noreferrer">
                        <Instagram className="h-4 w-4" />
                        <span className="sr-only">Instagram</span>
                      </a>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Follow us on Instagram</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" className="rounded-full" asChild>
                      <a href="https://www.youtube.com/@vladkuzmenkoai" target="_blank" rel="noopener noreferrer">
                        <Youtube className="h-4 w-4" />
                        <span className="sr-only">YouTube</span>
                      </a>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Subscribe on YouTube</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" className="rounded-full" asChild>
                      <a href="http://x.com/vladkuzmenkosxy" target="_blank" rel="noopener noreferrer">
                        <Twitter className="h-4 w-4" />
                        <span className="sr-only">Twitter/X</span>
                      </a>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Follow us on Twitter/X</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" className="rounded-full" asChild>
                      <a href="https://api.whatsapp.com/send/?phone=380951444853&text&type=phone_number&app_absent=0" target="_blank" rel="noopener noreferrer">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
                          <path d="M3 21l1.65-3.8a9 9 0 1 1 3.4 2.9L3 21" />
                          <path d="M9 10a.5.5 0 0 0 1 0V9a.5.5 0 0 0-1 0v1Z" />
                          <path d="M14 10a.5.5 0 0 0 1 0V9a.5.5 0 0 0-1 0v1Z" />
                          <path d="M9.5 13.5c.5 1 1.5 1 2.5 1s2-.5 2.5-1" />
                        </svg>
                        <span className="sr-only">WhatsApp</span>
                      </a>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Contact us on WhatsApp</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" className="rounded-full" asChild>
                      <a href="https://t.me/vladkuzmenkoai" target="_blank" rel="noopener noreferrer">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
                          <path d="m22 2-7 20-4-9-9-4Z" />
                          <path d="M22 2 11 13" />
                        </svg>
                        <span className="sr-only">Telegram</span>
                      </a>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Contact us on Telegram</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <div className="flex items-center space-x-2">
              <Sun className="h-4 w-4" />
              <Switch
                id="dark-mode"
                checked={isDarkMode}
                onCheckedChange={setIsDarkMode}
              />
              <Moon className="h-4 w-4" />
              <Label htmlFor="dark-mode" className="sr-only">
                Toggle dark mode
              </Label>
            </div>
          </div>
        </div>
        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t pt-8 text-center md:flex-row">
          <p className="text-sm text-muted-foreground">
            © 2025 VladKuzmenko. All rights reserved.
          </p>
          
          <nav className="flex gap-4 text-sm">
            <Dialog>
              <DialogTrigger className="text-muted-foreground hover:text-foreground transition-colors">
                Privacy Policy
              </DialogTrigger>
              <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Privacy Policy</DialogTitle>
                </DialogHeader>
                <div className="prose prose-sm dark:prose-invert mt-4">
                  <h2>1. Information We Collect</h2>
                  <p>We collect information that you provide directly to us, including:</p>
                  <ul>
                    <li>Name and contact information</li>
                    <li>Account credentials</li>
                    <li>Payment information</li>
                    <li>Communication preferences</li>
                  </ul>

                  <h2>2. How We Use Your Information</h2>
                  <p>We use the information we collect to:</p>
                  <ul>
                    <li>Provide and maintain our services</li>
                    <li>Process your transactions</li>
                    <li>Send you technical notices and support messages</li>
                    <li>Communicate with you about products, services, and events</li>
                  </ul>

                  <h2>3. Data Security</h2>
                  <p>We implement appropriate technical and organizational security measures to protect your personal information against unauthorized access, modification, or destruction.</p>

                  <h2>4. Your Rights</h2>
                  <p>You have the right to:</p>
                  <ul>
                    <li>Access your personal data</li>
                    <li>Correct inaccurate data</li>
                    <li>Request deletion of your data</li>
                    <li>Object to data processing</li>
                  </ul>
                </div>
              </DialogContent>
            </Dialog>

            <Dialog>
              <DialogTrigger className="text-muted-foreground hover:text-foreground transition-colors">
                Terms of Service
              </DialogTrigger>
              <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>Terms of Service</DialogTitle>
                </DialogHeader>
                <div className="prose prose-sm dark:prose-invert mt-4">
                  <h2>1. Acceptance of Terms</h2>
                  <p>By accessing and using our services, you agree to be bound by these Terms of Service and all applicable laws and regulations.</p>

                  <h2>2. Use License</h2>
                  <p>We grant you a limited, non-exclusive, non-transferable license to use our services for your business purposes in accordance with these terms.</p>

                  <h2>3. Service Availability</h2>
                  <p>We strive to provide uninterrupted service but may need to perform maintenance or updates. We are not liable for any service interruptions or data loss.</p>

                  <h2>4. User Obligations</h2>
                  <ul>
                    <li>Maintain accurate account information</li>
                    <li>Protect your account credentials</li>
                    <li>Use services in compliance with laws</li>
                    <li>Respect intellectual property rights</li>
                  </ul>

                  <h2>5. Payment Terms</h2>
                  <p>You agree to pay all fees associated with your subscription plan. Fees are non-refundable unless otherwise specified.</p>
                </div>
              </DialogContent>
            </Dialog>

            <Dialog>
              <DialogTrigger className="text-muted-foreground hover:text-foreground transition-colors">
                Cookie Settings
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Cookie Settings</DialogTitle>
                </DialogHeader>
                <div className="space-y-4 mt-4">
                  <div className="flex items-center justify-between py-3 border-b">
                    <div>
                      <h3 className="font-medium">Essential Cookies</h3>
                      <p className="text-sm text-muted-foreground">Required for basic site functionality. Cannot be disabled.</p>
                    </div>
                    <div className="w-11 h-6 bg-brand rounded-full relative pointer-events-none">
                      <div className="w-5 h-5 bg-white rounded-full absolute top-0.5 left-0.5"></div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between py-3 border-b">
                    <div>
                      <h3 className="font-medium">Analytics Cookies</h3>
                      <p className="text-sm text-muted-foreground">Help us improve our website by collecting anonymous usage data.</p>
                    </div>
                    <div className="w-11 h-6 bg-muted rounded-full relative cursor-pointer hover:bg-muted/80 transition-colors">
                      <div className="w-5 h-5 bg-background border rounded-full absolute top-0.5 right-0.5 shadow-sm"></div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between py-3 border-b">
                    <div>
                      <h3 className="font-medium">Marketing Cookies</h3>
                      <p className="text-sm text-muted-foreground">Used to deliver personalized advertisements and track their effectiveness.</p>
                    </div>
                    <div className="w-11 h-6 bg-muted rounded-full relative cursor-pointer hover:bg-muted/80 transition-colors">
                      <div className="w-5 h-5 bg-background border rounded-full absolute top-0.5 right-0.5 shadow-sm"></div>
                    </div>
                  </div>

                  <div className="flex justify-end gap-4 mt-6">
                    <button className="px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                      Reject All
                    </button>
                    <button className="px-4 py-2 text-sm font-medium bg-brand text-white rounded-md hover:bg-brand/90 transition-colors">
                      Accept All
                    </button>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </nav>
        </div>
      </div>
    </footer>
  );
}