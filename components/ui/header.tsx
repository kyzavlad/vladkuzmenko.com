"use client";

import { Button } from "@/components/ui/button";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
} from "@/components/ui/navigation-menu";
import { Menu, MoveRight, X } from "lucide-react";
import { useState } from "react";
import Link from "next/link";
import { ContactDialog } from "@/components/ui/contact-dialog";
import { StarBorder } from "@/components/ui/star-border";
import { cn } from "@/lib/utils";

export function Header() {
  const [isOpen, setOpen] = useState(false);
  const [showMenu, setShowMenu] = useState(false);

  const navigationItems = [
    {
      title: "Home",
      href: "/",
      description: "",
    },
    {
      title: "Product",
      description: "Explore our AI automation solutions for your business.",
      items: [
        {
          title: "AI Automation Solutions",
          href: "#board-section",
        },
        {
          title: "Features",
          href: "#features-section",
        },
        {
          title: "Voice Assistant",
          href: "#audio-section",
        },
        {
          title: "Pricing",
          href: "#pricing-section",
        },
      ],
    },
    {
      title: "Company",
      description: "Learn more about our company and success stories.",
      items: [
        {
          title: "Success Stories",
          href: "#projects-section",
        },
        {
          title: "Global Network",
          href: "#map-section",
        },
        {
          title: "Testimonials",
          href: "#testimonials-section",
        },
        {
          title: "Blog",
          href: "#blog-section",
        },
      ],
    },
  ];

  const handleNavClick = (
    e: React.MouseEvent<HTMLAnchorElement>,
    href: string
  ) => {
    e.preventDefault();
    if (href.startsWith("#")) {
      const element = document.querySelector(href);
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
        setOpen(false);
      }
    } else {
      window.location.href = href;
    }
  };

  return (
    <>
      {/* Blur overlay */}
      {showMenu && (
        <div 
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          onClick={() => setShowMenu(false)}
        />
      )}

      {/* Menu popup */}
      {showMenu && (
        <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-lg px-4">
          <div className="bg-background/95 backdrop-blur-sm border border-border/40 rounded-2xl p-6 space-y-4">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold">Choose Your Path</h2>
              <Button 
                variant="ghost" 
                size="icon"
                onClick={() => setShowMenu(false)}
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
            
            <Link href="/#pricing-section" onClick={() => setShowMenu(false)}>
              <StarBorder 
                className="w-full mb-4 hover:scale-[1.02] transition-transform"
                color="hsl(var(--brand))"
              >
                <div className="flex items-center justify-between">
                  <div className="text-left">
                    <h3 className="font-semibold text-lg">AI Automation Services</h3>
                    <p className="text-muted-foreground text-sm">Transform your business with AI</p>
                  </div>
                  <MoveRight className="h-5 w-5" />
                </div>
              </StarBorder>
            </Link>

            <Link href="/platform" onClick={() => setShowMenu(false)}>
              <StarBorder 
                className="w-full mb-4 hover:scale-[1.02] transition-transform"
                color="hsl(var(--color-2))"
              >
                <div className="flex items-center justify-between">
                  <div className="text-left">
                    <h3 className="font-semibold text-lg">AI Automation Platform</h3>
                    <p className="text-muted-foreground text-sm">Access our powerful platform</p>
                  </div>
                  <MoveRight className="h-5 w-5" />
                </div>
              </StarBorder>
            </Link>

            <Link href="/team" onClick={() => setShowMenu(false)}>
              <StarBorder 
                className="w-full hover:scale-[1.02] transition-transform"
                color="hsl(var(--color-3))"
              >
                <div className="flex items-center justify-between">
                  <div className="text-left">
                    <h3 className="font-semibold text-lg">Warriors Team</h3>
                    <p className="text-muted-foreground text-sm">Join our elite community</p>
                  </div>
                  <MoveRight className="h-5 w-5" />
                </div>
              </StarBorder>
            </Link>
          </div>
        </div>
      )}

      <header className="w-full z-40 fixed top-0 left-0 bg-background/95 backdrop-blur-sm border-b border-border/40">
        <div className="container relative mx-auto py-4 md:py-5 flex gap-4 flex-row lg:grid lg:grid-cols-3 items-center">
          {/* Left side (Desktop navigation) */}
          <div className="justify-start items-center gap-4 lg:flex hidden flex-row">
            <NavigationMenu className="flex justify-start items-start">
              <NavigationMenuList className="flex justify-start gap-4 flex-row">
                {navigationItems.map((item) => (
                  <NavigationMenuItem key={item.title}>
                    {item.href ? (
                      <>
                        <NavigationMenuLink asChild>
                          <a
                            href={item.href}
                            onClick={(e) => handleNavClick(e, item.href!)}
                            className="inline-flex h-10 w-full items-center justify-center whitespace-nowrap rounded-md px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50"
                          >
                            {item.title}
                          </a>
                        </NavigationMenuLink>
                      </>
                    ) : (
                      <>
                        <NavigationMenuTrigger className="font-medium text-sm">
                          {item.title}
                        </NavigationMenuTrigger>
                        <NavigationMenuContent className="!w-[450px] p-4">
                          <div className="flex flex-col lg:grid grid-cols-2 gap-4">
                            <div className="flex flex-col h-full justify-between">
                              <div className="flex flex-col">
                                <p className="text-base">{item.title}</p>
                                <p className="text-muted-foreground text-sm">
                                  {item.description}
                                </p>
                              </div>
                              <ContactDialog triggerText="Book a call today">
                                <Button size="sm" className="mt-10">
                                  Book a call today
                                </Button>
                              </ContactDialog>
                            </div>
                            <div className="flex flex-col text-sm h-full justify-end">
                              {item.items?.map((subItem) => (
                                <a
                                  href={subItem.href}
                                  key={subItem.title}
                                  onClick={(e) => handleNavClick(e, subItem.href)}
                                  className="flex flex-row justify-between items-center hover:bg-muted py-2 px-4 rounded"
                                >
                                  <span>{subItem.title}</span>
                                  <MoveRight className="w-4 h-4 text-muted-foreground" />
                                </a>
                              ))}
                            </div>
                          </div>
                        </NavigationMenuContent>
                      </>
                    )}
                  </NavigationMenuItem>
                ))}
              </NavigationMenuList>
            </NavigationMenu>
          </div>

          {/* Center (Logo) */}
          <div className="flex lg:justify-center">
            <div className="logo-container relative">
              <a href="/" className="flex items-center">
                <span className="sr-only">VladKuzmenko.com</span>
                <div className="flex items-center">
                  <span className="text-2xl font-bold tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-foreground via-foreground to-foreground/30 dark:from-white dark:via-white dark:to-white/30 font-serif italic">
                    VladKuzmenko.com
                  </span>
                </div>
              </a>
            </div>
          </div>

          {/* Right side (Desktop buttons) */}
          <div className="flex justify-end w-full gap-4">
            <Button 
              variant="ghost" 
              className="hidden md:inline"
              onClick={() => setShowMenu(true)}
            >
              Choose Path
            </Button>
            <div className="border-r hidden md:inline"></div>
            <ContactDialog triggerText="Get started">
              <Button className="hidden md:inline">Get started</Button>
            </ContactDialog>
          </div>

          {/* Mobile menu button */}
          <div className="flex items-center justify-end lg:hidden">
            <Button variant="ghost" onClick={() => setOpen(!isOpen)}>
              {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>

            {isOpen && (
              <div className="absolute top-[72px] left-0 right-0 border-t flex flex-col bg-background shadow-lg py-4 px-4 gap-8 z-50">
                {navigationItems.map((item) => (
                  <div key={item.title}>
                    <div className="flex flex-col gap-2">
                      {item.href ? (
                        <a
                          href={item.href}
                          onClick={(e) => handleNavClick(e, item.href!)}
                          className="flex justify-between items-center"
                        >
                          <span className="text-lg">{item.title}</span>
                          <MoveRight className="w-4 h-4 stroke-1 text-muted-foreground" />
                        </a>
                      ) : (
                        <p className="text-lg">{item.title}</p>
                      )}
                      {item.items &&
                        item.items.map((subItem) => (
                          <a
                            key={subItem.title}
                            href={subItem.href}
                            onClick={(e) => handleNavClick(e, subItem.href)}
                            className="flex justify-between items-center"
                          >
                            <span className="text-muted-foreground">
                              {subItem.title}
                            </span>
                            <MoveRight className="w-4 h-4 stroke-1" />
                          </a>
                        ))}
                    </div>
                  </div>
                ))}

                <div className="border-t pt-4">
                  <Button 
                    className="w-full mb-4"
                    onClick={() => setShowMenu(true)}
                  >
                    Choose Path
                  </Button>
                  <ContactDialog triggerText="Get started">
                    <Button className="w-full">Get started</Button>
                  </ContactDialog>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>
    </>
  );
}