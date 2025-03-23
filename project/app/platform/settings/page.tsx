"use client";

import { Header } from "@/components/ui/header";
import { FooterSection } from "@/components/FooterSection";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Bell, Mail, Shield, Moon } from "lucide-react";

export default function SettingsPage() {
  return (
    <main className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto py-12">
        <h1 className="text-3xl font-bold mb-8">Settings</h1>
        
        <div className="grid gap-6">
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Account Preferences</h2>
            <div className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="name">Display Name</Label>
                <Input id="name" defaultValue="Vlad Kuzmenko" />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="email">Email Address</Label>
                <Input id="email" type="email" defaultValue="vlad@example.com" />
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Notifications</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Bell className="w-4 h-4" />
                  <Label htmlFor="push">Push Notifications</Label>
                </div>
                <Switch id="push" defaultChecked />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Mail className="w-4 h-4" />
                  <Label htmlFor="email-notif">Email Notifications</Label>
                </div>
                <Switch id="email-notif" defaultChecked />
              </div>
            </div>
          </Card>

          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Privacy & Security</h2>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Shield className="w-4 h-4" />
                  <Label htmlFor="2fa">Two-Factor Authentication</Label>
                </div>
                <Switch id="2fa" />
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Moon className="w-4 h-4" />
                  <Label htmlFor="dark-mode">Dark Mode</Label>
                </div>
                <Switch id="dark-mode" defaultChecked />
              </div>
            </div>
          </Card>

          <div className="flex justify-end gap-4">
            <Button variant="outline">Cancel</Button>
            <Button>Save Changes</Button>
          </div>
        </div>
      </div>
      <FooterSection />
    </main>
  );
}