"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function LogoutPage() {
  const router = useRouter();

  useEffect(() => {
    // Clear local storage
    localStorage.clear();
    // Redirect to home page
    router.push("/");
  }, [router]);

  return null;
}