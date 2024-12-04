"use client";

// Redirects to AudioRecorder Page

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    router.push("/AudioRecorder");
  }, [router]);

  return <h1>Redirecting...</h1>;
}
