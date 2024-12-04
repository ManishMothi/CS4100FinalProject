"use client";

// Page which displays predicted MBTI score from audio spoken by user.

import { useSearchParams } from "next/navigation";
import { Typography, Box } from "@mui/material";

export default function Prediction() {
  const searchParams = useSearchParams();
  const result = searchParams.get("result");

  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="center"
      height="100vh"
    >
      <Typography variant="h4">Your Personality is: {result}</Typography>
    </Box>
  );
}
