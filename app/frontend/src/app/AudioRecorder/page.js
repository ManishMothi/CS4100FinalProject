"use client";
import React, { useState } from "react";
import { Button, Box, Stack, Typography } from "@mui/material/";
import MicIcon from "@mui/icons-material/Mic";
import { useRouter } from "next/navigation";

const config = require("../../config.json");

// Press buttom to talk for 20 seconds. Recording is sent to server and rerouted to Prediction Page for predicted MBTI type.

export default function AudioRecorder() {
  const router = useRouter();
  const [isRecording, setRecording] = useState(false);
  const [result, setResult] = useState("");

  const startRecording = async () => {
    setRecording(true);
    const response = await fetch(
      `http://${config.server_host}:${config.server_port}/api/record`,
      { method: "GET" }
    );

    let data = await response.text();
    data = data.trim().replace(/^"|"$/g, "");
    setResult(data);

    router.push(`/Prediction?result=${encodeURIComponent(data)}`);
  };

  const stopRecording = () => {
    setRecording(false);
  };

  const keyframes = `@keyframes pulse {
    0% {
      transform: scale(1);
      opacity: 1;
    }
    50% {
      transform: scale(1.1);
      opacity: 0.7;
    }
    100% {
      transform: scale(1);
      opacity: 1;
    }
  }
  @keyframes pulse-shadow {
    0% {
      box-shadow: 0 0 0 15px rgba(168, 85, 247, 0.5); /* Change shadow color to match the gradient */
    }
    50% {
      box-shadow: 0 0 0 25px rgba(168, 85, 247, 0.7);
    }
    100% {
      box-shadow: 0 0 0 15px rgba(168, 85, 247, 0.5);
    }
  }
`;

  return (
    <div>
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        height="90vh"
        width="100vw"
        flexDirection="column"
      >
        <Stack spacing={5}>
          <Button
            sx={{
              background: "linear-gradient(90deg, #ff7b72, #a855f7)",
              borderRadius: "50%",
              width: 200,
              height: 200,
              minWidth: 0,
              "&:hover": {
                boxShadow: "0 0 0 10px rgba(128, 128, 128, 0.3)",
              },
              animation: isRecording ? "pulse-shadow 1s infinite" : "none",
              boxShadow: isRecording
                ? "0 0 0 15px rgba(168, 85, 247, 0.5)"
                : "none",
            }}
            onClick={isRecording ? stopRecording : startRecording}
          >
            <MicIcon
              style={{ color: "white", width: 90, height: 90 }}
            ></MicIcon>
          </Button>
        </Stack>
        <Typography variant="h6" align="center" mt={4} sx={{ color: "white" }}>
          Press and talk for 20 seconds to get your personality!
        </Typography>
      </Box>
      <style>{keyframes}</style>
    </div>
  );
}
