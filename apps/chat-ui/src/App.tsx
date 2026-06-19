import { useEffect, useRef, useState } from "react";
import { Box, TextField, Button } from "@mui/material";

import ChatBubble from "@components/ChatBubble";
import Footer from "@components/Footer";
import MainContent from "@components/MainContent";
import Header from "@components/Header";
import Sidebar from "@components/Sidebar";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

import api from "./api";

export default function App() {
  const [drawerOpen, setDrawerOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");

  const sendPrompt = async () => {
    const cleaned = input.trim();
    // do more stuff as needed later on
    if (!cleaned) return;

    const newResp = crypto.randomUUID();
    // set new message first
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: "user",
        content: cleaned,
      },
      {
        id: newResp,
        role: "assistant",
        content: "",
      },
    ]);

    // clear the input so the user isn't staring at their text twice
    setInput("");

    // stream new message to latest item, guaranteed to be from the assistant
    for await (const chunk of api.sendPrompt(cleaned)) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === newResp ? { ...msg, content: msg.content + chunk } : msg,
        ),
      );
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages]);

  return (
    <Box sx={{ display: "flex", height: "100vh" }}>
      <Sidebar open={drawerOpen} />
      <Box
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        <Header onToolbarClick={() => setDrawerOpen((v) => !v)} />

        <MainContent>
          {messages.map((message, index) => (
            <ChatBubble
              key={index}
              isUser={message.role === "user"}
              content={message.content}
            />
          ))}
          <div ref={messagesEndRef} />
        </MainContent>

        {/* INPUT */}
        <Footer>
          <TextField
            fullWidth
            multiline
            maxRows={6}
            placeholder="Send a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendPrompt();
              }
            }}
          />

          <Button variant="contained" onClick={sendPrompt}>
            Send
          </Button>
        </Footer>
      </Box>
    </Box>
  );
}
