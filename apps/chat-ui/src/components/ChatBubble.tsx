import { Box, Avatar, Paper } from "@mui/material";
import ReactMarkdown from "react-markdown";

interface Props {
  isUser: boolean;
  content: string;
}

export default function ChatBubble({ isUser, content }: Props) {
  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
      }}
    >
      <Box
        sx={{
          display: "flex",
          gap: 2,
          maxWidth: "75%",
          flexDirection: isUser ? "row-reverse" : "row",
        }}
      >
        <Avatar
          sx={{
            bgcolor: isUser ? "primary.main" : "#374151",
          }}
        >
          {isUser ? "U" : "G"}
        </Avatar>

        <Paper
          elevation={0}
          sx={{
            p: 2,
            borderRadius: 3,
            bgcolor: isUser ? "primary.main" : "background.paper",
            border: "1px solid #1f2937",
          }}
        >
          <ReactMarkdown>{content}</ReactMarkdown>
        </Paper>
      </Box>
    </Box>
  );
}
