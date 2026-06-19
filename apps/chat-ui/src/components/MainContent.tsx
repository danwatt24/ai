import { Box } from "@mui/material";

export default function MainContent({ children }: React.PropsWithChildren) {
  return (
    <Box
      sx={{
        flex: 1,
        overflowY: "auto",
        px: 3,
        py: 4,
        display: "flex",
        flexDirection: "column",
        gap: 3,
      }}
    >
      {children}
    </Box>
  );
}
