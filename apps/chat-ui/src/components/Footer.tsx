import { Box } from "@mui/material";

export default function Footer({ children }: React.PropsWithChildren) {
  return (
    <Box
      sx={{
        p: 2,
        borderTop: "1px solid #1f2937",
        bgcolor: "background.default",
      }}
    >
      <Box
        sx={{
          display: "flex",
          gap: 2,
          maxWidth: 1000,
          mx: "auto",
        }}
      >
        {children}
      </Box>
    </Box>
  );
}
