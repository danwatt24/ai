import { AppBar, Toolbar, IconButton, Typography } from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";

interface Props {
  onToolbarClick?: () => void;
}

export default function Header({ onToolbarClick }: Props) {
  return (
    <AppBar
      position="static"
      elevation={0}
      sx={{
        bgcolor: "background.default",
        borderBottom: "1px solid #1f2937",
      }}
    >
      <Toolbar>
        <IconButton edge="start" color="inherit" onClick={onToolbarClick}>
          <MenuIcon />
        </IconButton>

        <Typography variant="h6">Custom GPT</Typography>
      </Toolbar>
    </AppBar>
  );
}
