import {
  Drawer,
  Toolbar,
  Typography,
  Divider,
  List,
  ListItemButton,
  ListItemText,
} from "@mui/material";

const drawerWidth = 260;

interface Props {
  open: boolean;
}

export default function Sidebar({ open: drawerOpen }: Props) {
  return (
    <Drawer
      variant="persistent"
      open={drawerOpen}
      sx={{
        width: drawerOpen ? drawerWidth : 0,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: drawerWidth,
          boxSizing: "border-box",
          bgcolor: "background.paper",
          borderRight: "1px solid #1f2937",
        },
      }}
    >
      <Toolbar>
        <Typography variant="h6">Chats</Typography>
      </Toolbar>

      <Divider />

      <List>
        <ListItemButton>
          <ListItemText primary="Stub Chat" />
        </ListItemButton>
      </List>
    </Drawer>
  );
}
