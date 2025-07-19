import React from 'react';
import { AppBar, Toolbar, Typography } from '@mui/material';
import SatelliteIcon from '@mui/icons-material/Satellite';

export const Header: React.FC = () => (
  <AppBar position="fixed" color="primary" elevation={1}>
    <Toolbar>
      <SatelliteIcon sx={{ mr: 1 }} />
      <Typography variant="h6" noWrap>
        MOSDAC AI Navigator
      </Typography>
    </Toolbar>
  </AppBar>
);
