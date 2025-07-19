import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Provider } from 'react-redux';
import { HelmetProvider } from 'react-helmet-async';

import { store } from './store';
import { Header } from './components/Common/Header';
import { Sidebar } from './components/Common/Sidebar';
import { Footer } from './components/Common/Footer';
import { Home } from './pages/Home';
import { Chat } from './pages/Chat';
import { Search } from './pages/Search';
import { Settings } from './pages/Settings';
import { ErrorBoundary } from './components/Common/ErrorBoundary';
import { LoadingProvider } from './contexts/LoadingContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#2a5298',
      dark: '#1e3c72',
      light: '#5a7bc8',
    },
    secondary: {
      main: '#ff6b35',
      dark: '#cc5429',
      light: '#ff8c61',
    },
    background: {
      default: '#f8f9fa',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        },
      },
    },
  },
});

// Create query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
  },
});

function App() {
  return (
    <HelmetProvider>
      <Provider store={store}>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <LoadingProvider>
              <NotificationProvider>
                <Router>
                  <ErrorBoundary>
                    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
                      <Header />
                      
                      <Box sx={{ display: 'flex', flex: 1 }}>
                        <Sidebar />
                        
                        <Box
                          component="main"
                          sx={{
                            flexGrow: 1,
                            p: 3,
                            width: { sm: `calc(100% - 240px)` },
                            ml: { sm: '240px' },
                          }}
                        >
                          <Routes>
                            <Route path="/" element={<Home />} />
                            <Route path="/chat" element={<Chat />} />
                            <Route path="/search" element={<Search />} />
                            <Route path="/settings" element={<Settings />} />
                          </Routes>
                        </Box>
                      </Box>
                      
                      <Footer />
                    </Box>
                  </ErrorBoundary>
                </Router>
              </NotificationProvider>
            </LoadingProvider>
          </ThemeProvider>
        </QueryClientProvider>
      </Provider>
    </HelmetProvider>
  );
}

export default App;
