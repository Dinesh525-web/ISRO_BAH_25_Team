import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  Avatar,
  Chip,
  LinearProgress,
  IconButton,
  Collapse,
  Alert,
} from '@mui/material';
import {
  Send as SendIcon,
  Person as PersonIcon,
  SmartToy as BotIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  ContentCopy as CopyIcon,
  ThumbUp as ThumbUpIcon,
  ThumbDown as ThumbDownIcon,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

import { MessageBubble } from './MessageBubble';
import { InputField } from './InputField';
import { useChat } from '../../hooks/useChat';
import { ChatMessage } from '../../types/chat';
import { useNotification } from '../../contexts/NotificationContext';

interface ChatInterfaceProps {
  sessionId?: string;
  className?: string;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  sessionId,
  className,
}) => {
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showSources, setShowSources] = useState<{ [key: string]: boolean }>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { showNotification } = useNotification();
  const {
    messages,
    currentSession,
    sendMessage,
    isLoading,
    error,
    clearMessages,
  } = useChat(sessionId);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setIsTyping(true);

    try {
      await sendMessage(userMessage);
    } catch {
      showNotification('Failed to send message', 'error');
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const toggleSources = (messageId: string) => {
    setShowSources(prev => ({
      ...prev,
      [messageId]: !prev[messageId],
    }));
  };

  const handleCopyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
    showNotification('Message copied to clipboard', 'success');
  };

  const handleFeedback = async (messageId: string, type: 'positive' | 'negative') => {
    showNotification('Feedback recorded', 'success');
  };

  const renderMessage = (message: ChatMessage, index: number) => {
    const isUser = message.type === 'user';

    return (
      <motion.div
        key={message.id}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        style={{ marginBottom: 16 }}
      >
        <Box sx={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', mb: 2 }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'flex-start',
              maxWidth: '80%',
              flexDirection: isUser ? 'row-reverse' : 'row',
            }}
          >
            <Avatar
              sx={{
                bgcolor: isUser ? 'primary.main' : 'secondary.main',
                mx: 1,
                mt: 0.5,
              }}
            >
              {isUser ? <PersonIcon /> : <BotIcon />}
            </Avatar>

            <Paper
              elevation={1}
              sx={{
                p: 2,
                bgcolor: isUser ? 'primary.light' : 'background.paper',
                color: isUser ? 'primary.contrastText' : 'text.primary',
                borderRadius: 2,
                position: 'relative',
              }}
            >
              <Box sx={{ mb: 1 }}>
                <Typography variant="body1" component="div">
                  {isUser ? (
                    message.content
                  ) : (
                    <ReactMarkdown
                      components={{
                        code({ inline, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || '');
                          return !inline && match ? (
                            <SyntaxHighlighter
                              style={tomorrow}
                              language={match[1]}
                              PreTag="div"
                              {...props}
                            >
                              {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                          ) : (
                            <code className={className} {...props}>
                              {children}
                            </code>
                          );
                        },
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  )}
                </Typography>
              </Box>

              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  mt: 1,
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </Typography>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <IconButton
                    size="small"
                    onClick={() => handleCopyMessage(message.content)}
                    sx={{ color: 'text.secondary' }}
                  >
                    <CopyIcon fontSize="small" />
                  </IconButton>

                  {!isUser && (
                    <>
                      <IconButton
                        size="small"
                        onClick={() => handleFeedback(message.id, 'positive')}
                        sx={{ color: 'text.secondary' }}
                      >
                        <ThumbUpIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => handleFeedback(message.id, 'negative')}
                        sx={{ color: 'text.secondary' }}
                      >
                        <ThumbDownIcon fontSize="small" />
                      </IconButton>
                    </>
                  )}
                </Box>
              </Box>

              {!isUser && message.metadata?.confidence_score && (
                <Box sx={{ mt: 1 }}>
                  <Chip
                    label={`Confidence: ${(message.metadata.confidence_score * 100).toFixed(0)}%`}
                    size="small"
                    color={
                      message.metadata.confidence_score > 0.8
                        ? 'success'
                        : message.metadata.confidence_score > 0.5
                        ? 'warning'
                        : 'error'
                    }
                  />
                </Box>
              )}

              {!isUser &&
                message.retrievedDocuments &&
                message.retrievedDocuments.length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    <Button
                      size="small"
                      onClick={() => toggleSources(message.id)}
                      endIcon={
                        showSources[message.id] ? <ExpandLessIcon /> : <ExpandMoreIcon />
                      }
                    >
                      Sources ({message.retrievedDocuments.length})
                    </Button>

                    <Collapse in={showSources[message.id]}>
                      <Box sx={{ mt: 1 }}>
                        {message.retrievedDocuments.map((doc, docIndex) => (
                          <Paper
                            key={docIndex}
                            variant="outlined"
                            sx={{ p: 1, mb: 1, bgcolor: 'background.default' }}
                          >
                            <Typography variant="body2" fontWeight="bold">
                              {doc.title}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              Relevance: {(doc.relevance_score * 100).toFixed(0)}%
                            </Typography>
                            {doc.source_url && (
                              <Typography
                                variant="caption"
                                component="a"
                                href={doc.source_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                sx={{ display: 'block', color: 'primary.main' }}
                              >
                                View Source
                              </Typography>
                            )}
                          </Paper>
                        ))}
                      </Box>
                    </Collapse>
                  </Box>
                )}
            </Paper>
          </Box>
        </Box>
      </motion.div>
    );
  };

  return (
    <Box className={className} sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6">Chat with MOSDAC AI</Typography>
        {currentSession && (
          <Typography variant="caption" color="text.secondary">
            Session: {currentSession.title}
          </Typography>
        )}
      </Box>

      <Box
        sx={{
          flex: 1,
          p: 2,
          overflow: 'auto',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <AnimatePresence>
          {messages.map((message, index) => renderMessage(message, index))}
        </AnimatePresence>

        {isTyping && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <Avatar sx={{ bgcolor: 'secondary.main', mr: 1 }}>
                <BotIcon />
              </Avatar>
              <Paper
                elevation={1}
                sx={{
                  p: 2,
                  bgcolor: 'background.paper',
                  borderRadius: 2,
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  MOSDAC AI is typing...
                </Typography>
                <LinearProgress sx={{ mt: 1, width: 200 }} />
              </Paper>
            </Box>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </Box>

      {error && (
        <Alert severity="error" sx={{ m: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything about MOSDAC, satellites, or meteorological data..."
            disabled={isLoading}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: 2,
              },
            }}
          />
          <Button
            variant="contained"
            onClick={handleSendMessage}
            disabled={!input.trim() || isLoading}
            sx={{
              minWidth: 56,
              borderRadius: 2,
            }}
          >
            <SendIcon />
          </Button>
        </Box>

        <Box sx={{ display: 'flex', gap: 1, mt: 1, flexWrap: 'wrap' }}>
          {[
            'Latest cyclone information',
            'INSAT-3D data download',
            'Satellite status',
            'Ocean wind data',
          ].map((suggestion) => (
            <Chip
              key={suggestion}
              label={suggestion}
              size="small"
              variant="outlined"
              onClick={() => setInput(suggestion)}
              sx={{ cursor: 'pointer' }}
            />
          ))}
        </Box>
      </Box>
    </Box>
  );
};
