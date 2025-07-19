import { useState } from 'react';
import { v4 as uuid } from 'uuid';
import { api } from '../services/api';
import { ChatMessage, ChatSession } from '../types/chat';

export const useChat = (initialSessionId?: string) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentSession, setSession] = useState<ChatSession | null>(null);
  const [isLoading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendMessage = async (content: string) => {
    setLoading(true);
    setError(null);

    try {
      const sessionId = currentSession?.id || initialSessionId || uuid();
      const res = await api.post('/api/v1/chat/', {
        message: content,
        session_id: sessionId,
        stream: false,
      });

      setSession({ id: res.session_id, title: res.metadata?.session_title });
      setMessages((prev) => [
        ...prev,
        { id: uuid(), type: 'user', content, timestamp: Date.now() } as ChatMessage,
        { ...res, type: 'assistant' } as ChatMessage,
      ]);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const clearMessages = () => setMessages([]);

  return { messages, currentSession, sendMessage, isLoading, error, clearMessages };
};
