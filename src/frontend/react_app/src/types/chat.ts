export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: number;
  metadata?: any;
  retrievedDocuments?: any[];
}

export interface ChatSession {
  id: string;
  title?: string;
}
