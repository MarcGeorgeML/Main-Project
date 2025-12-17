// Database model types matching backend
export interface Conversation {
  id: string;
  title: string;
  created_at: string;
}

export interface Message {
  id: string;
  sender: 'USER' | 'AI';
  message_type: 'TEXT' | 'AUDIO';
  message: string;
  created_at: string;
  transcription?: string;
}

export interface ConversationMessages {
  chat_id: string;
  messages: Message[];
}

// API Request types
export interface ConversationCreate {
  title?: string;
}

export interface ConversationUpdate {
  title: string;
}

export interface ChatRequest {
  message: string;
}

// API Response types
export interface ConversationResponse {
  id: string;
  title: string;
  created_at: string;
}

export type ConversationListResponse = ConversationResponse[];

export interface DeleteConversationResponse {
  message: string;
}

export interface UpdateConversationResponse {
  id: string;
  title: string;
}

export interface MessageResponse {
  id: string;
  message?: string;
  audio_filename?: string;
  created_at: string;
  transcribed_message?: string;
}

export interface TextMessageResponse {
  user_message: MessageResponse;
  ai_message: MessageResponse;
  request_id: string;
}

export interface AudioMessageResponse {
  user_message: MessageResponse;
  ai_message: MessageResponse;
  request_id: string;
}

export interface GetMessagesResponse {
  chat_id: string;
  messages: Message[];
}

// Axios response wrapper
export interface ApiResponse<T> {
  data: T;
  status: number;
  statusText: string;
}

// Auth types
export interface TokenData {
  user_id: string;
  email: string;
}

export interface User {
    id: string;
    email: string;
    first_name: string | null;
    last_name: string | null;
    profile_picture: string | null;
}

export interface AuthResponse {
    access_token: string;
    token_type: string;
    user: User;
}

export interface ChatPostRequestType{
    message: string
}