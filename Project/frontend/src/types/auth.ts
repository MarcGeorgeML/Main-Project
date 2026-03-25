// ==============================
// Chat Models
// ==============================

export interface Chat {
  id: string;
  video_url: string;
  transcription: string | null;
  ai_response: string | null;        
  emotion: string | null;
  confidence: number | null;
  created_at: string;
}

export interface ChatResponse {
  chat_id: string;
  transcription: string;
  ai_response: string;              
  emotion: string;
  confidence: number;
  latest_emotional_state: string;
  created_at: string;
  emotion_state?: string | null;
}

export type ChatHistoryResponse = Chat[];

export interface CurrentEmotionResponse {
  current_emotion: string;
}

// ==============================
// Axios response wrapper
// ==============================

export interface ApiResponse<T> {
  data: T;
  status: number;
  statusText: string;
}

// ==============================
// Auth types
// ==============================

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