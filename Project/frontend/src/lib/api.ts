import axios, { AxiosResponse } from 'axios';
import { getAccessToken, clearAuthData } from './auth';
import {
  ConversationUpdate,
  ChatRequest,
  ConversationResponse,
  ConversationListResponse,
  DeleteConversationResponse,
  UpdateConversationResponse,
  GetMessagesResponse,
  TextMessageResponse,
  AudioMessageResponse,
  User
} from '@/types/auth';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create Axios instance
const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add Authorization header automatically
api.interceptors.request.use((config) => {
    const token = getAccessToken();
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

// Handle unauthorized errors globally
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      clearAuthData();
    }
    return Promise.reject(error);
  }
);


// Generic GET request with auth
export const fetchWithAuth = async <T>(endpoint: string): Promise<AxiosResponse<T>> => {
    return api.get<T>(endpoint);
};

// Generic POST request with auth and payload
export const postWithAuth = async <TRequest, TResponse>(
    endpoint: string, 
    data?: TRequest
): Promise<AxiosResponse<TResponse>> => {
    return api.post<TResponse>(endpoint, data);
};

// Generic PATCH request with auth
export const patchWithAuth = async <TRequest, TResponse>(
    endpoint: string, 
    data: TRequest
): Promise<AxiosResponse<TResponse>> => {
    return api.patch<TResponse>(endpoint, data);
};

// Generic DELETE request with auth
export const deleteWithAuth = async <T>(endpoint: string): Promise<AxiosResponse<T>> => {
    return api.delete<T>(endpoint);
};

// POST request with FormData
export const postFormWithAuth = async <T>(
    endpoint: string, 
    data: FormData
): Promise<AxiosResponse<T>> => {
    return api.post<T>(endpoint, data, {
        headers: { "Content-Type": "multipart/form-data" }
    });
};

// Conversation APIs
export const createConversation = async (): Promise<AxiosResponse<ConversationResponse>> => {
    // Backend auto-generates title, so no payload needed
    return postWithAuth<void, ConversationResponse>('/chats');
};

export const getAllConversations = async (): Promise<AxiosResponse<ConversationListResponse>> => {
    return fetchWithAuth<ConversationListResponse>('/chats');
};

export const deleteConversation = async (
    chatId: string
): Promise<AxiosResponse<DeleteConversationResponse>> => {
    return deleteWithAuth<DeleteConversationResponse>(`/chats/${chatId}`);
};

export const updateConversationTitle = async (
    chatId: string, 
    title: string
): Promise<AxiosResponse<UpdateConversationResponse>> => {
    const payload: ConversationUpdate = { title };
    return patchWithAuth<ConversationUpdate, UpdateConversationResponse>(
        `/chats/${chatId}`, 
        payload
    );
};

export const getConversationMessages = async (
    chatId: string
): Promise<AxiosResponse<GetMessagesResponse>> => {
    return fetchWithAuth<GetMessagesResponse>(`/chats/${chatId}/messages`);
};

// Message APIs
export const sendTextMessage = async (
    chatId: string, 
    message: string
): Promise<AxiosResponse<TextMessageResponse>> => {
    const payload: ChatRequest = { message };
    return postWithAuth<ChatRequest, TextMessageResponse>(
        `/chats/${chatId}/messages/text`, 
        payload
    );
};

export const sendAudioMessage = async (
    chatId: string, 
    audioBlob: Blob
): Promise<AxiosResponse<AudioMessageResponse>> => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');
    return postFormWithAuth<AudioMessageResponse>(
        `/chats/${chatId}/messages/audio`, 
        formData
    );
};

// Get audio file URL
export const getAudioUrl = (filename: string): string => {
  return `${API_URL}/temp/${filename}`;
};


// Get current user
export const getCurrentUser = async (): Promise<AxiosResponse<User>> => {
    const response = await fetchWithAuth<User>('/auth/me');
    return response;
};

// Initiate Google login
export const initiateGoogleLogin = async () => {
    const response = await api.get('/auth/google/login');
    return response.data;
};

export default api;