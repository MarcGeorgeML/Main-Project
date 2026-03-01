import axios, { AxiosResponse } from "axios";
import { getAccessToken, clearAuthData } from "./auth";

import type {
  ChatResponse,
  ChatHistoryResponse,
  CurrentEmotionResponse,
  User,
} from "@/types/auth.js";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  headers: { "Content-Type": "application/json" },
});

// 🔐 Auth interceptor
api.interceptors.request.use((config) => {
  const token = getAccessToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (error) => {
    if (error.response?.status === 401) clearAuthData();
    return Promise.reject(error);
  }
);

// ---------- Generic helpers ----------
export const fetchWithAuth = async <T>(
  endpoint: string
): Promise<AxiosResponse<T>> => api.get<T>(endpoint);

export const postFormWithAuth = async <T>(
  endpoint: string,
  data: FormData
): Promise<AxiosResponse<T>> =>
  api.post<T>(endpoint, data, {
    headers: { "Content-Type": "multipart/form-data" },
  });


// =====================================================
// 🎥 Chat APIs
// =====================================================

/**
 * Upload a video chat
 */
export const sendVideoChat = async (
  videoBlob: Blob
): Promise<AxiosResponse<ChatResponse>> => {
  const formData = new FormData();
  formData.append("video", videoBlob, "recording.mp4");

  return postFormWithAuth<ChatResponse>("/chats/video", formData);
};

/**
 * Get chat history
 */
export const getChatHistory = async (): Promise<
  AxiosResponse<ChatHistoryResponse>
> => fetchWithAuth<ChatHistoryResponse>("/chats");

export const deleteChatHistory = async (): Promise<AxiosResponse<{ message: string }>> =>
  api.delete("/chats");

/**
 * Get user's current emotional state
 */
export const getCurrentEmotion = async (): Promise<
  AxiosResponse<CurrentEmotionResponse>
> => fetchWithAuth<CurrentEmotionResponse>("/chats/emotion");


// =====================================================
// 🔐 Auth APIs
// =====================================================

export const getCurrentUser = async (): Promise<AxiosResponse<User>> =>
  fetchWithAuth<User>("/auth/me");

export const initiateGoogleLogin = async () => {
  const response = await api.get("/auth/google/login");
  return response.data;
};

export default api;