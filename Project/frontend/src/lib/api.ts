// lib/api.ts
import axios, { AxiosResponse } from "axios";
import { getAccessToken, clearAuthData } from "./auth";
import type { ChatResponse, CurrentEmotionResponse, User } from "@/types/auth.js";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_URL,
  headers: { "Content-Type": "application/json" },
});

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

export const fetchWithAuth = async <T>(endpoint: string): Promise<AxiosResponse<T>> =>
  api.get<T>(endpoint);

export const postFormWithAuth = async <T>(
  endpoint: string,
  data: FormData
): Promise<AxiosResponse<T>> =>
  api.post<T>(endpoint, data, { headers: { "Content-Type": "multipart/form-data" } });

// ── Sessions ──────────────────────────────────────────────────────────────────

export interface SessionSummary {
  session_id: string;
  first_message: string | null;
  updated_at: string;
  created_at: string;
}

export const createSession = (): Promise<AxiosResponse<{ session_id: string; created_at: string }>> =>
  api.post("/chats/sessions");

export const getSessions = (): Promise<AxiosResponse<SessionSummary[]>> =>
  api.get("/chats/sessions");

export const deleteSession = (sessionId: string): Promise<AxiosResponse<{ message: string }>> =>
  api.delete(`/chats/sessions/${sessionId}`);

// ── Chat ──────────────────────────────────────────────────────────────────────

export const sendVideoChat = (
  sessionId: string,
  videoBlob: Blob
): Promise<AxiosResponse<ChatResponse>> => {
  const formData = new FormData();
  formData.append("video", videoBlob, "recording.mp4");
  return postFormWithAuth<ChatResponse>(`/chats/sessions/${sessionId}/video`, formData);
};

export const getSessionHistory = (sessionId: string) =>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  fetchWithAuth<any[]>(`/chats/sessions/${sessionId}`);

export const deleteChatHistory = (): Promise<AxiosResponse<{ message: string }>> =>
  api.delete("/chats");

export const getCurrentEmotion = (): Promise<AxiosResponse<CurrentEmotionResponse>> =>
  fetchWithAuth<CurrentEmotionResponse>("/chats/emotion");

export const getCurrentUser = (): Promise<AxiosResponse<User>> =>
  fetchWithAuth<User>("/auth/me");

export const initiateGoogleLogin = async () => {
  const response = await api.get("/auth/google/login");
  return response.data;
};

export default api;