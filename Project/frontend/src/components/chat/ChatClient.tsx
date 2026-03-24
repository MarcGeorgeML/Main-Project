/* eslint-disable @typescript-eslint/no-explicit-any */
// components/chat/ChatClient.tsx
"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  sendVideoChat,
  getSessionHistory,
  getCurrentEmotion,
  getSessions,
  createSession,
  deleteSession,
  SessionSummary,
} from "@/lib/api";
import { ChatResponse } from "@/types/auth";
import { AxiosError } from "axios";
import SessionDrawer from "./SessionDrawer";

interface DisplayMessage {
  id: string;
  sender: "USER" | "AI";
  text: string;
  emotion?: string | null;
  confidence?: number | null;
  created_at: string;
}

const EMOTION_CONFIG: Record<string, { label: string; dot: string; icon: string }> = {
  happy:     { label: "Happy",     dot: "#F5D000", icon: "😊" },
  sad:       { label: "Sad",       dot: "#4A90D9", icon: "😔" },
  angry:     { label: "Angry",     dot: "#E8362A", icon: "😠" },
  fearful:   { label: "Fearful",   dot: "#9B59B6", icon: "😨" },
  disgusted: { label: "Disgusted", dot: "#4CAF50", icon: "😒" },
  surprised: { label: "Surprised", dot: "#F5821F", icon: "😲" },
  neutral:   { label: "Neutral",   dot: "#94A3B8", icon: "😐" },
};

function getEmotionConfig(emotion?: string | null) {
  if (!emotion) return EMOTION_CONFIG["neutral"];
  return EMOTION_CONFIG[emotion.toLowerCase()] ?? EMOTION_CONFIG["neutral"];
}

function getBestMimeType(): string {
  const candidates = ["video/mp4", "video/webm;codecs=h264,opus", "video/webm;codecs=vp8,opus", "video/webm"];
  for (const type of candidates) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

// ─── Tooltip ──────────────────────────────────────────────────────────────────
function EmotionTooltip({ emotion, confidence, children }: { emotion?: string | null; confidence?: number | null; children: React.ReactNode }) {
  const [visible, setVisible] = useState(false);
  const cfg = getEmotionConfig(emotion);
  return (
    <div className="relative inline-flex items-center justify-center" onMouseEnter={() => setVisible(true)} onMouseLeave={() => setVisible(false)}>
      {children}
      {visible && emotion && (
        <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 z-50 pointer-events-none px-2.5 py-1.5 rounded-lg text-xs font-medium text-white whitespace-nowrap shadow-lg"
          style={{ background: "rgba(15,15,20,0.88)", backdropFilter: "blur(6px)" }}>
          <span className="mr-1">{cfg.icon}</span>
          <span className="capitalize">{cfg.label}</span>
          {confidence != null && <span className="ml-1 opacity-70">{Math.round(confidence * 100)}%</span>}
          <span className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent" style={{ borderTopColor: "rgba(15,15,20,0.88)" }} />
        </div>
      )}
    </div>
  );
}

// ─── Dot ──────────────────────────────────────────────────────────────────────
function EmotionDot({ emotion, confidence }: { emotion?: string | null; confidence?: number | null }) {
  const cfg = getEmotionConfig(emotion);
  if (!emotion) return null;
  return (
    <EmotionTooltip emotion={emotion} confidence={confidence}>
      <span className="block w-2.5 h-2.5 rounded-full cursor-default ring-2 ring-white shadow-sm transition-transform duration-200 hover:scale-125"
        style={{ backgroundColor: cfg.dot }} />
    </EmotionTooltip>
  );
}

// ─── Record Button ────────────────────────────────────────────────────────────
function RecordButton({ isRecording, isLoading, currentEmotion, onToggle, onLabelClick }: {
  isRecording: boolean; isLoading: boolean; currentEmotion: string | null;
  onToggle: () => void; onLabelClick: () => void;
}) {
  const [emotionTagVisible, setEmotionTagVisible] = useState(false);
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cfg = getEmotionConfig(currentEmotion);

  return (
    <div className="flex items-center gap-5">
      <div className="w-8" />
      <div className="flex flex-col items-center gap-2">
        <div className="relative flex items-center justify-center"
          onMouseEnter={() => { hoverTimerRef.current = setTimeout(() => setEmotionTagVisible(true), 2000); }}
          onMouseLeave={() => { if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current); setEmotionTagVisible(false); }}>
          {isRecording && (
            <>
              <span className="absolute w-32 h-32 rounded-full bg-red-400/20 animate-ping" />
              <span className="absolute w-28 h-28 rounded-full bg-red-400/15 animate-ping" style={{ animationDelay: "0.3s" }} />
            </>
          )}
          {currentEmotion && !isRecording && (
            <span className="absolute rounded-full transition-all duration-700"
              style={{ inset: "-4px", background: `${cfg.dot}30`, boxShadow: `0 0 0 2px ${cfg.dot}55` }} />
          )}
          <button onClick={onToggle} disabled={isLoading}
            className={`relative w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 cursor-pointer shadow-xl hover:shadow-2xl hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed ${isRecording ? "bg-red-500 hover:bg-red-600" : "bg-white hover:bg-gray-50"}`}
            style={!isRecording ? { boxShadow: "0 4px 24px rgba(139,92,246,0.25), 0 2px 8px rgba(0,0,0,0.08)" } : {}}
            aria-label={isRecording ? "Stop recording" : "Start recording"}>
            {isLoading ? (
              <svg className="w-7 h-7 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="#8B5CF6" strokeWidth="4" />
                <path className="opacity-75" fill="#8B5CF6" d="M4 12a8 8 0 018-8v8z" />
              </svg>
            ) : isRecording ? (
              <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="2.5" />
              </svg>
            ) : (
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24">
                <defs>
                  <linearGradient id="micGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#8B5CF6" />
                    <stop offset="100%" stopColor="#D946EF" />
                  </linearGradient>
                </defs>
                <path stroke="url(#micGrad)" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            )}
          </button>
          {emotionTagVisible && currentEmotion && (
            <div className="absolute top-full mt-3 left-1/2 -translate-x-1/2 z-50 pointer-events-none flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium text-white whitespace-nowrap shadow-md"
              style={{ background: `${cfg.dot}ee`, boxShadow: `0 2px 12px ${cfg.dot}55` }}>
              <span>{cfg.icon}</span>
              <span className="capitalize">{cfg.label}</span>
            </div>
          )}
        </div>
        <p onClick={!isRecording && !isLoading ? onLabelClick : undefined}
          className="text-xs text-gray-400/80 tracking-wide select-none">
          {isLoading ? "Processing…" : isRecording ? "Recording — tap to stop" : "Tap to speak"}
        </p>
      </div>
      <div className="w-8" />
    </div>
  );
}

// ─── Message Bubble ───────────────────────────────────────────────────────────
function MessageBubble({ message }: { message: DisplayMessage }) {
  const isUser = message.sender === "USER";
  return (
    <div className={`flex items-end gap-2.5 ${isUser ? "justify-end" : "justify-start"} opacity-0 animate-[fadeSlideUp_0.5s_ease-out_forwards]`}>
      {!isUser && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src="/ai-chat3.png" alt="AI" className="w-8 h-8 rounded-full flex-shrink-0 object-cover shadow-sm" />
      )}
      <div className={`flex flex-col gap-1 max-w-sm ${isUser ? "items-end" : "items-start"}`}>
        <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm relative ${isUser ? "text-white rounded-br-sm" : "bg-white border border-gray-100 text-gray-800 rounded-bl-sm"}`}
          style={isUser ? { background: "linear-gradient(to right, #8B5CF6, #D946EF)" } : {}}>
          {message.text}
          {isUser && message.emotion && (
            <span className="absolute -bottom-1.5 -left-1.5">
              <EmotionDot emotion={message.emotion} confidence={message.confidence} />
            </span>
          )}
        </div>
        <span className="text-[10px] text-gray-400 px-1">
          {new Date(message.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
      {isUser && (
        <div className="w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center shadow-sm"
          style={{ background: "linear-gradient(135deg, #EDE9FE, #DDD6FE)" }}>
          <svg className="w-4 h-4" fill="#7C3AED" viewBox="0 0 24 24">
            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v1h16v-1c0-2.66-5.33-4-8-4z" />
          </svg>
        </div>
      )}
    </div>
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function ChatClient() {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(true);
  const [currentEmotion, setCurrentEmotion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const videoChunksRef = useRef<Blob[]>([]);
  const hiddenFileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  // Load sessions + auto-select latest on mount
  useEffect(() => {
    const init = async () => {
      setIsFetching(true);
      try {
        const [sessionsRes, emotionRes] = await Promise.all([
          getSessions(),
          getCurrentEmotion().catch(() => null),
        ]);

        const sessionList = sessionsRes.data;
        setSessions(sessionList);

        if (emotionRes?.data?.current_emotion) {
          setCurrentEmotion(emotionRes.data.current_emotion);
        }

        if (sessionList.length > 0) {
          // Auto-select the most recent session
          await loadSession(sessionList[0].session_id, false);
        }
      } catch (err) {
        console.error("Failed to load sessions:", err);
        setError("Failed to load conversations.");
      } finally {
        setIsFetching(false);
      }
    };
    init();
  }, []);

  const loadSession = async (sessionId: string, showFetching = true) => {
    if (showFetching) setIsFetching(true);
    try {
      const historyRes = await getSessionHistory(sessionId);
      const chats = historyRes.data;
      const display: DisplayMessage[] = chats.flatMap((chat: any) => {
        const msgs: DisplayMessage[] = [];
        if (chat.transcription) msgs.push({ id: `user-${chat.id}`, sender: "USER", text: chat.transcription, emotion: chat.emotion, confidence: chat.confidence, created_at: chat.created_at });
        if (chat.ai_response) msgs.push({ id: `ai-${chat.id}`, sender: "AI", text: chat.ai_response, created_at: chat.created_at });
        return msgs;
      });
      setMessages(display);
      setActiveSessionId(sessionId);
      setError(null);
    } catch (err) {
      console.error("Failed to load session:", err);
      setError("Failed to load conversation.");
    } finally {
      if (showFetching) setIsFetching(false);
    }
  };

  const handleNewSession = async () => {
    setIsCreatingSession(true);
    try {
      const res = await createSession();
      const newSessionId = res.data.session_id;

      // Refresh session list
      const sessionsRes = await getSessions();
      setSessions(sessionsRes.data);

      setMessages([]);
      setActiveSessionId(newSessionId);
      setDrawerOpen(false);
      setError(null);
    } catch (err) {
      console.error("Failed to create session:", err);
      setError("Failed to create new session.");
    } finally {
      setIsCreatingSession(false);
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
      const sessionsRes = await getSessions();
      const updated = sessionsRes.data;
      setSessions(updated);

      if (sessionId === activeSessionId) {
        if (updated.length > 0) {
          await loadSession(updated[0].session_id);
        } else {
          setMessages([]);
          setActiveSessionId(null);
        }
      }
    } catch (err) {
      console.error("Failed to delete session:", err);
      setError("Failed to delete session.");
    }
  };

  // Recording
  const startRecording = async () => {
    // Auto-create a session if none exists
    let sessionId = activeSessionId;
    if (!sessionId) {
      try {
        const res = await createSession();
        sessionId = res.data.session_id;
        const sessionsRes = await getSessions();
        setSessions(sessionsRes.data);
        setActiveSessionId(sessionId);
      } catch {
        setError("Failed to create session. Please try again.");
        return;
      }
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      const mimeType = getBestMimeType();
      const mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

      mediaRecorderRef.current = mediaRecorder;
      videoChunksRef.current = [];

      mediaRecorder.ondataavailable = (e: BlobEvent) => { if (e.data.size > 0) videoChunksRef.current.push(e.data); };
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const actualMime = mediaRecorder.mimeType || "video/webm";
        const videoBlob = new Blob(videoChunksRef.current, { type: actualMime });
        await processAndSendVideo(videoBlob, sessionId!);
      };

      mediaRecorder.start(250);
      setIsRecording(true);
    } catch (err) {
      console.error("Camera/microphone access denied:", err);
      setError("Could not access camera or microphone. Please check permissions.");
    }
  };

  const stopRecording = () => { mediaRecorderRef.current?.stop(); setIsRecording(false); };
  const handleToggleRecording = () => { if (isLoading) return; if (isRecording) stopRecording(); else startRecording(); };

  const processAndSendVideo = async (videoBlob: Blob, sessionId: string) => {
    setIsLoading(true);
    try {
      const response = await sendVideoChat(sessionId, videoBlob);
      const data: ChatResponse = response.data;

      const userMsg: DisplayMessage = { id: `user-${data.chat_id}`, sender: "USER", text: data.transcription || "(no speech detected)", emotion: data.emotion, confidence: data.confidence, created_at: data.created_at };
      const aiMsg: DisplayMessage = { id: `ai-${data.chat_id}`, sender: "AI", text: data.ai_response || "", created_at: data.created_at };

      setMessages((prev) => [...prev, userMsg, aiMsg]);

      if (data.latest_emotional_state) setCurrentEmotion(data.latest_emotional_state);
      else if (data.emotion) setCurrentEmotion(data.emotion);

      // Refresh session list so first_message + updated_at stay current
      const sessionsRes = await getSessions();
      setSessions(sessionsRes.data);

      setError(null);
    } catch (err) {
      const axErr = err as AxiosError;
      console.error("Error sending video:", axErr);
      setError("Failed to process recording. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleLabelClick = () => hiddenFileInputRef.current?.click();
  const handleHiddenFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = "";
    let sessionId = activeSessionId;
    if (!sessionId) {
      try {
        const res = await createSession();
        sessionId = res.data.session_id;
        const sessionsRes = await getSessions();
        setSessions(sessionsRes.data);
        setActiveSessionId(sessionId);
      } catch { setError("Failed to create session."); return; }
    }
    await processAndSendVideo(file, sessionId);
  };

  if (isFetching) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 rounded-full border-2 border-t-transparent animate-spin"
            style={{ borderColor: "#8B5CF6", borderTopColor: "transparent" }} />
          <p className="text-sm text-gray-400">Loading your conversations…</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex flex-col h-full w-full overflow-hidden">
      <input ref={hiddenFileInputRef} type="file" accept="video/mp4,video/*" className="hidden" onChange={handleHiddenFileChange} />

      {/* Session Drawer */}
      <SessionDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={(id) => loadSession(id)}
        onNewSession={handleNewSession}
        onDeleteSession={handleDeleteSession}
        isCreating={isCreatingSession}
      />

      {/* Background gradient */}
      <div className="pointer-events-none absolute bottom-0 left-0 right-0 z-0"
        style={{ height: "75%", background: "radial-gradient(ellipse 80% 60% at 50% 100%, rgba(139,92,246,0.30) 0%, rgba(217,70,239,0.20) 30%, rgba(139,92,246,0.10) 60%, transparent 100%)" }} />

      {/* Top bar — hamburger left, clear right */}
      <div className="relative z-10 flex items-center justify-between px-4 pt-4 flex-shrink-0">
        {/* Hamburger */}
        <button
          onClick={() => setDrawerOpen(true)}
          className="p-2 rounded-xl text-gray-400 hover:text-purple-600 hover:bg-purple-50 transition-all duration-200 cursor-pointer"
          aria-label="Open conversations"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        {/* Clear history */}
        {messages.length > 0 && activeSessionId && (
          <button
            onClick={async () => {
              if (!confirm("Delete this session? This cannot be undone.")) return;
              await handleDeleteSession(activeSessionId);
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium text-gray-400 hover:text-red-500 hover:bg-red-50 border border-transparent hover:border-red-200 transition-all duration-200 cursor-pointer"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
            Clear session
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="relative z-10 flex-1 overflow-y-auto px-4 sm:px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 gap-4 text-center">
              <div className="w-16 h-16 rounded-full flex items-center justify-center"
                style={{ background: "linear-gradient(135deg, rgba(139,92,246,0.15), rgba(217,70,239,0.15))" }}>
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24">
                  <defs>
                    <linearGradient id="emptyGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#8B5CF6" />
                      <stop offset="100%" stopColor="#D946EF" />
                    </linearGradient>
                  </defs>
                  <path stroke="url(#emptyGrad)" strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <div>
                <p className="text-gray-700 font-medium">
                  {activeSessionId ? "Start speaking" : "Create a session to begin"}
                </p>
                <p className="text-sm text-gray-400 mt-1">
                  {activeSessionId ? "Tap the button below and start speaking" : "Use the menu on the top left to start a new session"}
                </p>
              </div>
            </div>
          )}

          {messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)}

          {isLoading && (
            <div className="flex items-end gap-2.5">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/ai-chat3.png" alt="AI" className="w-8 h-8 rounded-full flex-shrink-0 object-cover shadow-sm" />
              <div className="bg-white border border-gray-100 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
                <div className="flex gap-1.5 items-center">
                  <span className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: "#8B5CF6" }} />
                  <span className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: "#A855F7", animationDelay: "0.15s" }} />
                  <span className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: "#D946EF", animationDelay: "0.3s" }} />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="relative z-10 mx-4 sm:mx-8 mb-2 px-4 py-2.5 bg-red-50 border border-red-200 rounded-xl flex items-center justify-between">
          <p className="text-sm text-red-600">{error}</p>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600 cursor-pointer ml-3">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}

      {/* Record button */}
      <div className="relative z-10 flex-shrink-0 pb-8 pt-4 flex flex-col items-center">
        <RecordButton
          isRecording={isRecording}
          isLoading={isLoading}
          currentEmotion={currentEmotion}
          onToggle={handleToggleRecording}
          onLabelClick={handleLabelClick}
        />
      </div>

      <style jsx global>{`
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}