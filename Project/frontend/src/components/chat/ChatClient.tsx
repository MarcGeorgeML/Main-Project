"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { sendVideoChat, getChatHistory, getCurrentEmotion, deleteChatHistory } from "@/lib/api";
import { Chat, ChatResponse } from "@/types/auth";
import { AxiosError } from "axios";

interface DisplayMessage {
  id: string;
  sender: "USER" | "AI";
  text: string;
  emotion?: string | null;
  confidence?: number | null;
  created_at: string;
}

const EMOTION_CONFIG: Record<string, { color: string; bg: string; icon: string }> = {
  happy:     { color: "text-amber-600",  bg: "bg-amber-50 border-amber-200",   icon: "😊" },
  sad:       { color: "text-blue-500",   bg: "bg-blue-50 border-blue-200",     icon: "😔" },
  angry:     { color: "text-red-500",    bg: "bg-red-50 border-red-200",       icon: "😠" },
  fearful:   { color: "text-violet-500", bg: "bg-violet-50 border-violet-200", icon: "😨" },
  disgusted: { color: "text-green-600",  bg: "bg-green-50 border-green-200",   icon: "😒" },
  surprised: { color: "text-orange-500", bg: "bg-orange-50 border-orange-200", icon: "😲" },
  neutral:   { color: "text-slate-500",  bg: "bg-slate-50 border-slate-200",   icon: "😐" },
};

function getEmotionConfig(emotion?: string | null) {
  if (!emotion) return EMOTION_CONFIG["neutral"];
  return EMOTION_CONFIG[emotion.toLowerCase()] ?? EMOTION_CONFIG["neutral"];
}

// ─── Emotion Badge + Record Button ───────────────────────────────────────────

function RecordButton({
  isRecording,
  isLoading,
  currentEmotion,
  onToggle,
}: {
  isRecording: boolean;
  isLoading: boolean;
  currentEmotion: string | null;
  onToggle: () => void;
}) {
  const cfg = getEmotionConfig(currentEmotion);

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Emotion badge */}
      <div
        className={`
          flex items-center gap-2 px-4 py-1.5 rounded-full border text-xs font-semibold
          transition-all duration-500 backdrop-blur-sm
          ${currentEmotion
            ? `${cfg.bg} ${cfg.color}`
            : "bg-white/80 border-gray-200 text-gray-400"
          }
        `}
      >
        {currentEmotion ? (
          <>
            <span className="text-sm">{cfg.icon}</span>
            <span className="capitalize tracking-wide">{currentEmotion}</span>
            <span className="opacity-50">detected</span>
          </>
        ) : (
          <>
            <span className="w-1.5 h-1.5 rounded-full bg-gray-300 inline-block" />
            <span>no emotion detected yet</span>
          </>
        )}
      </div>

      {/* Button with pulse rings */}
      <div className="relative flex items-center justify-center">
        {isRecording && (
          <>
            <span className="absolute w-32 h-32 rounded-full bg-red-400/20 animate-ping" />
            <span
              className="absolute w-28 h-28 rounded-full bg-red-400/15 animate-ping"
              style={{ animationDelay: "0.3s" }}
            />
          </>
        )}

        {/* Ambient glow from emotion colour */}
        {currentEmotion && !isRecording && (
          <span
            className={`absolute w-28 h-28 rounded-full opacity-30 blur-xl transition-all duration-1000 ${cfg.bg.split(" ")[0]}`}
          />
        )}

        <button
          onClick={onToggle}
          disabled={isLoading}
          className={`
            relative w-20 h-20 rounded-full flex items-center justify-center
            transition-all duration-300 cursor-pointer
            shadow-xl hover:shadow-2xl hover:scale-105 active:scale-95
            disabled:opacity-50 disabled:cursor-not-allowed
            ${isRecording ? "bg-red-500 hover:bg-red-600" : "bg-purple-600 hover:bg-purple-700"}
          `}
          aria-label={isRecording ? "Stop recording" : "Start recording"}
        >
          {isLoading ? (
            <svg className="w-7 h-7 text-white animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
          ) : isRecording ? (
            <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="6" width="12" height="12" rx="2.5" />
            </svg>
          ) : (
            <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
              />
            </svg>
          )}
        </button>
      </div>

      <p className="text-xs text-gray-400 tracking-wide">
        {isLoading
          ? "Processing…"
          : isRecording
          ? "Recording — tap to stop"
          : "Tap to speak"}
      </p>
    </div>
  );
}

// ─── Single message bubble ────────────────────────────────────────────────────

function MessageBubble({ message }: { message: DisplayMessage }) {
  const isUser = message.sender === "USER";
  const cfg = getEmotionConfig(message.emotion);

  return (
    <div
      className={`flex items-end gap-3 ${isUser ? "justify-end" : "justify-start"}
        opacity-0 animate-[fadeSlideUp_0.5s_ease-out_forwards]`}
    >
      {/* AI avatar */}
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-purple-700 flex-shrink-0 flex items-center justify-center shadow-md">
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15M14.25 3.104c.251.023.501.05.75.082M19.8 15l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.607L5 14.5m14.8.5l.36.144a.75.75 0 010 1.413l-.36.144M5 14.5l-.36.144a.75.75 0 000 1.413L5 16.5"
            />
          </svg>
        </div>
      )}

      <div className={`flex flex-col gap-1 max-w-sm ${isUser ? "items-end" : "items-start"}`}>
        {/* Emotion chip — only on user messages */}
        {isUser && message.emotion && (
          <div
            className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border text-xs font-medium ${cfg.bg} ${cfg.color}`}
          >
            <span>{cfg.icon}</span>
            <span className="capitalize">{message.emotion}</span>
            {message.confidence != null && (
              <span className="opacity-50">{Math.round(message.confidence * 100)}%</span>
            )}
          </div>
        )}

        <div
          className={`
            px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm
            ${isUser
              ? "bg-purple-600 text-white rounded-br-sm"
              : "bg-white border border-gray-100 text-gray-800 rounded-bl-sm"
            }
          `}
        >
          {message.text}
        </div>

        <span className="text-[10px] text-gray-400 px-1">
          {new Date(message.created_at).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </span>
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-zinc-300 flex-shrink-0 flex items-center justify-center shadow-sm">
          <svg className="w-4 h-4 text-zinc-600" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z" />
          </svg>
        </div>
      )}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function ChatClient() {
  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isFetching, setIsFetching] = useState(true);
  const [currentEmotion, setCurrentEmotion] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const videoChunksRef = useRef<Blob[]>([]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Load history + current emotion on mount
  useEffect(() => {
    const loadHistory = async () => {
      setIsFetching(true);
      try {
        const [historyRes, emotionRes] = await Promise.all([
          getChatHistory(),
          getCurrentEmotion().catch(() => null),
        ]);

        const chats: Chat[] = historyRes.data;

        // Build flat list: user message then AI message per chat entry
        const display: DisplayMessage[] = chats.flatMap((chat) => {
          const msgs: DisplayMessage[] = [];

          if (chat.transcription) {
            msgs.push({
              id: `user-${chat.id}`,
              sender: "USER",
              text: chat.transcription,
              emotion: chat.emotion,
              confidence: chat.confidence,
              created_at: chat.created_at,
            });
          }

          if (chat.ai_response) {
            msgs.push({
              id: `ai-${chat.id}`,
              sender: "AI",
              text: chat.ai_response,
              created_at: chat.created_at,
            });
          }

          return msgs;
        });

        setMessages(display);

        if (emotionRes?.data?.current_emotion) {
          setCurrentEmotion(emotionRes.data.current_emotion);
        }
      } catch (err) {
        console.error("Failed to load history:", err);
        setError("Failed to load chat history.");
      } finally {
        setIsFetching(false);
      }
    };

    loadHistory();
  }, []);

  // ── Recording ──────────────────────────────────────────────────────────────

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "video/webm;codecs=vp8,opus",
      });
      mediaRecorderRef.current = mediaRecorder;
      videoChunksRef.current = [];

      mediaRecorder.ondataavailable = (e: BlobEvent) => {
        if (e.data.size > 0) videoChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        // Stop all camera/mic tracks immediately
        stream.getTracks().forEach((t) => t.stop());

        const videoBlob = new Blob(videoChunksRef.current, { type: "video/webm" });

        setIsLoading(true);
        try {
          const response = await sendVideoChat(videoBlob);
          const data: ChatResponse = response.data;

          const userMsg: DisplayMessage = {
            id: `user-${data.chat_id}`,
            sender: "USER",
            text: data.transcription || "(no speech detected)",
            emotion: data.emotion,
            confidence: data.confidence,
            created_at: data.created_at,
          };

          const aiMsg: DisplayMessage = {
            id: `ai-${data.chat_id}`,
            sender: "AI",
            text: data.ai_response || "",
            created_at: data.created_at,
          };

          setMessages((prev) => [...prev, userMsg, aiMsg]);

          // Update ambient emotion indicator
          if (data.latest_emotional_state) {
            setCurrentEmotion(data.latest_emotional_state);
          } else if (data.emotion) {
            setCurrentEmotion(data.emotion);
          }

          setError(null);
        } catch (err) {
          const axErr = err as AxiosError;
          console.error("Error sending video:", axErr);
          setError("Failed to process recording. Please try again.");
        } finally {
          setIsLoading(false);
        }
      };

      mediaRecorder.start(250); // collect a chunk every 250 ms
      setIsRecording(true);
    } catch (err) {
      console.error("Camera/microphone access denied:", err);
      setError("Could not access camera or microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  const handleToggleRecording = () => {
    if (isLoading) return;
    if (isRecording) stopRecording();
    else startRecording();
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  if (isFetching) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="flex flex-col items-center gap-3">
          <div className="w-10 h-10 rounded-full border-2 border-purple-500 border-t-transparent animate-spin" />
          <p className="text-sm text-gray-400">Loading your conversations…</p>
        </div>
      </div>
    );
  }

  const handleClearHistory = async () => {
    if (!confirm("Clear all chat history? This cannot be undone.")) return;
    try {
      await deleteChatHistory();
      setMessages([]);
      setCurrentEmotion(null);
      setError(null);
    } catch (err) {
      console.error("Failed to clear history:", err);
      setError("Failed to clear chat history.");
    }
  };

  return (
    <div className="relative flex flex-col h-full">
      {/* ── Clear history button ── */}
      {messages.length > 0 && (
        <div className="absolute top-4 right-4 z-10">
          <button
            onClick={handleClearHistory}
            title="Clear chat history"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium text-gray-400 hover:text-red-500 hover:bg-red-50 border border-transparent hover:border-red-200 transition-all duration-200 cursor-pointer"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
            Clear history
          </button>
        </div>
      )}

      {/* ── Messages ── */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-6">
        <div className="max-w-2xl mx-auto space-y-4">

          {/* Empty state */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-24 gap-4 text-center">
              <div className="w-16 h-16 rounded-full bg-purple-100 flex items-center justify-center">
                <svg className="w-8 h-8 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                  />
                </svg>
              </div>
              <div>
                <p className="text-gray-700 font-medium">Start a conversation</p>
                <p className="text-sm text-gray-400 mt-1">
                  Tap the button below and start speaking
                </p>
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}

          {/* Processing / thinking indicator */}
          {isLoading && (
            <div className="flex items-end gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-purple-700 flex-shrink-0 flex items-center justify-center shadow-md">
                <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M14.25 3.104c.251.023.501.05.75.082M19.8 15l-1.57.393A9.065 9.065 0 0112 15"
                  />
                </svg>
              </div>
              <div className="bg-white border border-gray-100 rounded-2xl rounded-bl-sm px-4 py-3 shadow-sm">
                <div className="flex gap-1.5 items-center">
                  <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" />
                  <span
                    className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.15s" }}
                  />
                  <span
                    className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.3s" }}
                  />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* ── Error banner ── */}
      {error && (
        <div className="mx-4 sm:mx-8 mb-2 px-4 py-2.5 bg-red-50 border border-red-200 rounded-xl flex items-center justify-between">
          <p className="text-sm text-red-600">{error}</p>
          <button
            onClick={() => setError(null)}
            className="text-red-400 hover:text-red-600 cursor-pointer ml-3"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}

      {/* ── Record button ── */}
      <div className="flex-shrink-0 pb-8 pt-4 flex flex-col items-center bg-gradient-to-t from-white via-white/90 to-transparent">
        <RecordButton
          isRecording={isRecording}
          isLoading={isLoading}
          currentEmotion={currentEmotion}
          onToggle={handleToggleRecording}
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