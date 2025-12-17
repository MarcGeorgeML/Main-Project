"use client";

import { useState, useEffect, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import AudioPlayer, { RHAP_UI } from "react-h5-audio-player";
import "react-h5-audio-player/lib/styles.css";
import {
  getConversationMessages,
  sendTextMessage,
  sendAudioMessage,
  updateConversationTitle,
  getAudioUrl,
} from "@/lib/api";
import { Message, MessageResponse } from "@/types/auth";
import { AxiosError } from "axios";

interface DisplayMessage extends Omit<Message, "id"> {
  id: string;
  transcription?: string;
}

export default function ChatClient() {
  const params = useParams();
  const router = useRouter();
  const chatId = params?.id as string;

  const [messages, setMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState<string>("");
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isFetchingMessages, setIsFetchingMessages] = useState<boolean>(true);
  const [conversationTitle, setConversationTitle] = useState<string>("Chat");
  const [isEditingTitle, setIsEditingTitle] = useState<boolean>(false);
  const [editedTitle, setEditedTitle] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const fetchMessages = async (): Promise<void> => {
      setIsFetchingMessages(true);
      try {
        const response = await getConversationMessages(chatId);

        const normalizedMessages = response.data.messages.map(
          (msg: Message) => {
            if (msg.message_type === "AUDIO") {
              return {
                ...msg,
                message: getAudioUrl(msg.message),
                transcription: msg.transcription,
              };
            }
            return msg;
          }
        );

        setMessages(normalizedMessages);
        setError(null);
      } catch (err) {
        const error = err as AxiosError;
        console.error("Error fetching messages:", error);
        setError("Failed to load messages");
        router.push("/conversations");
      } finally {
        setIsFetchingMessages(false);
      }
    };
    if (chatId) {
      fetchMessages();
    }
  }, [chatId, router]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = (): void => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  };

  const handleUpdateTitle = async (): Promise<void> => {
    if (!editedTitle.trim()) {
      setIsEditingTitle(false);
      return;
    }

    try {
      await updateConversationTitle(chatId, editedTitle);
      setConversationTitle(editedTitle);
      setIsEditingTitle(false);
      setError(null);
    } catch (err) {
      const error = err as AxiosError;
      console.error("Error updating title:", error);
      setError("Failed to update title");
    }
  };

  const convertMessageResponseToDisplay = (
    msg: MessageResponse,
    sender: "USER" | "AI",
    messageType: "TEXT" | "AUDIO"
  ): DisplayMessage => {
    return {
      id: msg.id,
      sender: sender,
      message_type: messageType,
      message: msg.message || msg.audio_filename || "",
      created_at: msg.created_at,
    };
  };

  const handleSend = async (): Promise<void> => {
    if (!input.trim() || isLoading) return;

    const tempUserMsg: DisplayMessage = {
      id: "temp-" + Date.now(),
      sender: "USER",
      message_type: "TEXT",
      message: input,
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, tempUserMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await sendTextMessage(chatId, tempUserMsg.message);

      const userMsg = convertMessageResponseToDisplay(
        response.data.user_message,
        "USER",
        "TEXT"
      );
      const aiMsg = convertMessageResponseToDisplay(
        response.data.ai_message,
        "AI",
        "TEXT"
      );

      setMessages((prev) => {
        const filtered = prev.filter((m) => m.id !== tempUserMsg.id);
        return [...filtered, userMsg, aiMsg];
      });

      setError(null);
    } catch (err) {
      const error = err as AxiosError;
      console.error("Error sending message:", error);

      setMessages((prev) => prev.filter((m) => m.id !== tempUserMsg.id));

      const errorMsg: DisplayMessage = {
        id: "error-" + Date.now(),
        sender: "AI",
        message_type: "TEXT",
        message: "Sorry, I couldn't process your message. Please try again.",
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMsg]);
      setError("Failed to send message");
    } finally {
      setIsLoading(false);
      requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>): void => {
    if (e.key === "Enter") handleSend();
  };

  const startRecording = async (): Promise<void> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e: BlobEvent) => {
        audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });

        const tempUserMsg: DisplayMessage = {
          id: "temp-audio-" + Date.now(),
          sender: "USER",
          message_type: "AUDIO",
          message: URL.createObjectURL(audioBlob),
          created_at: new Date().toISOString(),
        };

        setMessages((prev) => [...prev, tempUserMsg]);
        setIsLoading(true);

        try {
          const response = await sendAudioMessage(chatId, audioBlob);

          const filename =
            response.data.user_message.audio_filename ||
            response.data.user_message.message ||
            "";

          const userMsg: DisplayMessage = {
            id: response.data.user_message.id,
            sender: "USER",
            message_type: "AUDIO",
            message: getAudioUrl(filename),
            transcription: response.data.user_message.transcribed_message, // Get transcription from response
            created_at: response.data.user_message.created_at,
          };

          const aiMsg = convertMessageResponseToDisplay(
            response.data.ai_message,
            "AI",
            "TEXT"
          );

          setMessages((prev) => {
            const filtered = prev.filter((m) => m.id !== tempUserMsg.id);
            return [...filtered, userMsg, aiMsg];
          });

          setError(null);
        } catch (err) {
          const error = err as AxiosError;
          console.error("Error sending audio:", error);
          setMessages((prev) => prev.filter((m) => m.id !== tempUserMsg.id));
          setError("Failed to send audio message");
        } finally {
          setIsLoading(false);
        }

        stream.getTracks().forEach((t) => t.stop());
        requestAnimationFrame(() => {
          inputRef.current?.focus();
        });
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      const error = err as Error;
      console.error("Error accessing microphone:", error);
      alert("Could not access microphone. Please check permissions.");
    }
  };

  const stopRecording = (): void => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);
  };

  const handleMicClick = (): void => {
    if (isLoading) return;
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const handleTitleClick = (): void => {
    setEditedTitle(conversationTitle);
    setIsEditingTitle(true);
  };

  const handleTitleInputChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ): void => {
    setEditedTitle(e.target.value);
  };

  const handleTitleKeyDown = (
    e: React.KeyboardEvent<HTMLInputElement>
  ): void => {
    if (e.key === "Enter") {
      handleUpdateTitle();
    }
  };

  const handleBackClick = (): void => {
    router.push("/conversations");
  };

  if (isFetchingMessages) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  return (
    <>
      <div className="lg:col-span-3 h-full flex flex-col min-h-0">
        <div className="flex-shrink-0 px-6 py-4 border-b border-gray-100 bg-white">
          <div className="flex items-center justify-between">
            <button
              onClick={handleBackClick}
              className="lg:hidden mr-3 p-2 hover:bg-gray-100 rounded-lg"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 19l-7-7 7-7"
                />
              </svg>
            </button>
            {isEditingTitle ? (
              <input
                type="text"
                value={editedTitle}
                onChange={handleTitleInputChange}
                onBlur={handleUpdateTitle}
                onKeyDown={handleTitleKeyDown}
                className="flex-1 px-2 py-1 border rounded"
                autoFocus
              />
            ) : (
              <h1
                className="text-xl font-semibold cursor-pointer hover:text-purple-500"
                onClick={handleTitleClick}
              >
                {conversationTitle}
              </h1>
            )}
          </div>
          {error && <div className="mt-2 text-sm text-red-600">{error}</div>}
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-6 min-h-0">
          <div className="space-y-4">
            {messages.length === 0 && (
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-purple-500 rounded-full flex-shrink-0"></div>
                <div className="flex flex-col space-y-1 max-w-lg">
                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                    <p className="text-sm text-gray-800">
                      Hi! How can I help you today?
                    </p>
                  </div>
                </div>
              </div>
            )}

            {messages.map((message, index) =>
              message.sender === "AI" ? (
                <div
                  key={message.id}
                  className="flex items-start space-x-3 opacity-0 animate-[fadeSlideIn_0.6s_ease-out_forwards]"
                  style={{ animationDelay: `${index * 0.02}s` }}
                >
                  <div className="w-8 h-8 bg-purple-500 rounded-full flex-shrink-0"></div>
                  <div className="flex flex-col space-y-1 max-w-lg">
                    <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                      <p className="text-sm text-gray-800">{message.message}</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div
                  key={message.id}
                  className="flex items-start space-x-3 justify-end opacity-0 animate-[fadeSlideIn_0.6s_ease-out_forwards]"
                  style={{ animationDelay: `${index * 0.02}s` }}
                >
                  <div className="flex flex-col space-y-1 max-w-lg items-end">
                    <div className="bg-purple-500 rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm">
                      {message.message_type === "TEXT" ? (
                        <p className="text-sm text-white">{message.message}</p>
                      ) : (
                        <div className="space-y-2">
                          <div className="audio-player-wrapper">
                            <AudioPlayer
                              src={message.message}
                              layout="horizontal-reverse"
                              showJumpControls={false}
                              customProgressBarSection={[RHAP_UI.PROGRESS_BAR]}
                              customControlsSection={[
                                RHAP_UI.MAIN_CONTROLS,
                                RHAP_UI.DURATION,
                              ]}
                              customAdditionalControls={[]}
                              customVolumeControls={[]}
                              autoPlayAfterSrcChange={false}
                              className="voice-message-player"
                            />
                          </div>

                          {message.transcription && (
                            <p className="text-xs text-purple-100 italic leading-snug">
                              {message.transcription}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                    <span className="text-xs text-gray-400 mr-2">
                      {new Date(message.created_at).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                  <div className="w-8 h-8 bg-zinc-400 rounded-full flex-shrink-0"></div>
                </div>
              )
            )}
            {isLoading && (
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-purple-500 rounded-full flex-shrink-0"></div>
                <div className="flex flex-col space-y-1 max-w-lg">
                  <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                    <div className="flex space-x-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.1s" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="flex-shrink-0 px-6 py-4 border-t border-gray-100 bg-gray-50/50">
          <div className="flex items-center space-x-3 bg-white rounded-2xl border border-gray-200 px-4 py-3 shadow-sm hover:shadow-md transition-shadow">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message here..."
              disabled={isLoading}
              className="flex-1 outline-none text-sm bg-transparent placeholder:text-gray-400 disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
              className="px-4 py-2 bg-purple-500 text-white rounded-xl font-medium hover:shadow-lg active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>

      <div className="hidden lg:flex bg-gradient-to-b from-blue-400/10 to-gray-50/30 border-l border-gray-100 p-6 items-center justify-center">
        <button
          onClick={handleMicClick}
          disabled={isLoading}
          className={`w-24 h-24 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-105 active:scale-95 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed ${
            isRecording ? "bg-red-400 animate-pulse" : "bg-purple-500"
          }`}
        >
          {isRecording ? (
            <svg
              className="w-12 h-12 text-white"
              fill="currentColor"
              viewBox="0 0 24 24"
            >
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          ) : (
            <svg
              className="w-12 h-12 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
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
      <style jsx global>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        @keyframes fadeSlideIn {
          from {
            opacity: 0;
            transform: translateY(12px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        ::-webkit-scrollbar {
          width: 8px;
        }

        ::-webkit-scrollbar-track {
          background: transparent;
        }

        ::-webkit-scrollbar-thumb {
          background: rgba(156, 163, 175, 0.3);
          border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
          background: rgba(156, 163, 175, 0.5);
        }

        .audio-player-wrapper {
          width: 200px;
          max-width: 100%;
        }

        .audio-player-wrapper .rhap_container {
          background: transparent;
          box-shadow: none;
          padding: 0;
        }

        .audio-player-wrapper .rhap_progress-section {
          width: 200px;
          margin-bottom: 4px;
        }

        .audio-player-wrapper .rhap_progress-bar {
          background-color: rgba(255, 255, 255, 0.3);
          height: 4px;
        }

        .audio-player-wrapper .rhap_progress-filled,
        .audio-player-wrapper .rhap_progress-indicator {
          background-color: white;
        }

        .audio-player-wrapper .rhap_controls-section {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .audio-player-wrapper .rhap_time {
          margin-left: 6px;
          color: white;
          font-size: 12px;
        }

        .audio-player-wrapper .rhap_main-controls button {
          color: white;
          width: 32px;
          height: 32px;
        }

        .audio-player-wrapper .rhap_main-controls-button svg {
          width: 32px;
          height: 32px;
        }
      `}</style>
    </>
  );
}