import ChatClient from "@/components/chat/ChatClient";
import Navbar from "@/components/ui/Navbar";
import { getServerSession } from "@/lib/server-auth";
import { redirect } from "next/navigation";

export default async function Chat() {
  const session = await getServerSession();

  if (!session) {
    redirect("/login");
  }

  return (
    <div className="h-screen flex flex-col bg-white">
      <Navbar user={session.user} showClose />

      <main className="flex-1 overflow-hidden">
        <div className="h-full">
          <ChatClient />
        </div>
      </main>
    </div>
  );
}