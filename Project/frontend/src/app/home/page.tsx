import { redirect } from "next/navigation";
import { getServerSession } from "@/lib/server-auth";
import Navbar from "@/components/ui/Navbar";
import ChatClient from "@/components/chat/ChatClient";

export default async function Home() {
  const session = await getServerSession();

  if (!session) redirect("/login");

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-gray-50 via-blue-50/20 to-purple-50/20">
      <Navbar user={session.user} />
      <main className="flex-1 overflow-hidden">
              <div className="h-full">
                <ChatClient />
              </div>
            </main>
    </div>
  );
}