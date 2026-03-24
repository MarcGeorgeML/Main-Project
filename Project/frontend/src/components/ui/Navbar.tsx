"use client";

import { useRouter } from "next/navigation";
import Image from "next/image";
import LogoutButton from "@/components/ui/button/LogoutButton";
import { User } from "@/types/auth";

type NavbarProps = {
  showClose?: boolean;
  user: User;
};

export default function Navbar({ showClose = false, user }: NavbarProps) {
  const router = useRouter();

  return (
    <nav className="bg-white border-b border-purple-100 flex-shrink-0">
      <div className="mx-auto px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left */}
          <div className="flex items-center gap-4">
            {showClose && (
              <button
                onClick={() => router.push("/home")}
                className="p-2 rounded-full hover:bg-purple-50 transition-colors cursor-pointer"
                aria-label="Go to home"
              >
                <svg
                  className="w-5 h-5 text-gray-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            )}

            {/* Logo from /public/logo.png */}
            <div className="flex items-center gap-2.5">
              <Image
                src="/logo2.png"
                alt="SentiCore logo"
                width={32}
                height={32}
                className="object-contain"
              />
              <h1 className="text-xl font-bold text-purple-600">SentiCore</h1>
            </div>
          </div>

          {/* Right */}
          <div className="flex items-center gap-4">
            <div className="text-right hidden sm:block">
              <p className="font-semibold text-gray-800 text-sm">
                {user.first_name} {user.last_name}
              </p>
              <p className="text-xs text-gray-500">{user.email}</p>
            </div>
            <LogoutButton />
          </div>
        </div>
      </div>
    </nav>
  );
}




