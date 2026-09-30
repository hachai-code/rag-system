"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { clearToken, getToken } from "@/lib/auth";

const PUBLIC_ROUTES = ["/login", "/register"];

// Client-side gate: the API is the real access control (it 401s without a valid token);
// this only decides what UI to show and redirects so people don't land on a dead page.
export function AuthGate({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const isPublic = PUBLIC_ROUTES.includes(pathname);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const authed = !!getToken();
    if (!authed && !isPublic) {
      router.replace("/login");
    } else if (authed && isPublic) {
      router.replace("/");
    } else {
      setReady(true);
    }
  }, [isPublic, router]);

  if (!ready) return null; // don't flash protected content before the redirect lands

  function logout() {
    clearToken();
    router.replace("/login");
  }

  return (
    <>
      {!isPublic && (
        <header className="flex justify-end border-b border-gray-100 px-4 py-2">
          <button onClick={logout} className="text-sm text-gray-500 hover:text-gray-800">
            Log out
          </button>
        </header>
      )}
      {children}
    </>
  );
}
