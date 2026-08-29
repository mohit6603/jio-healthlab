import {
  createContext,
  ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState
} from "react";
import {
  ApiError,
  getMe,
  login as loginRequest,
  logout as logoutRequest,
  refreshSessionFromCookie,
  setAccessToken,
  setSessionLostHandler
} from "../api";
import type { AuthUser } from "../types";

interface AuthState {
  user: AuthUser | null;
  permissions: Set<string>;
  /** True until the initial silent-refresh attempt has finished. */
  initialising: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  can: (permission: string) => boolean;
}

const AuthContext = createContext<AuthState | null>(null);

/**
 * Session state for the app.
 *
 * On mount it tries to restore a session from the stored refresh token, so a
 * page reload does not sign the user out even though the access token lives
 * only in memory.
 *
 * `can()` is a convenience for hiding actions the user cannot perform. It is
 * not a security boundary -- the server enforces the same permission list
 * independently on every request.
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [permissions, setPermissions] = useState<Set<string>>(new Set());
  const [initialising, setInitialising] = useState(true);

  const clear = useCallback(() => {
    setAccessToken(null);
    setUser(null);
    setPermissions(new Set());
  }, []);

  useEffect(() => {
    setSessionLostHandler(clear);
    return () => setSessionLostHandler(null);
  }, [clear]);

  useEffect(() => {
    let cancelled = false;

    async function restore() {
      // The refresh token is an httpOnly cookie, so this code cannot check
      // whether a session exists -- it asks the server. A 401 simply means
      // "not signed in", which is the common case on a first visit.
      try {
        const session = await refreshSessionFromCookie();
        if (cancelled) return;
        setAccessToken(session.access_token);
        const me = await getMe();
        if (cancelled) return;
        setUser(me.user);
        setPermissions(new Set(me.permissions));
      } catch {
        // No cookie, expired, or revoked. Stay signed out quietly.
        if (!cancelled) clear();
      } finally {
        if (!cancelled) setInitialising(false);
      }
    }

    void restore();
    return () => {
      cancelled = true;
    };
  }, [clear]);

  const signIn = useCallback(async (email: string, password: string) => {
    const session = await loginRequest(email, password);
    // The server also set the httpOnly refresh cookie on this response.
    setAccessToken(session.access_token);
    setUser(session.user);
    try {
      const me = await getMe();
      setPermissions(new Set(me.permissions));
    } catch (caught) {
      if (!(caught instanceof ApiError)) throw caught;
      setPermissions(new Set());
    }
  }, []);

  const signOut = useCallback(async () => {
    try {
      await logoutRequest();
    } catch {
      // Signing out locally matters more than the server acknowledging it.
    } finally {
      clear();
    }
  }, [clear]);

  const value = useMemo<AuthState>(
    () => ({
      user,
      permissions,
      initialising,
      signIn,
      signOut,
      can: (permission: string) => permissions.has(permission)
    }),
    [user, permissions, initialising, signIn, signOut]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside <AuthProvider>");
  }
  return context;
}
