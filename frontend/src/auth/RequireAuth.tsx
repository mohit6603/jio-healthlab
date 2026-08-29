import { ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { useAuth } from "./AuthContext";

/**
 * Route guard.
 *
 * Blocks rendering until the initial session-restore attempt has finished,
 * so a page reload does not flash the login screen for an authenticated user.
 */
export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, initialising } = useAuth();
  const location = useLocation();

  if (initialising) {
    return (
      <div className="app-booting" role="status">
        <Loader2 size={20} className="spin" />
        <span>Restoring your session…</span>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  return <>{children}</>;
}

/** Hides a page the user's role cannot reach, rather than 403-ing on load. */
export function RequirePermission({
  permission,
  children
}: {
  permission: string;
  children: ReactNode;
}) {
  const { can } = useAuth();

  if (!can(permission)) {
    return (
      <div className="page">
        <div className="analytics-error" role="alert">
          <div>
            <strong>Your role does not have access to this page.</strong>
            <span className="assistant-error-code">{permission}</span>
          </div>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
