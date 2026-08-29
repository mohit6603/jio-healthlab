import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { Bot, FlaskConical, LayoutDashboard, LogOut, TrendingUp } from "lucide-react";
import { useAuth } from "./auth/AuthContext";
import { getAIHealth } from "./api";
import type { AIHealthResponse } from "./types";

/**
 * Application shell: navigation, the AI availability banner, and the routed
 * page.
 *
 * The banner exists because AI is an optional dependency. When it is down the
 * reports side of the product keeps working, and the user should be told that
 * plainly rather than discovering it by clicking into a broken page.
 */
function App() {
  const { user, can, signOut } = useAuth();
  const [aiHealth, setAiHealth] = useState<AIHealthResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    getAIHealth()
      .then((health) => {
        if (!cancelled) setAiHealth(health);
      })
      .catch(() => {
        // The health endpoint reports rather than throws; if even that fails,
        // stay quiet rather than showing a scary banner for a transient blip.
        if (!cancelled) setAiHealth(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const aiDown = aiHealth !== null && !aiHealth.reachable;
  const aiDegraded = aiHealth?.reachable === true && aiHealth.status === "degraded";

  return (
    <div className="shell">
      <header className="app-nav">
        <div className="app-nav-brand">
          <FlaskConical size={20} />
          <span>JIO HealthLab</span>
        </div>
        <nav aria-label="Main">
          <NavLink to="/" end className={navClass}>
            <LayoutDashboard size={17} />
            <span>Dashboard</span>
          </NavLink>
          <NavLink to="/assistant" className={navClass}>
            <Bot size={17} />
            <span>AI Assistant</span>
          </NavLink>
          {can("ai:risk_analytics") && (
            <NavLink to="/analytics" className={navClass}>
              <TrendingUp size={17} />
              <span>AI Analytics</span>
            </NavLink>
          )}
        </nav>
        <div className="app-nav-user">
          <span className="app-nav-name">{user?.full_name}</span>
          <span className="tag">{user?.role}</span>
          <button
            type="button"
            className="icon-action"
            onClick={() => void signOut()}
            title="Sign out"
            aria-label="Sign out"
          >
            <LogOut size={17} />
          </button>
        </div>
      </header>

      {aiDown && (
        <div className="service-banner" role="status">
          The AI assistant is currently unavailable. Reports and the dashboard
          are unaffected.
        </div>
      )}
      {aiDegraded && (
        <div className="service-banner subtle" role="status">
          The AI assistant is running in reduced capability.{" "}
          {aiHealth?.components.find((item) => item.state !== "ok")?.detail ?? ""}
        </div>
      )}

      <Outlet />
    </div>
  );
}

function navClass({ isActive }: { isActive: boolean }) {
  return isActive ? "app-nav-link active" : "app-nav-link";
}

export default App;
