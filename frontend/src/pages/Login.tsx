import { FormEvent, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { AlertTriangle, FlaskConical, Loader2, LogIn } from "lucide-react";
import { ApiError } from "../api";
import { useAuth } from "../auth/AuthContext";

function Login() {
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const from = (location.state as { from?: string } | null)?.from ?? "/";

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;

    setBusy(true);
    setError("");
    try {
      await signIn(email, password);
      navigate(from, { replace: true });
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.message
          : "Sign-in failed. Please try again."
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-shell">
      <form className="login-card" onSubmit={handleSubmit}>
        <div className="login-brand">
          <FlaskConical size={24} />
          <span>JIO HealthLab</span>
        </div>
        <h1>Sign in</h1>
        <p className="login-intro">
          Laboratory operations and diagnostics platform.
        </p>

        {error && (
          <div className="login-error" role="alert">
            <AlertTriangle size={16} />
            <span>{error}</span>
          </div>
        )}

        <label className="field">
          <span>Email</span>
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="username"
            required
            autoFocus
          />
        </label>

        <label className="field">
          <span>Password</span>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            required
          />
        </label>

        <button type="submit" className="primary-action login-submit" disabled={busy}>
          {busy ? <Loader2 size={18} className="spin" /> : <LogIn size={18} />}
          {busy ? "Signing in" : "Sign in"}
        </button>
      </form>
    </div>
  );
}

export default Login;
