import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./AuthContext";
import { RequireAuth, RequirePermission } from "./RequireAuth";
import { ApiError } from "../api";
import type { AuthUser, MeResponse, TokenResponse } from "../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return {
    ...actual,
    login: vi.fn(),
    logout: vi.fn(),
    getMe: vi.fn(),
    refreshSessionFromCookie: vi.fn(),
    setAccessToken: vi.fn(),
    setSessionLostHandler: vi.fn()
  };
});

const api = await import("../api");
const mockLogin = vi.mocked(api.login);
const mockLogout = vi.mocked(api.logout);
const mockGetMe = vi.mocked(api.getMe);
const mockRefresh = vi.mocked(api.refreshSessionFromCookie);

const USER: AuthUser = {
  id: 1,
  email: "tech@test.example.com",
  full_name: "Ravi Technician",
  role: "LAB_TECH",
  is_active: true,
  last_login_at: null
};

function session(): TokenResponse {
  return {
    access_token: "access-token",
    refresh_token: "refresh-token",
    token_type: "bearer",
    expires_in: 900,
    expires_at: "2026-08-28T12:00:00+00:00",
    user: USER
  };
}

function me(permissions: string[]): MeResponse {
  return { user: USER, permissions };
}

function Probe() {
  const { user, can, signIn, signOut, initialising } = useAuth();
  if (initialising) return <span>initialising</span>;
  return (
    <div>
      <span data-testid="user">{user ? user.full_name : "anonymous"}</span>
      <span data-testid="role">{user?.role ?? "-"}</span>
      <span data-testid="analytics">{can("ai:risk_analytics") ? "yes" : "no"}</span>
      <span data-testid="admin">{can("admin:users") ? "yes" : "no"}</span>
      <button onClick={() => void signIn("tech@test.example.com", "pw")}>sign in</button>
      <button onClick={() => void signOut()}>sign out</button>
    </div>
  );
}

function renderProbe() {
  return render(
    <MemoryRouter>
      <AuthProvider>
        <Probe />
      </AuthProvider>
    </MemoryRouter>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  // No cookie: the server rejects the refresh, which is "not signed in".
  mockRefresh.mockRejectedValue(
    new ApiError(401, { code: "INVALID_REFRESH_TOKEN", message: "no session" })
  );
});

describe("AuthProvider", () => {
  it("starts anonymous when the server reports no session", async () => {
    renderProbe();

    expect(await screen.findByTestId("user")).toHaveTextContent("anonymous");
  });

  it("signs in and loads permissions", async () => {
    mockLogin.mockResolvedValue(session());
    mockGetMe.mockResolvedValue(me(["reports:read", "ai:risk_analytics"]));
    const user = userEvent.setup();
    renderProbe();

    await user.click(await screen.findByText("sign in"));

    await waitFor(() =>
      expect(screen.getByTestId("user")).toHaveTextContent("Ravi Technician")
    );
    expect(screen.getByTestId("role")).toHaveTextContent("LAB_TECH");
    expect(screen.getByTestId("analytics")).toHaveTextContent("yes");
    expect(screen.getByTestId("admin")).toHaveTextContent("no");
  });

  it("restores a session from the httpOnly cookie", async () => {
    mockRefresh.mockResolvedValue(session());
    mockGetMe.mockResolvedValue(me(["reports:read"]));

    renderProbe();

    await waitFor(() =>
      expect(screen.getByTestId("user")).toHaveTextContent("Ravi Technician")
    );
    // No token is passed: the browser holds it where this code cannot read it.
    expect(mockRefresh).toHaveBeenCalledWith();
  });

  it("stays anonymous when the cookie is expired or revoked", async () => {
    mockRefresh.mockRejectedValue(
      new ApiError(401, { code: "INVALID_REFRESH_TOKEN", message: "no" })
    );

    renderProbe();

    expect(await screen.findByTestId("user")).toHaveTextContent("anonymous");
  });

  it("signs out and clears permissions", async () => {
    mockLogin.mockResolvedValue(session());
    mockGetMe.mockResolvedValue(me(["reports:read", "ai:risk_analytics"]));
    mockLogout.mockResolvedValue({ sessions_ended: 1 });
    const user = userEvent.setup();
    renderProbe();

    await user.click(await screen.findByText("sign in"));
    await waitFor(() =>
      expect(screen.getByTestId("user")).toHaveTextContent("Ravi Technician")
    );

    await user.click(screen.getByText("sign out"));

    await waitFor(() =>
      expect(screen.getByTestId("user")).toHaveTextContent("anonymous")
    );
    expect(screen.getByTestId("analytics")).toHaveTextContent("no");
  });

  it("signs out locally even when the server call fails", async () => {
    mockLogin.mockResolvedValue(session());
    mockGetMe.mockResolvedValue(me(["reports:read"]));
    mockLogout.mockRejectedValue(new ApiError(0, { code: "NETWORK_ERROR", message: "x" }));
    const user = userEvent.setup();
    renderProbe();

    await user.click(await screen.findByText("sign in"));
    await waitFor(() =>
      expect(screen.getByTestId("user")).toHaveTextContent("Ravi Technician")
    );

    await user.click(screen.getByText("sign out"));

    await waitFor(() =>
      expect(screen.getByTestId("user")).toHaveTextContent("anonymous")
    );
  });
});

describe("RequireAuth", () => {
  function renderGuarded() {
    return render(
      <MemoryRouter initialEntries={["/private"]}>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<span>login page</span>} />
            <Route
              path="/private"
              element={
                <RequireAuth>
                  <span>secret</span>
                </RequireAuth>
              }
            />
          </Routes>
        </AuthProvider>
      </MemoryRouter>
    );
  }

  it("redirects an anonymous visitor to login", async () => {
    renderGuarded();

    expect(await screen.findByText("login page")).toBeInTheDocument();
  });

  it("shows a restoring state rather than flashing the login page", async () => {
    mockRefresh.mockReturnValue(new Promise(() => {}));

    renderGuarded();

    expect(await screen.findByRole("status")).toHaveTextContent(/restoring your session/i);
    expect(screen.queryByText("login page")).not.toBeInTheDocument();
  });

  it("renders the page once a session is restored", async () => {
    mockRefresh.mockResolvedValue(session());
    mockGetMe.mockResolvedValue(me(["reports:read"]));

    renderGuarded();

    expect(await screen.findByText("secret")).toBeInTheDocument();
  });
});

describe("RequirePermission", () => {
  function renderPermissioned(permissions: string[]) {
    mockRefresh.mockResolvedValue(session());
    mockGetMe.mockResolvedValue(me(permissions));

    return render(
      <MemoryRouter>
        <AuthProvider>
          <RequirePermission permission="ai:risk_analytics">
            <span>analytics page</span>
          </RequirePermission>
        </AuthProvider>
      </MemoryRouter>
    );
  }

  it("renders when the permission is held", async () => {
    renderPermissioned(["ai:risk_analytics"]);

    expect(await screen.findByText("analytics page")).toBeInTheDocument();
  });

  it("explains the block when it is not", async () => {
    renderPermissioned(["reports:read"]);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /does not have access/i
    );
    expect(screen.queryByText("analytics page")).not.toBeInTheDocument();
  });
});
