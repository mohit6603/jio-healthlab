import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import Login from "./Login";
import { ApiError } from "../api";

const signIn = vi.fn();
vi.mock("../auth/AuthContext", () => ({
  useAuth: () => ({ signIn })
}));

function renderLogin(from?: string) {
  return render(
    <MemoryRouter
      initialEntries={[{ pathname: "/login", state: from ? { from } : undefined }]}
    >
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<span>dashboard</span>} />
        <Route path="/analytics" element={<span>analytics</span>} />
      </Routes>
    </MemoryRouter>
  );
}

beforeEach(() => signIn.mockReset());

describe("Login", () => {
  it("renders the sign-in form", () => {
    renderLogin();

    expect(screen.getByLabelText("Email")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("uses a password input so the value is masked", () => {
    renderLogin();

    expect(screen.getByLabelText("Password")).toHaveAttribute("type", "password");
  });

  it("sets autocomplete hints so password managers work", () => {
    renderLogin();

    expect(screen.getByLabelText("Email")).toHaveAttribute("autocomplete", "username");
    expect(screen.getByLabelText("Password")).toHaveAttribute(
      "autocomplete",
      "current-password"
    );
  });

  it("submits the credentials", async () => {
    signIn.mockResolvedValue(undefined);
    const user = userEvent.setup();
    renderLogin();

    await user.type(screen.getByLabelText("Email"), "tech@test.example.com");
    await user.type(screen.getByLabelText("Password"), "TestPassword!2026");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() =>
      expect(signIn).toHaveBeenCalledWith("tech@test.example.com", "TestPassword!2026")
    );
  });

  it("navigates to the dashboard on success", async () => {
    signIn.mockResolvedValue(undefined);
    const user = userEvent.setup();
    renderLogin();

    await user.type(screen.getByLabelText("Email"), "a@test.example.com");
    await user.type(screen.getByLabelText("Password"), "pw");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByText("dashboard")).toBeInTheDocument();
  });

  it("returns the user to the page they were blocked from", async () => {
    signIn.mockResolvedValue(undefined);
    const user = userEvent.setup();
    renderLogin("/analytics");

    await user.type(screen.getByLabelText("Email"), "a@test.example.com");
    await user.type(screen.getByLabelText("Password"), "pw");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByText("analytics")).toBeInTheDocument();
  });

  it("shows the server's message on bad credentials", async () => {
    signIn.mockRejectedValue(
      new ApiError(401, {
        code: "INVALID_CREDENTIALS",
        message: "Incorrect email or password."
      })
    );
    const user = userEvent.setup();
    renderLogin();

    await user.type(screen.getByLabelText("Email"), "a@test.example.com");
    await user.type(screen.getByLabelText("Password"), "wrong");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Incorrect email or password."
    );
  });

  it("does not reveal whether the account exists", async () => {
    signIn.mockRejectedValue(
      new ApiError(401, {
        code: "INVALID_CREDENTIALS",
        message: "Incorrect email or password."
      })
    );
    const user = userEvent.setup();
    renderLogin();

    await user.type(screen.getByLabelText("Email"), "nobody@test.example.com");
    await user.type(screen.getByLabelText("Password"), "wrong");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).not.toMatch(/no such user|not found|unknown/i);
  });

  it("explains an unreachable server", async () => {
    signIn.mockRejectedValue(
      new ApiError(0, {
        code: "NETWORK_ERROR",
        message: "Could not reach the server. Check your connection and retry."
      })
    );
    const user = userEvent.setup();
    renderLogin();

    await user.type(screen.getByLabelText("Email"), "a@test.example.com");
    await user.type(screen.getByLabelText("Password"), "pw");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/could not reach/i);
  });

  it("disables the button while signing in", async () => {
    let resolve!: () => void;
    signIn.mockReturnValue(new Promise<void>((r) => (resolve = r)));
    const user = userEvent.setup();
    renderLogin();

    await user.type(screen.getByLabelText("Email"), "a@test.example.com");
    await user.type(screen.getByLabelText("Password"), "pw");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByRole("button", { name: /signing in/i })).toBeDisabled();
    resolve();
  });

  it("clears a previous error on retry", async () => {
    signIn.mockRejectedValueOnce(
      new ApiError(401, { code: "INVALID_CREDENTIALS", message: "Incorrect." })
    );
    signIn.mockResolvedValueOnce(undefined);
    const user = userEvent.setup();
    renderLogin();

    await user.type(screen.getByLabelText("Email"), "a@test.example.com");
    await user.type(screen.getByLabelText("Password"), "wrong");
    await user.click(screen.getByRole("button", { name: /sign in/i }));
    await screen.findByRole("alert");

    await user.click(screen.getByRole("button", { name: /sign in/i }));

    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
  });
});
