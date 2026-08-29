import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AIAssistant from "./AIAssistant";
import type { StreamHandlers, StreamSummary } from "../api";
import type { Citation } from "../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, streamAssistant: vi.fn() };
});

const { streamAssistant } = await import("../api");
const mockStream = vi.mocked(streamAssistant);

const SOURCES: Citation[] = [
  {
    title: "Complete Blood Count (CBC) Guide",
    source: "cbc.md",
    chunk_id: "lab_tests/cbc::0",
    score: 0.664,
    section: "What a CBC measures",
    category: "lab_tests"
  }
];

function summary(overrides: Partial<StreamSummary> = {}): StreamSummary {
  return {
    grounded: true,
    finish_reason: "stop",
    model: "Qwen/Qwen2.5-0.5B-Instruct",
    provider: "local",
    retrieval_count: 1,
    drop_sources: false,
    disclaimer: "AI-generated informational explanation. Not a medical diagnosis.",
    timings: { retrieval_ms: 12, generation_ms: 900, total_ms: 912 },
    ...overrides
  };
}

/** Captures the handlers so a test can drive the stream frame by frame. */
let captured: StreamHandlers | null = null;
let aborted = false;

function captureStream() {
  mockStream.mockImplementation((_question, handlers) => {
    captured = handlers;
    return () => {
      aborted = true;
    };
  });
}

/** Drives a complete, successful stream. */
function playHappyPath(text = "A CBC measures red cells, white cells and platelets.") {
  mockStream.mockImplementation((_question, handlers) => {
    handlers.onSources?.(SOURCES);
    for (const word of text.split(" ")) handlers.onToken?.(`${word} `);
    handlers.onDone?.(summary());
    return () => {};
  });
}

beforeEach(() => {
  mockStream.mockReset();
  captured = null;
  aborted = false;
});

describe("AIAssistant", () => {
  it("shows sample prompts when the conversation is empty", () => {
    render(<AIAssistant />);

    expect(screen.getByText("What does a CBC test measure?")).toBeInTheDocument();
    expect(
      screen.getByText("Explain a lipid profile in simple terms.")
    ).toBeInTheDocument();
  });

  it("always shows the healthcare boundary notice", () => {
    render(<AIAssistant />);

    const note = screen.getByRole("note");
    expect(note).toHaveTextContent(/not a medical diagnosis/i);
    expect(note).toHaveTextContent(/does not interpret personal results/i);
  });

  it("sends the typed question to the stream", async () => {
    playHappyPath();
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.type(screen.getByLabelText(/your question/i), "What is a CBC?");
    await user.click(screen.getByRole("button", { name: /ask/i }));

    await waitFor(() =>
      expect(mockStream).toHaveBeenCalledWith(
        "What is a CBC?",
        expect.objectContaining({ onToken: expect.any(Function) })
      )
    );
  });

  it("clicking a sample prompt asks it", async () => {
    playHappyPath();
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    await waitFor(() =>
      expect(mockStream).toHaveBeenCalledWith(
        "What does a CBC test measure?",
        expect.objectContaining({ onSources: expect.any(Function) })
      )
    );
  });

  // ── streaming behaviour ────────────────────────────────────────
  it("shows a waiting indicator until the first token", async () => {
    captureStream();
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(await screen.findByRole("status")).toHaveTextContent(
      /searching the knowledge base/i
    );
  });

  it("replaces the waiting indicator once tokens arrive", async () => {
    captureStream();
    const user = userEvent.setup();
    render(<AIAssistant />);
    await user.click(screen.getByText("What does a CBC test measure?"));
    await screen.findByRole("status");

    act(() => captured?.onToken?.("A CBC "));

    await waitFor(() =>
      expect(
        screen.queryByText(/searching the knowledge base/i)
      ).not.toBeInTheDocument()
    );
    expect(screen.getByText(/A CBC/)).toBeInTheDocument();
  });

  it("renders tokens incrementally as they arrive", async () => {
    captureStream();
    const user = userEvent.setup();
    render(<AIAssistant />);
    await user.click(screen.getByText("What does a CBC test measure?"));

    act(() => captured?.onToken?.("A CBC "));
    expect(await screen.findByText(/A CBC/)).toBeInTheDocument();

    act(() => captured?.onToken?.("measures blood cells."));

    await waitFor(() =>
      expect(screen.getByText(/A CBC measures blood cells\./)).toBeInTheDocument()
    );
  });

  it("shows grounding before the answer finishes", async () => {
    captureStream();
    const user = userEvent.setup();
    render(<AIAssistant />);
    await user.click(screen.getByText("What does a CBC test measure?"));

    act(() => {
      captured?.onSources?.(SOURCES);
      captured?.onToken?.("A CBC ");
    });

    expect(await screen.findByText(/grounding in 1 source/i)).toBeInTheDocument();
  });

  it("renders the finished answer with citations", async () => {
    playHappyPath();
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(
      await screen.findByText(/A CBC measures red cells, white cells and platelets/)
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /1 source/i })
    ).toBeInTheDocument();
  });

  it("shows the disclaimer with every answer", async () => {
    playHappyPath();
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(
      await screen.findByText(/AI-generated informational explanation/)
    ).toBeInTheDocument();
  });

  it("expands sources on click", async () => {
    playHappyPath();
    const user = userEvent.setup();
    render(<AIAssistant />);
    await user.click(screen.getByText("What does a CBC test measure?"));

    const toggle = await screen.findByRole("button", { name: /1 source/i });
    expect(
      screen.queryByText("Complete Blood Count (CBC) Guide")
    ).not.toBeInTheDocument();

    await user.click(toggle);

    expect(screen.getByText("Complete Blood Count (CBC) Guide")).toBeInTheDocument();
    expect(screen.getByText("cbc.md · What a CBC measures")).toBeInTheDocument();
    expect(screen.getByText("0.664")).toBeInTheDocument();
  });

  it("flags an ungrounded answer", async () => {
    mockStream.mockImplementation((_q, handlers) => {
      handlers.onToken?.("I don't have enough information.");
      handlers.onDone?.(summary({ grounded: false, retrieval_count: 0 }));
      return () => {};
    });
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(
      await screen.findByText(/not backed by the knowledge base/i)
    ).toBeInTheDocument();
  });

  // ── safety ─────────────────────────────────────────────────────
  it("replaces a suppressed answer and drops its citations", async () => {
    mockStream.mockImplementation((_q, handlers) => {
      handlers.onSources?.(SOURCES);
      handlers.onToken?.("Your haemoglobin indicates mild anaemia.");
      handlers.onReplace?.("I can't interpret results or advise on treatment.");
      handlers.onDone?.(summary({ grounded: false, drop_sources: true }));
      return () => {};
    });
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(
      await screen.findByText(/can't interpret results or advise on treatment/i)
    ).toBeInTheDocument();
    // The suppressed text must be gone, and its citations with it.
    expect(screen.queryByText(/indicates mild anaemia/)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /source/i })
    ).not.toBeInTheDocument();
  });

  // ── errors ─────────────────────────────────────────────────────
  it("shows actionable guidance when the AI service is unavailable", async () => {
    mockStream.mockImplementation((_q, handlers) => {
      handlers.onError?.({
        code: "AI_SERVICE_UNAVAILABLE",
        message: "The AI assistant is currently unavailable."
      });
      return () => {};
    });
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/assistant is offline/i);
    expect(alert).toHaveTextContent(/reports and the dashboard still work/i);
    expect(alert).toHaveTextContent("AI_SERVICE_UNAVAILABLE");
  });

  it("distinguishes generation being switched off from a failure", async () => {
    mockStream.mockImplementation((_q, handlers) => {
      handlers.onError?.({ code: "GENERATION_DISABLED", message: "off" });
      return () => {};
    });
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(await screen.findByRole("alert")).toHaveTextContent(/switched off/i);
  });

  it("reports a dropped connection mid-stream", async () => {
    captureStream();
    const user = userEvent.setup();
    render(<AIAssistant />);
    await user.click(screen.getByText("What does a CBC test measure?"));

    act(() => captured?.onToken?.("A CBC "));
    act(() =>
      captured?.onError?.({
        code: "STREAM_INTERRUPTED",
        message: "The connection dropped before the answer finished."
      })
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "STREAM_INTERRUPTED"
    );
  });

  it("keeps the conversation usable after an error", async () => {
    mockStream.mockImplementationOnce((_q, handlers) => {
      handlers.onError?.({ code: "AI_SERVICE_UNAVAILABLE", message: "down" });
      return () => {};
    });
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));
    await screen.findByRole("alert");

    playHappyPath();
    await user.type(screen.getByLabelText(/your question/i), "retry");
    await user.click(screen.getByRole("button", { name: /^ask$/i }));

    expect(await screen.findByText(/A CBC measures/)).toBeInTheDocument();
  });

  // ── controls ───────────────────────────────────────────────────
  it("clears the conversation", async () => {
    playHappyPath();
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));
    await screen.findByText(/A CBC measures/);

    await user.click(screen.getByRole("button", { name: /clear conversation/i }));

    expect(screen.queryByText(/A CBC measures/)).not.toBeInTheDocument();
    expect(screen.getByText("What is a thyroid function test?")).toBeInTheDocument();
  });

  it("disables clear when there is nothing to clear", () => {
    render(<AIAssistant />);

    expect(screen.getByRole("button", { name: /clear conversation/i })).toBeDisabled();
  });

  it("does not submit an empty question", async () => {
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.type(screen.getByLabelText(/your question/i), "   ");

    expect(screen.getByRole("button", { name: /ask/i })).toBeDisabled();
    expect(mockStream).not.toHaveBeenCalled();
  });

  it("echoes the user question in the thread", async () => {
    playHappyPath();
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.type(screen.getByLabelText(/your question/i), "Why is my report late?");
    await user.click(screen.getByRole("button", { name: /^ask$/i }));

    const thread = screen.getByRole("log");
    expect(within(thread).getByText("Why is my report late?")).toBeInTheDocument();
  });

  it("aborts an in-flight stream when the page unmounts", async () => {
    captureStream();
    const user = userEvent.setup();
    const { unmount } = render(<AIAssistant />);
    await user.click(screen.getByText("What does a CBC test measure?"));

    unmount();

    expect(aborted).toBe(true);
  });
});
