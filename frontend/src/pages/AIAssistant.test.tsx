import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AIAssistant from "./AIAssistant";
import { ApiError } from "../api";
import type { ChatResponse } from "../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, askAssistant: vi.fn() };
});

const { askAssistant } = await import("../api");
const mockAsk = vi.mocked(askAssistant);

function answer(overrides: Partial<ChatResponse> = {}): ChatResponse {
  return {
    answer: "A CBC measures red cells, white cells and platelets.",
    sources: [
      {
        title: "Complete Blood Count (CBC) Guide",
        source: "cbc.md",
        chunk_id: "lab_tests/cbc::0",
        score: 0.664,
        section: "What a CBC measures",
        category: "lab_tests"
      }
    ],
    retrieval_count: 1,
    grounded: true,
    disclaimer: "AI-generated informational explanation. Not a medical diagnosis.",
    model: "Qwen/Qwen2.5-0.5B-Instruct",
    provider: "local",
    finish_reason: "stop",
    timings: { retrieval_ms: 12, generation_ms: 900, total_ms: 912 },
    ...overrides
  };
}

beforeEach(() => {
  mockAsk.mockReset();
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

    expect(screen.getByRole("note")).toHaveTextContent(/not a medical diagnosis/i);
    expect(screen.getByRole("note")).toHaveTextContent(/does not interpret personal results/i);
  });

  it("sends the typed question to the API", async () => {
    mockAsk.mockResolvedValue(answer());
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.type(screen.getByLabelText(/your question/i), "What is a CBC?");
    await user.click(screen.getByRole("button", { name: /ask/i }));

    await waitFor(() => expect(mockAsk).toHaveBeenCalledWith("What is a CBC?"));
  });

  it("clicking a sample prompt asks it", async () => {
    mockAsk.mockResolvedValue(answer());
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    await waitFor(() =>
      expect(mockAsk).toHaveBeenCalledWith("What does a CBC test measure?")
    );
  });

  it("shows a loading state while awaiting the answer", async () => {
    let resolve!: (value: ChatResponse) => void;
    mockAsk.mockReturnValue(new Promise((r) => (resolve = r)));
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.type(screen.getByLabelText(/your question/i), "cbc");
    await user.click(screen.getByRole("button", { name: /ask/i }));

    expect(await screen.findByRole("status")).toHaveTextContent(/searching the knowledge base/i);
    expect(screen.getByRole("button", { name: /thinking/i })).toBeDisabled();

    resolve(answer());
    await waitFor(() =>
      expect(screen.queryByText(/searching the knowledge base/i)).not.toBeInTheDocument()
    );
  });

  it("renders the answer text", async () => {
    mockAsk.mockResolvedValue(answer());
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(
      await screen.findByText(/A CBC measures red cells, white cells and platelets/)
    ).toBeInTheDocument();
  });

  it("shows the disclaimer with every answer", async () => {
    mockAsk.mockResolvedValue(answer());
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(
      await screen.findByText(/AI-generated informational explanation/)
    ).toBeInTheDocument();
  });

  it("collapses sources behind a toggle and expands them on click", async () => {
    mockAsk.mockResolvedValue(answer());
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    const toggle = await screen.findByRole("button", { name: /1 source/i });
    expect(screen.queryByText("Complete Blood Count (CBC) Guide")).not.toBeInTheDocument();

    await user.click(toggle);

    expect(screen.getByText("Complete Blood Count (CBC) Guide")).toBeInTheDocument();
    expect(screen.getByText("cbc.md · What a CBC measures")).toBeInTheDocument();
    expect(screen.getByText("0.664")).toBeInTheDocument();
  });

  it("pluralises the source count", async () => {
    mockAsk.mockResolvedValue(
      answer({
        sources: [
          ...answer().sources,
          { ...answer().sources[0], chunk_id: "lab_tests/cbc::1", score: 0.55 }
        ]
      })
    );
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(await screen.findByRole("button", { name: /2 sources/i })).toBeInTheDocument();
  });

  it("flags an ungrounded answer", async () => {
    mockAsk.mockResolvedValue(
      answer({
        answer: "I don't have enough information in the knowledge base.",
        sources: [],
        retrieval_count: 0,
        grounded: false,
        finish_reason: "no_context"
      })
    );
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(
      await screen.findByText(/not backed by the knowledge base/i)
    ).toBeInTheDocument();
  });

  it("does not render a source toggle when there are no sources", async () => {
    mockAsk.mockResolvedValue(answer({ sources: [], retrieval_count: 0 }));
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    await screen.findByText(/A CBC measures/);
    expect(screen.queryByRole("button", { name: /source/i })).not.toBeInTheDocument();
  });

  it("shows actionable guidance when the AI service is unavailable", async () => {
    mockAsk.mockRejectedValue(
      new ApiError(503, {
        code: "AI_SERVICE_UNAVAILABLE",
        message: "The AI assistant is currently unavailable."
      })
    );
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/assistant is offline/i);
    expect(alert).toHaveTextContent(/reports and the dashboard still work/i);
    expect(alert).toHaveTextContent("AI_SERVICE_UNAVAILABLE");
  });

  it("distinguishes generation being switched off from a failure", async () => {
    mockAsk.mockRejectedValue(
      new ApiError(503, { code: "GENERATION_DISABLED", message: "off" })
    );
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(await screen.findByRole("alert")).toHaveTextContent(/switched off/i);
  });

  it("explains a timeout", async () => {
    mockAsk.mockRejectedValue(
      new ApiError(504, { code: "AI_SERVICE_TIMEOUT", message: "slow" })
    );
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));

    expect(await screen.findByRole("alert")).toHaveTextContent(/took too long/i);
  });

  it("keeps the conversation usable after an error", async () => {
    mockAsk.mockRejectedValueOnce(
      new ApiError(503, { code: "AI_SERVICE_UNAVAILABLE", message: "down" })
    );
    mockAsk.mockResolvedValueOnce(answer());
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.click(screen.getByText("What does a CBC test measure?"));
    await screen.findByRole("alert");

    await user.type(screen.getByLabelText(/your question/i), "retry");
    await user.click(screen.getByRole("button", { name: /^ask$/i }));

    expect(await screen.findByText(/A CBC measures/)).toBeInTheDocument();
  });

  it("clears the conversation", async () => {
    mockAsk.mockResolvedValue(answer());
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
    expect(mockAsk).not.toHaveBeenCalled();
  });

  it("echoes the user question in the thread", async () => {
    mockAsk.mockResolvedValue(answer());
    const user = userEvent.setup();
    render(<AIAssistant />);

    await user.type(screen.getByLabelText(/your question/i), "Why is my report late?");
    await user.click(screen.getByRole("button", { name: /^ask$/i }));

    const thread = screen.getByRole("log");
    expect(within(thread).getByText("Why is my report late?")).toBeInTheDocument();
  });
});
