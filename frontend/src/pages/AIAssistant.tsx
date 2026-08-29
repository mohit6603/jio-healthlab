import { FormEvent, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Bot,
  ChevronDown,
  Info,
  Loader2,
  Send,
  ShieldAlert,
  Sparkles,
  Trash2,
  UserRound
} from "lucide-react";
import { streamAssistant } from "../api";
import type { StreamSummary } from "../api";
import type { ApiErrorShape, ChatMessage, ChatResponse, Citation } from "../types";

const SAMPLE_PROMPTS = [
  "What does a CBC test measure?",
  "Explain a lipid profile in simple terms.",
  "What is a thyroid function test?",
  "What is the purpose of an HbA1c test?"
];

/**
 * Shown only until the first token arrives. Once text is streaming, the answer
 * itself is the progress indicator.
 */
const WAITING_STAGES = [
  "Searching the knowledge base…",
  "Reading the retrieved sections…",
  "Composing a grounded answer…"
];

function newId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function AIAssistant() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const threadRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<(() => void) | null>(null);

  // Cancel an in-flight stream if the user navigates away mid-answer.
  useEffect(() => () => abortRef.current?.(), []);

  useEffect(() => {
    threadRef.current?.scrollTo({
      top: threadRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [messages]);

  function ask(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;

    const pendingId = newId();
    setMessages((current) => [
      ...current,
      { id: newId(), role: "user", text: trimmed },
      { id: pendingId, role: "assistant", text: "", pending: true, streaming: true }
    ]);
    setQuestion("");
    setBusy(true);

    const patch = (updater: (message: ChatMessage) => ChatMessage) =>
      setMessages((current) =>
        current.map((message) => (message.id === pendingId ? updater(message) : message))
      );

    abortRef.current = streamAssistant(trimmed, {
      // Sources land before the first token, so the user can see what the
      // answer is being built from while it is still being written.
      onSources: (sources) => patch((m) => ({ ...m, sources })),

      onToken: (chunk) =>
        patch((m) => ({ ...m, pending: false, text: m.text + chunk })),

      // The safety screen rejected the finished answer; replace what was shown.
      onReplace: (replacement) =>
        patch((m) => ({ ...m, text: replacement, sources: [], suppressed: true })),

      onDone: (summary: StreamSummary) => {
        patch((m) => ({
          ...m,
          pending: false,
          streaming: false,
          answer: {
            answer: m.text,
            sources: summary.drop_sources ? [] : (m.sources ?? []),
            retrieval_count: summary.retrieval_count,
            grounded: summary.grounded,
            disclaimer: summary.disclaimer,
            model: summary.model,
            provider: summary.provider,
            finish_reason: summary.finish_reason,
            timings: summary.timings
          }
        }));
        setBusy(false);
      },

      onError: (error: ApiErrorShape) => {
        patch((m) => ({ ...m, pending: false, streaming: false, text: "", error }));
        setBusy(false);
      }
    });
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    void ask(question);
  }

  const empty = messages.length === 0;

  return (
    <div className="page assistant-page">
      <section className="assistant-header">
        <div>
          <p className="eyebrow">Knowledge assistant</p>
          <h1>Ask about tests, terminology and lab workflow</h1>
          <p className="assistant-intro">
            Answers are grounded in JIO HealthLab's own laboratory documentation
            and every response shows the sources it used.
          </p>
        </div>
        <button
          type="button"
          className="plain-action"
          onClick={() => setMessages([])}
          disabled={empty || busy}
        >
          <Trash2 size={16} />
          Clear conversation
        </button>
      </section>

      <div className="assistant-boundary" role="note">
        <ShieldAlert size={18} />
        <p>
          Informational only — not a medical diagnosis. This assistant does not
          interpret personal results or advise on treatment. Consult a qualified
          healthcare professional for clinical interpretation.
        </p>
      </div>

      <section
        className="assistant-thread"
        ref={threadRef}
        role="log"
        aria-live="polite"
        aria-label="Conversation"
      >
        {empty ? (
          <div className="assistant-empty">
            <Sparkles size={26} />
            <h2>Try one of these</h2>
            <div className="sample-prompts">
              {SAMPLE_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  className="sample-prompt"
                  onClick={() => void ask(prompt)}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))
        )}
      </section>

      <form className="assistant-composer" onSubmit={handleSubmit}>
        <label className="sr-only" htmlFor="assistant-question">
          Your question
        </label>
        <input
          id="assistant-question"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask what a test measures, or how the lab workflow works…"
          maxLength={1000}
          disabled={busy}
          autoComplete="off"
        />
        <button type="submit" className="primary-action" disabled={busy || !question.trim()}>
          {busy ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
          {busy ? "Thinking" : "Ask"}
        </button>
      </form>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <article className="bubble user">
        <div className="bubble-avatar">
          <UserRound size={16} />
        </div>
        <div className="bubble-body">{message.text}</div>
      </article>
    );
  }

  return (
    <article className="bubble assistant">
      <div className="bubble-avatar assistant">
        <Bot size={16} />
      </div>
      <div className="bubble-body">
        {message.pending && <PendingIndicator />}
        {message.error && <ErrorNotice error={message.error} />}
        {/* While streaming, render the partial text with a caret; once the
            done event lands, render the full answer with citations. */}
        {!message.pending && !message.error && message.streaming && (
          <StreamingBody text={message.text} sources={message.sources ?? []} />
        )}
        {!message.pending && !message.error && !message.streaming && message.answer && (
          <AnswerBody answer={message.answer} />
        )}
      </div>
    </article>
  );
}

function PendingIndicator() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(
      () => setStage((current) => Math.min(current + 1, WAITING_STAGES.length - 1)),
      3500
    );
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="assistant-pending" role="status">
      <Loader2 size={16} className="spin" />
      <span>{WAITING_STAGES[stage]}</span>
    </div>
  );
}

/** Maps an error code onto guidance the user can act on. */
function ErrorNotice({ error }: { error: ApiErrorShape }) {
  const guidance: Record<string, string> = {
    AI_SERVICE_UNAVAILABLE:
      "The assistant is offline right now. Reports and the dashboard still work.",
    AI_SERVICE_TIMEOUT:
      "That took too long to answer. Try a shorter or more specific question.",
    GENERATION_DISABLED:
      "Answer generation is switched off in this deployment. Search still works.",
    MODEL_UNAVAILABLE:
      "The language model could not be loaded. Search still works.",
    VECTOR_STORE_UNAVAILABLE:
      "The knowledge index is unavailable, so nothing can be retrieved.",
    NETWORK_ERROR: "Could not reach the server. Check your connection and retry."
  };

  return (
    <div className="assistant-error" role="alert">
      <AlertTriangle size={16} />
      <div>
        <strong>{guidance[error.code] ?? error.message}</strong>
        <span className="assistant-error-code">{error.code}</span>
      </div>
    </div>
  );
}

function StreamingBody({
  text,
  sources
}: {
  text: string;
  sources: Citation[];
}) {
  return (
    <>
      {sources.length > 0 && (
        <div className="streaming-sources">
          Grounding in {sources.length} source{sources.length === 1 ? "" : "s"}…
        </div>
      )}
      <p className="answer-text">
        {text}
        <span className="caret" aria-hidden="true" />
      </p>
    </>
  );
}

function AnswerBody({ answer }: { answer: ChatResponse }) {
  return (
    <>
      {!answer.grounded && (
        <div className="assistant-ungrounded" role="note">
          <Info size={15} />
          <span>Not backed by the knowledge base</span>
        </div>
      )}

      <p className="answer-text">{answer.answer}</p>

      {answer.sources.length > 0 && <SourceList sources={answer.sources} />}

      <p className="answer-disclaimer">{answer.disclaimer}</p>

      {answer.model && (
        <p className="answer-meta">
          {answer.model} · {Math.round(answer.timings.total_ms)} ms
        </p>
      )}
    </>
  );
}

function SourceList({ sources }: { sources: Citation[] }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="answer-sources">
      <button
        type="button"
        className="sources-toggle"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
      >
        <ChevronDown size={15} className={open ? "rotated" : ""} />
        {sources.length} source{sources.length === 1 ? "" : "s"}
      </button>

      {open && (
        <ol className="sources-list">
          {sources.map((source, index) => (
            <li key={`${source.chunk_id}-${index}`}>
              <div className="source-head">
                <span className="source-index">[{index + 1}]</span>
                <span className="source-title">{source.title}</span>
                <span className="source-score">{source.score.toFixed(3)}</span>
              </div>
              <div className="source-meta">
                {source.source}
                {source.section ? ` · ${source.section}` : ""}
              </div>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

export default AIAssistant;
