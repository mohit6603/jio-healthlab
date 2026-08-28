import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AIAnalytics from "./AIAnalytics";
import { ApiError } from "../api";
import type { RiskAnalytics } from "../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return { ...actual, getRiskAnalytics: vi.fn() };
});

const { getRiskAnalytics } = await import("../api");
const mockGet = vi.mocked(getRiskAnalytics);

function analytics(overrides: Partial<RiskAnalytics> = {}): RiskAnalytics {
  return {
    generated_at: "2026-08-28T10:00:00+00:00",
    model_version: "v20260828065813",
    synthetic_model: true,
    reports_scored: 4,
    at_risk: 3,
    high_risk: 2,
    predicted_late: 3,
    average_probability: 0.58,
    highest_risk_branch: {
      branch: "Salt Lake",
      average_probability: 0.82,
      reports: 2,
      high_risk: 2
    },
    risk_distribution: { low: 1, medium: 1, high: 2 },
    by_branch: [
      { label: "Salt Lake", count: 2, average_probability: 0.82, high_risk: 2 },
      { label: "BKC Flagship", count: 2, average_probability: 0.34, high_risk: 0 }
    ],
    by_test_type: [
      { label: "Thyroid", count: 1, average_probability: 0.91, high_risk: 1 }
    ],
    reports: [
      {
        report_id: 12,
        test_type: "Thyroid",
        branch: "Salt Lake",
        city: "Kolkata",
        priority: "urgent",
        status: "processing",
        result_due_at: null,
        delay_probability: 0.91,
        risk_level: "high"
      },
      {
        report_id: 7,
        test_type: "CBC Panel",
        branch: "BKC Flagship",
        city: "Mumbai",
        priority: "routine",
        status: "registered",
        result_due_at: null,
        delay_probability: 0.22,
        risk_level: "low"
      }
    ],
    ...overrides
  };
}

beforeEach(() => mockGet.mockReset());

describe("AIAnalytics", () => {
  it("shows a loading state first", async () => {
    mockGet.mockReturnValue(new Promise(() => {}));
    render(<AIAnalytics />);

    expect(await screen.findByRole("status")).toHaveTextContent(/scoring in-flight/i);
  });

  it("renders the headline metrics", async () => {
    mockGet.mockResolvedValue(analytics());
    render(<AIAnalytics />);

    const metrics = await screen.findByLabelText("Risk metrics");
    expect(within(metrics).getByText("Reports at risk")).toBeInTheDocument();
    expect(within(metrics).getByText("High risk")).toBeInTheDocument();
    expect(within(metrics).getByText("Average probability")).toBeInTheDocument();
    expect(within(metrics).getByText("Predicted late")).toBeInTheDocument();
    expect(within(metrics).getByText("Highest-risk branch")).toBeInTheDocument();
  });

  it("formats the average probability as a percentage", async () => {
    mockGet.mockResolvedValue(analytics({ average_probability: 0.58 }));
    render(<AIAnalytics />);

    expect(await screen.findByText("58%")).toBeInTheDocument();
  });

  it("names the highest-risk branch", async () => {
    mockGet.mockResolvedValue(analytics());
    render(<AIAnalytics />);

    const metrics = await screen.findByLabelText("Risk metrics");
    expect(within(metrics).getByText("Salt Lake")).toBeInTheDocument();
  });

  it("renders the triage table with risk and probability", async () => {
    mockGet.mockResolvedValue(analytics());
    render(<AIAnalytics />);

    const table = await screen.findByRole("table");
    expect(within(table).getByText("#12")).toBeInTheDocument();
    expect(within(table).getByText("Thyroid")).toBeInTheDocument();
    expect(within(table).getByText("91%")).toBeInTheDocument();
    expect(within(table).getByText("high")).toBeInTheDocument();
  });

  it("shows every documented table column", async () => {
    mockGet.mockResolvedValue(analytics());
    render(<AIAnalytics />);

    const table = await screen.findByRole("table");
    for (const heading of [
      "Report",
      "Test",
      "Branch",
      "Priority",
      "Status",
      "Risk",
      "Probability"
    ]) {
      expect(within(table).getByText(heading)).toBeInTheDocument();
    }
  });

  it("never shows a patient name", async () => {
    mockGet.mockResolvedValue(analytics());
    const { container } = render(<AIAnalytics />);

    await screen.findByRole("table");
    expect(container.textContent).not.toMatch(/patient/i);
  });

  it("warns that the model is trained on synthetic data", async () => {
    mockGet.mockResolvedValue(analytics());
    render(<AIAnalytics />);

    expect(await screen.findByRole("note")).toHaveTextContent(/synthetic/i);
    expect(screen.getByRole("note")).toHaveTextContent(/not an operational forecast/i);
  });

  it("omits the synthetic warning for a real model", async () => {
    mockGet.mockResolvedValue(analytics({ synthetic_model: false }));
    render(<AIAnalytics />);

    await screen.findByRole("table");
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });

  it("reports the model version", async () => {
    mockGet.mockResolvedValue(analytics());
    render(<AIAnalytics />);

    expect(await screen.findByText(/v20260828065813/)).toBeInTheDocument();
  });

  it("shows an empty state when nothing is in flight", async () => {
    mockGet.mockResolvedValue(
      analytics({ reports_scored: 0, reports: [], at_risk: 0, high_risk: 0 })
    );
    render(<AIAnalytics />);

    expect(await screen.findByText(/nothing in flight/i)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });

  it("explains an untrained model and offers a retry", async () => {
    mockGet.mockRejectedValue(
      new ApiError(503, { code: "ML_MODEL_NOT_TRAINED", message: "train it" })
    );
    render(<AIAnalytics />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/no delay model has been trained/i);
    expect(within(alert).getByRole("button", { name: /retry/i })).toBeInTheDocument();
  });

  it("explains an AI outage", async () => {
    mockGet.mockRejectedValue(
      new ApiError(503, { code: "AI_SERVICE_UNAVAILABLE", message: "down" })
    );
    render(<AIAnalytics />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /reports and the dashboard still work/i
    );
  });

  it("retries on demand", async () => {
    mockGet.mockRejectedValueOnce(
      new ApiError(503, { code: "AI_SERVICE_UNAVAILABLE", message: "down" })
    );
    mockGet.mockResolvedValueOnce(analytics());
    const user = userEvent.setup();
    render(<AIAnalytics />);

    await user.click(await screen.findByRole("button", { name: /retry/i }));

    expect(await screen.findByRole("table")).toBeInTheDocument();
  });

  it("refreshes on demand", async () => {
    mockGet.mockResolvedValue(analytics());
    const user = userEvent.setup();
    render(<AIAnalytics />);

    await screen.findByRole("table");
    await user.click(screen.getByRole("button", { name: /refresh/i }));

    await waitFor(() => expect(mockGet).toHaveBeenCalledTimes(2));
  });

  it("renders the risk distribution", async () => {
    mockGet.mockResolvedValue(analytics());
    render(<AIAnalytics />);

    expect(await screen.findByText("Risk distribution")).toBeInTheDocument();
    expect(screen.getByText("Low")).toBeInTheDocument();
    expect(screen.getByText("Medium")).toBeInTheDocument();
    expect(screen.getByText("High")).toBeInTheDocument();
  });

  it("renders branch and test-type breakdowns", async () => {
    mockGet.mockResolvedValue(analytics());
    render(<AIAnalytics />);

    expect(await screen.findByText("By branch")).toBeInTheDocument();
    expect(screen.getByText("By test type")).toBeInTheDocument();
    expect(screen.getByText("82%")).toBeInTheDocument();
  });
});
