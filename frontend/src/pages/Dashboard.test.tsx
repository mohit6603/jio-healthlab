import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Dashboard from "./Dashboard";
import type { DashboardSummary, Report } from "../types";

vi.mock("../api", async () => {
  const actual = await vi.importActual<typeof import("../api")>("../api");
  return {
    ...actual,
    getDashboard: vi.fn(),
    listReports: vi.fn(),
    getTestTypes: vi.fn(),
    createReport: vi.fn(),
    updateReport: vi.fn(),
    updateReportStatus: vi.fn(),
    deleteReport: vi.fn()
  };
});

const api = await import("../api");
const mockDashboard = vi.mocked(api.getDashboard);
const mockList = vi.mocked(api.listReports);
const mockTestTypes = vi.mocked(api.getTestTypes);
const mockCreate = vi.mocked(api.createReport);
const mockStatus = vi.mocked(api.updateReportStatus);
const mockDelete = vi.mocked(api.deleteReport);

function report(overrides: Partial<Report> = {}): Report {
  return {
    id: 1,
    patient_name: "Asha Nair",
    age: 34,
    test_type: "CBC Panel",
    gender: "Female",
    phone: null,
    email: null,
    city: "Mumbai",
    lab_branch: "Andheri Hub",
    doctor_name: "Dr Meera Iyer",
    status: "processing",
    priority: "urgent",
    sample_collected_at: null,
    result_due_at: null,
    notes: null,
    created_at: "2026-08-28T09:00:00",
    updated_at: "2026-08-28T09:00:00",
    ...overrides
  };
}

function summary(overrides: Partial<DashboardSummary> = {}): DashboardSummary {
  return {
    total_reports: 29,
    ready_reports: 4,
    urgent_reports: 10,
    avg_age: 42.1,
    unique_tests: 6,
    latest_report_id: 31,
    due_soon: [],
    recent_reports: [],
    by_status: { processing: 12, ready: 4 },
    by_test_type: { "CBC Panel": 9, Thyroid: 5 },
    by_city: { Mumbai: 14, Pune: 8 },
    ...overrides
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockDashboard.mockResolvedValue(summary());
  mockList.mockResolvedValue([report()]);
  mockTestTypes.mockResolvedValue(["CBC Panel", "Thyroid", "Kidney Function"]);
});

describe("Dashboard", () => {
  it("loads dashboard, reports and test types on mount", async () => {
    render(<Dashboard />);

    await waitFor(() => expect(mockDashboard).toHaveBeenCalled());
    expect(mockList).toHaveBeenCalled();
    expect(mockTestTypes).toHaveBeenCalled();
  });

  it("renders the headline metrics", async () => {
    render(<Dashboard />);

    // Scoped: "Ready" and "Urgent" also appear as status pills in the table.
    const metrics = await screen.findByLabelText("Lab metrics");
    expect(within(metrics).getByText("29")).toBeInTheDocument();
    expect(within(metrics).getByText("Total reports")).toBeInTheDocument();
    expect(within(metrics).getByText("Ready")).toBeInTheDocument();
    expect(within(metrics).getByText("Urgent")).toBeInTheDocument();
    expect(within(metrics).getByText("42.1 yrs")).toBeInTheDocument();
  });

  it("renders reports in the table", async () => {
    render(<Dashboard />);

    const table = await screen.findByRole("table");
    expect(within(table).getByText("Asha Nair")).toBeInTheDocument();
    expect(within(table).getByText("CBC Panel")).toBeInTheDocument();
  });

  it("shows an empty state when there are no reports", async () => {
    mockList.mockResolvedValue([]);
    render(<Dashboard />);

    expect(
      await screen.findByText(/no reports match the current filters/i)
    ).toBeInTheDocument();
  });

  it("surfaces a load failure without blanking the page", async () => {
    mockDashboard.mockRejectedValue(new Error("service unavailable"));
    mockList.mockRejectedValue(new Error("service unavailable"));
    render(<Dashboard />);

    expect(await screen.findByText("service unavailable")).toBeInTheDocument();
    // The shell is still rendered, not replaced by the error.
    expect(screen.getByText("Total reports")).toBeInTheDocument();
  });

  it("filters by search text", async () => {
    const user = userEvent.setup();
    render(<Dashboard />);
    await screen.findByRole("table");

    await user.type(screen.getByPlaceholderText(/search patient/i), "Asha");

    await waitFor(() =>
      expect(mockList).toHaveBeenCalledWith(
        expect.objectContaining({ search: "Asha" })
      )
    );
  });

  it("filters by status", async () => {
    const user = userEvent.setup();
    render(<Dashboard />);
    await screen.findByRole("table");

    // "Status" labels both the filter and the intake form field; the filter
    // is the one inside the filters bar.
    const [filterStatus] = screen.getAllByLabelText("Status");
    await user.selectOptions(filterStatus, "ready");

    await waitFor(() =>
      expect(mockList).toHaveBeenCalledWith(
        expect.objectContaining({ status: "ready" })
      )
    );
  });

  it("creates a report from the intake form", async () => {
    mockCreate.mockResolvedValue(report({ id: 2 }));
    const user = userEvent.setup();
    render(<Dashboard />);
    await screen.findByRole("table");

    await user.type(screen.getByLabelText("Patient name"), "New Patient");
    await user.type(screen.getByLabelText("Age"), "44");
    await user.click(screen.getByRole("button", { name: /create report/i }));

    await waitFor(() => expect(mockCreate).toHaveBeenCalled());
    expect(mockCreate.mock.calls[0][0].patient_name).toBe("New Patient");
  });

  it("marks a report delivered", async () => {
    mockStatus.mockResolvedValue(report({ status: "delivered" }));
    const user = userEvent.setup();
    render(<Dashboard />);
    const table = await screen.findByRole("table");

    await user.click(within(table).getByTitle("Mark delivered"));

    await waitFor(() => expect(mockStatus).toHaveBeenCalledWith(1, "delivered"));
  });

  it("asks for confirmation before deleting", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    mockDelete.mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<Dashboard />);
    const table = await screen.findByRole("table");

    await user.click(within(table).getByTitle("Delete"));

    expect(confirm).toHaveBeenCalled();
    await waitFor(() => expect(mockDelete).toHaveBeenCalledWith(1));
  });

  it("does not delete when confirmation is declined", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    const user = userEvent.setup();
    render(<Dashboard />);
    const table = await screen.findByRole("table");

    await user.click(within(table).getByTitle("Delete"));

    expect(mockDelete).not.toHaveBeenCalled();
  });

  it("marks urgent reports visibly", async () => {
    render(<Dashboard />);

    const table = await screen.findByRole("table");
    expect(within(table).getByText("urgent")).toBeInTheDocument();
  });

  it("surfaces a delete failure to the user", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    mockDelete.mockRejectedValue(new Error("could not delete"));
    const user = userEvent.setup();
    render(<Dashboard />);
    const table = await screen.findByRole("table");

    await user.click(within(table).getByTitle("Delete"));

    expect(await screen.findByText("could not delete")).toBeInTheDocument();
  });

  it("renders the insight breakdowns", async () => {
    render(<Dashboard />);

    expect(await screen.findByText(/status mix/i)).toBeInTheDocument();
    expect(screen.getByText(/test demand/i)).toBeInTheDocument();
  });
});
