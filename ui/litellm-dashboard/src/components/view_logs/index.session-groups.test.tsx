import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SpendLogsTable from "./index";
import type { LogEntry } from "./columns";
import { renderWithProviders } from "../../../tests/test-utils";

// Hold a mutable reference so individual tests can change the return value.
let mockHookReturn: any;

vi.mock("./log_filter_logic", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./log_filter_logic")>();
  return {
    ...actual,
    useLogFilterLogic: vi.fn(() => mockHookReturn),
  };
});

vi.mock("../networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../networking")>();
  return {
    ...actual,
    uiSpendLogsCall: vi.fn().mockResolvedValue({
      data: [], total: 0, page: 1, page_size: 50, total_pages: 0,
    }),
    keyListCall: vi.fn().mockResolvedValue({ keys: [] }),
    keyInfoV1Call: vi.fn().mockResolvedValue({ info: {} }),
    allEndUsersCall: vi.fn().mockResolvedValue([]),
    sessionSpendLogsCall: vi.fn().mockResolvedValue({ data: [] }),
  };
});

vi.mock("../key_team_helpers/filter_helpers", () => ({
  fetchAllTeams: vi.fn().mockResolvedValue([]),
}));

const defaultProps = {
  accessToken: "test-token",
  token: "test-token",
  userRole: "Admin",
  userID: "user-1",
  premiumUser: false,
};

const makeLog = (overrides: Partial<LogEntry> = {}): LogEntry =>
  ({
    request_id: "req-x",
    call_type: "completion",
    api_key: "k",
    spend: 0,
    total_tokens: 10,
    prompt_tokens: 5,
    completion_tokens: 5,
    model: "gpt-4",
    model_id: "gpt-4",
    custom_llm_provider: "openai",
    startTime: "2026-01-01T00:00:00Z",
    endTime: "2026-01-01T00:00:01Z",
    request_duration_ms: 1000,
    metadata: { status: "success" },
    ...overrides,
  }) as LogEntry;

const baselineHookReturn = (data: LogEntry[]) => ({
  logsQuery: { isLoading: false, isFetching: false, isPlaceholderData: false, refetch: vi.fn() },
  filteredLogs: { data, total: data.length, page: 1, page_size: 50, total_pages: 1 },
  allTeams: [],
  handleFilterChange: vi.fn(),
  handleFilterReset: vi.fn(),
});

describe("SpendLogsTable — session group expansion (PR P)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    sessionStorage.clear();
  });

  it("renders one parent row plus its child sub-rows for a multi-call session, defaulting to fully expanded", async () => {
    // Three rows in the same session => one representative parent + two children.
    const sessionRows: LogEntry[] = [
      makeLog({ request_id: "req-1", session_id: "sess-A", session_total_count: 3, startTime: "2026-01-01T10:00:00Z" }),
      makeLog({ request_id: "req-2", session_id: "sess-A", session_total_count: 3, startTime: "2026-01-01T10:00:10Z" }),
      makeLog({ request_id: "req-3", session_id: "sess-A", session_total_count: 3, startTime: "2026-01-01T10:00:20Z" }),
    ];
    mockHookReturn = baselineHookReturn(sessionRows);

    renderWithProviders(<SpendLogsTable {...defaultProps} />);

    // Wait until the table body has rendered.
    await waitFor(() => {
      const rows = screen.getAllByRole("row");
      // 1 header row + 3 body rows (parent + 2 children, default expanded)
      expect(rows.length).toBeGreaterThanOrEqual(4);
    });

    // All 3 distinct request_ids should be visible somewhere in the table
    // (representative row + child sub-rows are all rendered when defaultExpandAll=true).
    expect(screen.getAllByText(/req-1|req-2|req-3/i).length).toBeGreaterThan(0);

    // The session row should show the multi-call type-cell badge "3" (session_total_count).
    // The type-cell badge for multi-call renders the count as text inside the badge span.
    const threes = screen.getAllByText("3");
    expect(threes.length).toBeGreaterThan(0);
  });

  it("renders a stand-alone (non-session) row without an expander chevron", async () => {
    mockHookReturn = baselineHookReturn([
      makeLog({ request_id: "req-solo", session_id: undefined, session_total_count: undefined }),
    ]);

    renderWithProviders(<SpendLogsTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getAllByRole("row").length).toBeGreaterThanOrEqual(2);
    });

    // No expander button should be present for a stand-alone row.
    expect(screen.queryByRole("button", { name: /expand session/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /collapse session/i })).not.toBeInTheDocument();
  });

  it("collapsing a session group hides its child sub-rows", async () => {
    const sessionRows: LogEntry[] = [
      makeLog({ request_id: "req-c1", session_id: "sess-B", session_total_count: 3, startTime: "2026-01-01T11:00:00Z" }),
      makeLog({ request_id: "req-c2", session_id: "sess-B", session_total_count: 3, startTime: "2026-01-01T11:00:10Z" }),
      makeLog({ request_id: "req-c3", session_id: "sess-B", session_total_count: 3, startTime: "2026-01-01T11:00:20Z" }),
    ];
    mockHookReturn = baselineHookReturn(sessionRows);

    renderWithProviders(<SpendLogsTable {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getAllByRole("row").length).toBeGreaterThanOrEqual(4);
    });

    const expandedRowCount = screen.getAllByRole("row").length;

    // Default expanded: chevron should read "Collapse session".
    const collapseBtn = await screen.findByRole("button", { name: /collapse session/i });
    const user = userEvent.setup();
    await user.click(collapseBtn);

    // After collapsing, body row count must drop by exactly 2 (the two child rows).
    await waitFor(() => {
      const rowsAfter = screen.getAllByRole("row").length;
      expect(rowsAfter).toBe(expandedRowCount - 2);
    });
  });
});
