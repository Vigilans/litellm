import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LogDetailsDrawer } from "./LogDetailsDrawer";
import type { LogEntry } from "../columns";
import { renderWithProviders } from "../../../../tests/test-utils";

vi.mock("../../networking", () => ({
  sessionSpendLogsCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/logDetails/useLogDetails", () => ({
  useLogDetails: vi.fn(() => ({ data: undefined, isLoading: false })),
}));

// LogDetailContent / DrawerHeader / GuardrailJumpLink only show on the right
// pane; we don't want their full markup driving our sidebar-click assertions.
vi.mock("./LogDetailContent", () => ({
  LogDetailContent: () => <div data-testid="log-detail-content" />,
  GuardrailJumpLink: () => null,
}));
vi.mock("./DrawerHeader", () => ({
  DrawerHeader: ({ log }: { log: LogEntry | null }) => (
    <div data-testid="drawer-header">{log?.request_id}</div>
  ),
}));

import { sessionSpendLogsCall } from "../../networking";

const mockedSessionSpendLogsCall = vi.mocked(sessionSpendLogsCall);

const makeRow = (overrides: Partial<LogEntry> = {}): LogEntry =>
  ({
    request_id: "req-a",
    call_type: "completion",
    api_key: "k",
    spend: 0,
    total_tokens: 10,
    prompt_tokens: 5,
    completion_tokens: 5,
    model: "gpt-4",
    model_id: "gpt-4",
    startTime: "2026-01-01T00:00:00Z",
    endTime: "2026-01-01T00:00:01Z",
    request_duration_ms: 1000,
    session_id: "sess-1",
    session_total_count: 2,
    metadata: { status: "success" },
    ...overrides,
  }) as LogEntry;

describe("LogDetailsDrawer — session-sidebar selection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockedSessionSpendLogsCall.mockResolvedValue({
      data: [
        makeRow({ request_id: "req-a", startTime: "2026-01-01T00:00:00Z" }),
        makeRow({ request_id: "req-b", startTime: "2026-01-01T00:00:30Z" }),
      ],
    } as any);
  });

  const openProps = (overrides: Partial<React.ComponentProps<typeof LogDetailsDrawer>> = {}) => ({
    open: true,
    onClose: vi.fn(),
    logEntry: makeRow({ request_id: "req-a" }),
    sessionId: "sess-1",
    accessToken: "token",
    allLogs: [],
    onSelectLog: vi.fn(),
    startTime: "2026-01-01T00:00:00Z",
    ...overrides,
  });

  it("session mode: clicking a sidebar row does NOT propagate selection to parent (onSelectLog)", async () => {
    const onSelectLog = vi.fn();
    const props = openProps({ onSelectLog });
    renderWithProviders(<LogDetailsDrawer {...props} />);

    // Wait until the sidebar list is rendered (sessionLogs query resolved).
    const sidebarButtons = await waitFor(() => {
      const btns = screen.getAllByRole("button");
      expect(btns.length).toBeGreaterThanOrEqual(2);
      return btns;
    });

    // Sidebar row buttons in session mode use a pl-8 class; click the
    // non-selected one (i.e. not the currently-highlighted req-a row).
    const secondRowBtn = sidebarButtons.find((b) =>
      b.className.includes("pl-8") && !b.className.includes("bg-blue-50"),
    );
    expect(secondRowBtn).toBeDefined();

    const user = userEvent.setup();
    await user.click(secondRowBtn!);

    // Core regression assertion: parent's onSelectLog must NOT be called
    // when in session mode. The drawer manages selection internally via
    // setSelectedSessionRequestId. Propagating it would overwrite parent's
    // selectedLog with a row that lacks session_total_count, which then
    // truncates the sidebar back to 50 entries on the next refetch.
    expect(onSelectLog).not.toHaveBeenCalled();
  });

  it("session mode: sessionSpendLogsCall receives session_total_count as page_size", async () => {
    // Use a distinct sessionId so the QueryClient cache (shared across tests
    // via tests/test-utils.tsx) doesn't return a cached hit from another test.
    const sessionId = "sess-pagesize-test";
    const props = openProps({
      sessionId,
      logEntry: makeRow({ request_id: "req-a", session_id: sessionId, session_total_count: 227 }),
    });
    renderWithProviders(<LogDetailsDrawer {...props} />);

    await waitFor(() => {
      expect(mockedSessionSpendLogsCall).toHaveBeenCalled();
    });

    expect(mockedSessionSpendLogsCall).toHaveBeenCalledWith("token", sessionId, 227);
  });
});
