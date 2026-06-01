import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Row } from "@tanstack/react-table";
import { createColumns, type LogEntry } from "./columns";

// Build a fake TanStack Row<LogEntry> sufficient for the expander cell.
function makeFakeRow({
  canExpand,
  isExpanded,
  onToggle,
}: {
  canExpand: boolean;
  isExpanded: boolean;
  onToggle: () => void;
}): Row<LogEntry> {
  return {
    getCanExpand: () => canExpand,
    getIsExpanded: () => isExpanded,
    getToggleExpandedHandler: () => onToggle,
  } as unknown as Row<LogEntry>;
}

const renderExpanderCell = (row: Row<LogEntry>) => {
  const cols = createColumns();
  const expander = cols.find((c) => (c as any).id === "expander");
  expect(expander).toBeDefined();
  const cellRenderer = (expander as any).cell as (ctx: { row: Row<LogEntry> }) => React.ReactNode;
  return render(<>{cellRenderer({ row })}</>);
};

describe("createColumns — expander column", () => {
  it("renders an expandable session row with a clickable chevron button", async () => {
    const onToggle = vi.fn();
    renderExpanderCell(makeFakeRow({ canExpand: true, isExpanded: false, onToggle }));

    const btn = screen.getByRole("button", { name: /expand session/i });
    expect(btn).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(btn);
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("renders 'collapse' label and rotated chevron when already expanded", () => {
    renderExpanderCell(makeFakeRow({ canExpand: true, isExpanded: true, onToggle: vi.fn() }));
    expect(screen.getByRole("button", { name: /collapse session/i })).toBeInTheDocument();
  });

  it("renders nothing for non-expandable rows (single-call sessions / stand-alone)", () => {
    const { container } = renderExpanderCell(
      makeFakeRow({ canExpand: false, isExpanded: false, onToggle: vi.fn() }),
    );
    expect(container.firstChild).toBeNull();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("expander click stops propagation so it doesn't trigger row onClick", async () => {
    const onToggle = vi.fn();
    const onRowClick = vi.fn();
    const row = makeFakeRow({ canExpand: true, isExpanded: false, onToggle });
    const cols = createColumns();
    const expander = cols.find((c) => (c as any).id === "expander") as any;
    render(
      <div onClick={onRowClick}>
        {expander.cell({ row })}
      </div>,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: /expand session/i }));
    expect(onToggle).toHaveBeenCalledTimes(1);
    // stopPropagation: row's click handler must NOT fire.
    expect(onRowClick).not.toHaveBeenCalled();
  });

  it("expander column has a narrow size so DataTable applies tight column styling", () => {
    const cols = createColumns();
    const expander = cols.find((c) => (c as any).id === "expander") as any;
    expect(expander.size).toBe(28);
  });
});
