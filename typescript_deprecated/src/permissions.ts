import {
  type Component,
  type Focusable,
  SelectList,
  type SelectListTheme,
  Text,
  TUI,
} from "@mariozechner/pi-tui";

export type Decision = "once" | "always" | "deny";

const identity = (s: string) => s;
const listTheme: SelectListTheme = {
  selectedPrefix: identity,
  selectedText: identity,
  description: identity,
  scrollInfo: identity,
  noMatch: identity,
};

/**
 * Component that stacks a question (Text) above a SelectList and
 * forwards keyboard input to the list. Used as an overlay payload.
 */
class ConfirmOverlay implements Component, Focusable {
  focused = false;
  constructor(
    private readonly text: Text,
    private readonly list: SelectList,
  ) {}
  render(width: number): string[] {
    return [...this.text.render(width), "", ...this.list.render(width)];
  }
  handleInput(data: string): void {
    this.list.handleInput(data);
  }
  invalidate(): void {
    this.text.invalidate();
    this.list.invalidate();
  }
}

/** Format the tool args for the prompt — short single-line preview. */
function previewArgs(args: unknown): string {
  try {
    const json = JSON.stringify(args);
    return json.length > 200 ? json.slice(0, 200) + "…" : json;
  } catch {
    return String(args);
  }
}

/**
 * Show a modal overlay asking the user to allow/deny a tool call.
 * Resolves with the user's choice (or "deny" on Esc/cancel).
 */
export function confirmToolCall(
  tui: TUI,
  toolName: string,
  args: unknown,
): Promise<Decision> {
  return new Promise<Decision>((resolve) => {
    const question = new Text(
      `Allow tool "${toolName}"?\n  args: ${previewArgs(args)}`,
    );
    const list = new SelectList(
      [
        { value: "once", label: "Allow once", description: "Run this call only" },
        { value: "always", label: "Allow always", description: `Auto-allow "${toolName}" for this session` },
        { value: "deny", label: "Deny", description: "Block this tool call" },
      ],
      5,
      listTheme,
    );

    const overlay = new ConfirmOverlay(question, list);
    const handle = tui.showOverlay(overlay, {
      width: "60%",
      anchor: "center",
      margin: 1,
    });
    tui.setFocus(overlay);

    const finish = (decision: Decision) => {
      handle.hide();
      resolve(decision);
    };
    list.onSelect = (item) => finish(item.value as Decision);
    list.onCancel = () => finish("deny");
  });
}
