import React from "react";

interface Props {
  children: React.ReactNode;
  /** Shown under the title. Defaults to a generic message. */
  hint?: string;
  /** When this value changes, a crashed boundary resets and re-renders its children. */
  resetKey?: unknown;
}

interface State {
  error: Error | null;
}

/**
 * Catches render errors below it so one broken component shows an inline
 * error card instead of white-screening the whole app.
 */
export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidUpdate(prevProps: Props) {
    if (this.state.error && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ error: null });
    }
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div style={s.card} className="fade-in">
        <h3 style={s.title}>Something went wrong displaying this</h3>
        <p style={s.message}>
          {this.props.hint ?? "The rest of the app is unaffected."}
        </p>
        <p style={s.detail}>{this.state.error.message}</p>
        <div style={s.actions}>
          <button style={s.btnRetry} onClick={() => this.setState({ error: null })}>
            Try again
          </button>
          <button style={s.btnReload} onClick={() => window.location.reload()}>
            Reload page
          </button>
        </div>
      </div>
    );
  }
}

const s: Record<string, React.CSSProperties> = {
  card: {
    background: "var(--dp-surface)",
    border: "1px solid var(--dp-line)",
    borderRadius: 10,
    padding: "28px 32px",
    margin: "40px auto",
    boxShadow: "var(--dp-shadow)",
    maxWidth: 560,
  },
  title: {
    color: "var(--dp-danger)",
    marginTop: 0,
    fontSize: 18,
    fontWeight: 600,
  },
  message: { color: "var(--dp-ink-secondary)", fontSize: 14, lineHeight: 1.6 },
  detail: {
    color: "var(--dp-ink-muted)",
    fontSize: 12,
    fontFamily: "var(--dp-mono, monospace)",
    background: "var(--dp-surface-2)",
    borderRadius: 8,
    padding: "8px 12px",
    overflowWrap: "break-word",
  },
  actions: { display: "flex", gap: 12, marginTop: 20, flexWrap: "wrap" },
  btnRetry: {
    padding: "11px 24px",
    background: "var(--dp-accent)",
    color: "var(--dp-accent-ink)",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    fontWeight: 600,
    fontSize: 14,
  },
  btnReload: {
    padding: "11px 24px",
    background: "transparent",
    color: "var(--dp-ink-secondary)",
    border: "1px solid var(--dp-line-strong)",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 500,
  },
};
