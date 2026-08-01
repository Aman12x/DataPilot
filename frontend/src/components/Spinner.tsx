const styles: Record<string, React.CSSProperties> = {
  button: {
    width: 16, height: 16,
    border: "2px solid rgba(255,255,255,0.35)", borderTop: "2px solid #fff",
    borderRadius: "50%", animation: "spin 0.7s linear infinite",
    display: "inline-block",
  },
  page: {
    width: 32, height: 32,
    border: "3px solid var(--dp-line)", borderTop: "3px solid var(--dp-accent)",
    borderRadius: "50%", animation: "spin 0.8s linear infinite",
  },
  inline: {
    display: "inline-block", width: 10, height: 10,
    border: "2px solid rgba(14,124,107,0.25)", borderTop: "2px solid var(--dp-accent)",
    borderRadius: "50%", animation: "spin 0.7s linear infinite",
  },
};

export default function Spinner({ variant = "button" }: { variant?: "button" | "page" | "inline" }) {
  return <span style={styles[variant]} />;
}
