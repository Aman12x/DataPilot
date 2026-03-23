const styles: Record<string, React.CSSProperties> = {
  button: {
    width: 16, height: 16,
    border: "2px solid #1e1e2e44", borderTop: "2px solid #1e1e2e",
    borderRadius: "50%", animation: "spin 0.7s linear infinite",
    display: "inline-block",
  },
  page: {
    width: 32, height: 32,
    border: "3px solid #313244", borderTop: "3px solid #89b4fa",
    borderRadius: "50%", animation: "spin 0.8s linear infinite",
  },
  inline: {
    display: "inline-block", width: 10, height: 10,
    border: "2px solid #89b4fa44", borderTop: "2px solid #89b4fa",
    borderRadius: "50%", animation: "spin 0.7s linear infinite",
  },
};

export default function Spinner({ variant = "button" }: { variant?: "button" | "page" | "inline" }) {
  return <span style={styles[variant]} />;
}
