const f: Record<string, React.CSSProperties> = {
  group: { display: "flex", flexDirection: "column", gap: 6 },
  label: { fontSize: 12, fontWeight: 600, color: "#a6adc8", letterSpacing: "0.04em", textTransform: "uppercase" },
  input: { padding: "10px 12px", background: "#181825", color: "#cdd6f4", border: "1px solid #313244", borderRadius: 8, fontSize: 14, transition: "border-color 0.2s" },
};

export default function FormField({ label, type, value, onChange, placeholder, required = true, autoFocus }: {
  label: string; type: string; value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder: string; required?: boolean; autoFocus?: boolean;
}) {
  return (
    <div style={f.group}>
      <label style={f.label}>{label}</label>
      <input
        style={f.input}
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        autoFocus={autoFocus}
        autoComplete={type === "password" ? "current-password" : "on"}
      />
    </div>
  );
}
