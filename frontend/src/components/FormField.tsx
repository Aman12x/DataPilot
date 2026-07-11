const f: Record<string, React.CSSProperties> = {
  group: { display: "flex", flexDirection: "column", gap: 6 },
  label: { fontSize: 11, fontWeight: 600, color: "#a6adc8", letterSpacing: "0.06em", textTransform: "uppercase" },
};

export default function FormField({ label, type, value, onChange, placeholder, required = true, autoFocus, autoComplete }: {
  label: string; type: string; value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder: string; required?: boolean; autoFocus?: boolean;
  autoComplete?: string;
}) {
  const resolvedAutoComplete = autoComplete ?? (type === "password" ? "current-password" : "on");

  return (
    <div style={f.group}>
      <label style={f.label}>{label}</label>
      <input
        className="dp-input"
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        autoFocus={autoFocus}
        autoComplete={resolvedAutoComplete}
        minLength={type === "password" ? 8 : undefined}
      />
    </div>
  );
}
