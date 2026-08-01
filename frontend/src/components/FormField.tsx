import { useId } from "react";

const f: Record<string, React.CSSProperties> = {
  group: { display: "flex", flexDirection: "column", gap: 6 },
  label: {
    fontSize: 11,
    fontWeight: 650,
    color: "var(--dp-ink-muted)",
    letterSpacing: "0.06em",
    textTransform: "uppercase",
  },
};

export default function FormField({ label, type, value, onChange, placeholder, required = true, autoFocus, autoComplete }: {
  label: string; type: string; value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder: string; required?: boolean; autoFocus?: boolean;
  autoComplete?: string;
}) {
  const resolvedAutoComplete = autoComplete ?? (type === "password" ? "current-password" : "on");
  // Sibling label, so it needs an explicit htmlFor/id pair. Without it screen
  // readers announce the field unlabelled and clicking the label does nothing.
  // (Other forms in this app nest the input inside the label, which associates
  // the two implicitly and needs no id.)
  const id = useId();

  return (
    <div style={f.group}>
      <label style={f.label} htmlFor={id}>{label}</label>
      <input
        id={id}
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
