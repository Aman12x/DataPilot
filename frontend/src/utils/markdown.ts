/** Strip markdown formatting to plain text (for clipboard copy). */
export function stripMarkdown(md: string): string {
  return md
    .replace(/^#{1,3}\s+/gm, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/^[-*]\s/gm, "• ")
    .trim();
}

/** Remove SQL / code fences from narrative before rendering. */
export function sanitiseNarrative(md: string): string {
  return md.replace(/```[\w]*\n[\s\S]*?```/g, "").replace(/\n{3,}/g, "\n\n").trim();
}
