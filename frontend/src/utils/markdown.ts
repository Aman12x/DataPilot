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
  const trimmed = md.trim();
  // If the entire narrative is wrapped in a single outer code fence, extract it.
  const outerFence = trimmed.match(/^```[\w]*\n([\s\S]*?)```\s*$/);
  if (outerFence) return outerFence[1].trim().replace(/\n{3,}/g, "\n\n");
  // Otherwise strip embedded code blocks (e.g. SQL examples).
  return trimmed.replace(/```[\w]*\n[\s\S]*?```/g, "").replace(/\n{3,}/g, "\n\n").trim();
}
