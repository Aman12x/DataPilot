/**
 * Normalize LLM typography for the UI: the product style carries no em
 * dashes, so replace mid-sentence em dashes with commas and bare ones with
 * hyphens before rendering.
 */
export function normalizeTypography(text: string): string {
  return text
    .replace(/\s+[\u2014\u2013]\s+/g, ", ")
    .replace(/[\u2014\u2013]/g, "-");
}

export interface Block {
  type: "h1" | "h2" | "h3" | "p" | "li" | "quote";
  text: string;
  /** Indent level for `li`; 0 unless the source bullet was nested. */
  depth?: number;
}

// `<!-- details -->` is a layout marker for the UI's show-more split. Nothing
// consumed it, so it rendered into the narrative as literal text.
const HTML_COMMENT_RE = /<!--[\s\S]*?-->/g;

// A marker only opens a bullet when whitespace follows it, which is what keeps
// `**Bold**` from being read as a `*` list item.
const BULLET_RE = /^(\s*)[-*•]\s+(.*)$/;

// The narrative writes its section headers as a whole line of bold rather than
// with `##`, so a bold-only line is a heading.
const BOLD_HEADING_RE = /^\*\*(.+?)\*\*:?\s*$/;

const BLOCKQUOTE_RE = /^\s*>\s?(.*)$/;

// A marker with nothing after it, or a `---` rule.
const MARKER_ONLY_RE = /^[-*•_]+$/;

const EMOJI_LEAD_RE = /^[✅⚠️🔴🟡🟢❌✔️]️?\s?/u;

/**
 * Split narrative markdown into renderable blocks.
 *
 * Lives here rather than in the component so it can be exercised without a DOM.
 */
export function parseBlocks(markdown: string): Block[] {
  const lines = normalizeTypography(markdown.replace(HTML_COMMENT_RE, "")).split("\n");
  const blocks: Block[] = [];

  for (const raw of lines) {
    const line = raw.trimEnd();
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (MARKER_ONLY_RE.test(trimmed)) continue;

    const bullet = BULLET_RE.exec(line);
    // "- **Key Findings**" is a heading the model happened to bullet.
    const headingText =
      BOLD_HEADING_RE.exec(trimmed)?.[1] ??
      (bullet ? BOLD_HEADING_RE.exec(bullet[2].trim())?.[1] : undefined);

    if (headingText) {
      blocks.push({ type: "h3", text: headingText });
    } else if (line.startsWith("### ")) {
      blocks.push({ type: "h3", text: line.slice(4) });
    } else if (line.startsWith("## ")) {
      blocks.push({ type: "h2", text: line.slice(3) });
    } else if (line.startsWith("# ")) {
      blocks.push({ type: "h1", text: line.slice(2) });
    } else if (bullet) {
      blocks.push({
        type: "li",
        text: bullet[2].trim(),
        depth: bullet[1].length >= 2 ? 1 : 0,
      });
    } else if (BLOCKQUOTE_RE.test(line)) {
      blocks.push({ type: "quote", text: BLOCKQUOTE_RE.exec(line)![1].trim() });
    } else if (EMOJI_LEAD_RE.test(line)) {
      // The UI carries no emoji glyphs, so drop the lead and treat it as an item.
      blocks.push({ type: "li", text: line.replace(EMOJI_LEAD_RE, ""), depth: 0 });
    } else {
      blocks.push({ type: "p", text: line });
    }
  }
  return blocks;
}

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

/**
 * Remove SQL / code fences from narrative before rendering.
 *
 * `partial` is for the live draft, which arrives token-batch by token-batch:
 * a fence the model has opened but not yet closed must not render as a
 * literal ``` line (the outer wrapper the model sometimes puts around the
 * whole answer) or as raw SQL (an embedded example mid-stream). For the
 * finished narrative an unclosed fence is left alone — stripping to the end
 * would hide real content behind a formatting slip.
 */
export function sanitiseNarrative(md: string, partial = false): string {
  let trimmed = md.trim();
  // If the entire narrative is wrapped in a single outer code fence, extract it.
  const outerFence = trimmed.match(/^```[\w]*\n([\s\S]*?)```\s*$/);
  if (outerFence) return outerFence[1].trim().replace(/\n{3,}/g, "\n\n");
  if (partial) {
    // Outer fence opened, not yet closed: drop the opener line and keep the body.
    trimmed = trimmed.replace(/^```[\w]*\n?/, "");
  }
  // Strip embedded (closed) code blocks, e.g. SQL examples.
  let out = trimmed.replace(/```[\w]*\n[\s\S]*?```/g, "");
  if (partial) {
    // An embedded fence still open at the end of the buffer: hide it until
    // it closes rather than rendering the raw code.
    const open = out.lastIndexOf("```");
    if (open !== -1) out = out.slice(0, open);
  }
  return out.replace(/\n{3,}/g, "\n\n").trim();
}
