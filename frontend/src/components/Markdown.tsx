/**
 * Markdown — lightweight renderer for DataPilot narrative output.
 *
 * Handles the specific format the LLM produces:
 *   ## Section headers
 *   **bold** and *italic* inline
 *   - bullet lists
 *   ✅ / ⚠️ lines (treated as list items)
 *   `inline code`
 *
 * No external dependencies — keeps the bundle small.
 */

import { Fragment } from "react";
import { normalizeTypography } from "../utils/markdown";

// ── Inline parser ─────────────────────────────────────────────────────────────

function parseInline(text: string): React.ReactNode[] {
  // Split on **bold**, *italic*, `code`
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("*") && part.endsWith("*"))
      return <em key={i}>{part.slice(1, -1)}</em>;
    if (part.startsWith("`") && part.endsWith("`"))
      return <code key={i}>{part.slice(1, -1)}</code>;
    return <Fragment key={i}>{part}</Fragment>;
  });
}

// ── Block parser ──────────────────────────────────────────────────────────────

interface Block {
  type: "h1" | "h2" | "h3" | "p" | "li";
  text: string;
}

function parseBlocks(markdown: string): Block[] {
  const lines = normalizeTypography(markdown).split("\n");
  const blocks: Block[] = [];

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line) continue;
    if (line.startsWith("### "))
      blocks.push({ type: "h3", text: line.slice(4) });
    else if (line.startsWith("## "))
      blocks.push({ type: "h2", text: line.slice(3) });
    else if (line.startsWith("# "))
      blocks.push({ type: "h1", text: line.slice(2) });
    else if (/^[-*]\s/.test(line))
      blocks.push({ type: "li", text: line.slice(2) });
    else if (/^[✅⚠️🔴🟡🟢❌✔️]️?\s?/u.test(line))
      // LLM output sometimes leads with a status emoji. Render as a plain
      // list item with the emoji stripped: the UI carries no emoji glyphs.
      blocks.push({ type: "li", text: line.replace(/^[✅⚠️🔴🟡🟢❌✔️]️?\s?/u, "") });
    else
      blocks.push({ type: "p", text: line });
  }
  return blocks;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function Markdown({ content }: { content: string }) {
  const blocks = parseBlocks(content);
  const nodes: React.ReactNode[] = [];
  let listBuffer: Block[] = [];

  const flushList = () => {
    if (!listBuffer.length) return;
    nodes.push(
      <ul key={`ul-${nodes.length}`}>
        {listBuffer.map((b, i) => (
          <li key={i}>{parseInline(b.text)}</li>
        ))}
      </ul>
    );
    listBuffer = [];
  };

  for (const block of blocks) {
    if (block.type === "li") {
      listBuffer.push(block);
    } else {
      flushList();
      if (block.type === "h1")
        nodes.push(<h1 key={nodes.length}>{parseInline(block.text)}</h1>);
      else if (block.type === "h2")
        nodes.push(<h2 key={nodes.length}>{parseInline(block.text)}</h2>);
      else if (block.type === "h3")
        nodes.push(<h3 key={nodes.length}>{parseInline(block.text)}</h3>);
      else
        nodes.push(<p key={nodes.length}>{parseInline(block.text)}</p>);
    }
  }
  flushList();

  return <div className="md">{nodes}</div>;
}
