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
import { parseBlocks, type Block } from "../utils/markdown";

// ── Inline parser ─────────────────────────────────────────────────────────────

function parseInline(text: string): React.ReactNode[] {
  // Split on **bold**, *italic*, `code`. The bold and italic bodies are
  // non-greedy rather than `[^*]+` so a span containing a `*` still closes on
  // its own delimiter instead of swallowing the rest of the line.
  const parts = text.split(/(\*\*.+?\*\*|\*[^*]+\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.length > 4 && part.startsWith("**") && part.endsWith("**"))
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.length > 2 && part.startsWith("*") && part.endsWith("*"))
      return <em key={i}>{part.slice(1, -1)}</em>;
    if (part.startsWith("`") && part.endsWith("`"))
      return <code key={i}>{part.slice(1, -1)}</code>;
    return <Fragment key={i}>{part}</Fragment>;
  });
}

// Block parsing lives in utils/markdown.ts so it can be tested without a DOM.

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
          <li key={i} className={b.depth ? "md-li-nested" : undefined}>
            {parseInline(b.text)}
          </li>
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
      if (block.type === "quote")
        nodes.push(
          <blockquote key={nodes.length} className="md-quote">
            {parseInline(block.text)}
          </blockquote>
        );
      else if (block.type === "h1")
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
