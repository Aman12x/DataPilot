import client from "../api/client";

/**
 * Download a run's PDF report without window.open.
 *
 * The old flow opened the /pdf URL in a new tab after awaiting the token
 * fetch; popup blockers (Safari especially) silently drop a window.open that
 * runs after an async gap, so the button appeared to do nothing. Fetching the
 * bytes and clicking a temporary <a download> anchor avoids popups entirely
 * and surfaces real failures to the caller.
 */
export async function downloadRunPdf(runId: string): Promise<void> {
  const { data: tok } = await client.get<{ pdf_token: string }>(`/runs/${runId}/pdf-token`);
  const { data: pdf } = await client.get<Blob>(`/runs/${runId}/pdf`, {
    params: { pdf_token: tok.pdf_token },
    responseType: "blob",
  });
  const url = URL.createObjectURL(pdf);
  const a = document.createElement("a");
  a.href = url;
  a.download = `datapilot-${runId.slice(0, 8)}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
