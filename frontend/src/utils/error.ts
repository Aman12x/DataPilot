/** Extract a human-readable error message from an Axios error response. */
export function extractApiError(err: unknown, fallback: string): string {
  return (
    (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? fallback
  );
}
