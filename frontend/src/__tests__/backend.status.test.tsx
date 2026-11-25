/**
 * The health probe: the site must say the engine is down whenever the API is unreachable.
 */
import { describe, it, expect, vi } from "vitest";
import { probeBackend } from "../hooks/useBackendStatus";

const response = (status: number, body: string, type = "application/json") =>
  new Response(body, { status, headers: { "Content-Type": type } });

describe("probeBackend", () => {
  it("is online only for the API's own ok body", async () => {
    const f = vi.fn().mockResolvedValue(response(200, '{"status":"ok","version":"1"}'));
    expect(await probeBackend(f as unknown as typeof fetch)).toBe(true);
  });

  // Regression: on Vercel an unproxied /health falls through to index.html with a 200.
  it("treats an HTML 200 (SPA fallback) as offline", async () => {
    const f = vi.fn().mockResolvedValue(response(200, "<!doctype html><html></html>", "text/html"));
    expect(await probeBackend(f as unknown as typeof fetch)).toBe(false);
  });

  it("treats errors, 5xx and other bodies as offline", async () => {
    expect(await probeBackend(vi.fn().mockRejectedValue(new TypeError("fetch failed")) as unknown as typeof fetch)).toBe(false);
    expect(await probeBackend(vi.fn().mockResolvedValue(response(503, "")) as unknown as typeof fetch)).toBe(false);
    expect(await probeBackend(vi.fn().mockResolvedValue(response(200, '{"status":"degraded"}')) as unknown as typeof fetch)).toBe(false);
  });

  it("gives up after the timeout", async () => {
    const hang = vi.fn((_: unknown, init: RequestInit) => new Promise((_res, rej) => {
      init.signal!.addEventListener("abort", () => rej(new DOMException("aborted", "AbortError")));
    }));
    expect(await probeBackend(hang as unknown as typeof fetch, 10)).toBe(false);
  });
});

