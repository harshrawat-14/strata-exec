/**
 * Offline handling: the site must say the engine is down whenever the API is unreachable,
 * and must send visitors to the results tour, which needs no backend.
 */
import { describe, it, expect, vi, beforeAll } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { probeBackend } from "../hooks/useBackendStatus";
import { SystemDownBanner } from "../components/SystemDownBanner";
import ResultsTour from "../pages/ResultsTour";

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

describe("SystemDownBanner", () => {
  it("apologises and takes the visitor to the tour", () => {
    render(
      <MemoryRouter initialEntries={["/simulator"]}>
        <Routes>
          <Route path="/simulator" element={<SystemDownBanner />} />
          <Route path="/results" element={<div>tour page</div>} />
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByText("The simulation engine is down")).toBeInTheDocument();
    expect(screen.getByText(/sorry for the inconvenience/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /take the tour/i }));
    expect(screen.getByText("tour page")).toBeInTheDocument();
  });
});

describe("ResultsTour", () => {
  beforeAll(() => {
    globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver;
  });

  const renderTour = () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={["/results"]}>
          <ResultsTour />
        </MemoryRouter>
      </QueryClientProvider>,
    );
  };

  it("renders from the snapshot with the backend down and filters by split", async () => {
    renderTour();
    expect(await screen.findByText("The simulation engine is down")).toBeInTheDocument();
    expect(screen.getByText("8 / 8")).toBeInTheDocument();
    expect(screen.getByText("1 / 17")).toBeInTheDocument();
    expect(screen.getByText("6,250")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /held-out · 9/i }));
    expect(screen.getByText("4 / 4")).toBeInTheDocument();
    expect(screen.getByText("0 / 5")).toBeInTheDocument();
    expect(screen.getByText("9 days")).toBeInTheDocument();
  });
});
