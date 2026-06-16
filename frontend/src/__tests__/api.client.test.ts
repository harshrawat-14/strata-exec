/**
 * Tests for the API client (axios instance).
 *
 * Verifies:
 * 1. Request interceptor attaches the Bearer token from localStorage
 * 2. Response interceptor removes token and redirects on 401
 * 3. Successful responses pass through unchanged
 * 4. Non-401 errors are re-rejected without touching the token
 * 5. All expected API functions are exported
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// Helper: pull the last registered request interceptor from an axios instance
function getLastRequestInterceptor(inst: any) {
  const handlers = inst.interceptors.request.handlers;
  return handlers[handlers.length - 1]?.fulfilled;
}

// Helper: pull the last registered response interceptor
function getLastResponseInterceptor(inst: any) {
  const handlers = inst.interceptors.response.handlers;
  return handlers[handlers.length - 1];
}

// ── Request interceptor ────────────────────────────────────────────────────────

describe("API client — request interceptor", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("attaches Authorization header when token is in localStorage", async () => {
    localStorage.setItem("strataexec_token", "test_jwt_token_123");
    const { api } = await import("../api/client");
    const interceptor = getLastRequestInterceptor(api);

    const config = { headers: {} as Record<string, string> };
    const result = interceptor(config);
    expect(result.headers.Authorization).toBe("Bearer test_jwt_token_123");
  });

  it("does NOT attach Authorization header when no token in localStorage", async () => {
    localStorage.removeItem("strataexec_token");
    const { api } = await import("../api/client");
    const interceptor = getLastRequestInterceptor(api);

    const config = { headers: {} as Record<string, string> };
    const result = interceptor(config);
    expect(result.headers.Authorization).toBeUndefined();
  });
});

// ── Response interceptor ──────────────────────────────────────────────────────

describe("API client — response interceptor", () => {
  beforeEach(() => {
    localStorage.setItem("strataexec_token", "valid_token");
    // Reset window.location to a non-login page
    Object.defineProperty(window, "location", {
      writable: true,
      value: {
        pathname: "/simulator",
        href: "http://localhost/simulator",
        origin: "http://localhost",
        assign: vi.fn(),
        replace: vi.fn(),
      },
    });
  });

  afterEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("passes through successful responses unchanged", async () => {
    const { api } = await import("../api/client");
    const handlers = getLastResponseInterceptor(api);

    const mockResponse = { data: { status: "ok" }, status: 200 };
    const result = handlers.fulfilled(mockResponse);
    expect(result).toBe(mockResponse);
  });

  it("removes token on 401 response", async () => {
    const { api } = await import("../api/client");
    const handlers = getLastResponseInterceptor(api);

    const error = { response: { status: 401 }, message: "Unauthorized" };
    await expect(handlers.rejected(error)).rejects.toBeDefined();
    // Token must be removed from storage
    expect(localStorage.getItem("strataexec_token")).toBeNull();
  });

  it("sets window.location.href to /login on 401 when not already there", async () => {
    const { api } = await import("../api/client");
    const handlers = getLastResponseInterceptor(api);

    const error = { response: { status: 401 }, message: "Unauthorized" };
    await expect(handlers.rejected(error)).rejects.toBeDefined();
    expect(window.location.href).toBe("/login");
  });

  it("does NOT remove token on 500 errors", async () => {
    const { api } = await import("../api/client");
    const handlers = getLastResponseInterceptor(api);

    const error = { response: { status: 500 }, message: "Server Error" };
    await expect(handlers.rejected(error)).rejects.toBeDefined();
    expect(localStorage.getItem("strataexec_token")).toBe("valid_token");
  });

  it("does NOT redirect if already on /login page", async () => {
    Object.defineProperty(window, "location", {
      writable: true,
      value: {
        pathname: "/login",
        href: "http://localhost/login",
        origin: "http://localhost",
        assign: vi.fn(),
        replace: vi.fn(),
      },
    });

    const { api } = await import("../api/client");
    const handlers = getLastResponseInterceptor(api);

    const error = { response: { status: 401 }, message: "Unauthorized" };
    await expect(handlers.rejected(error)).rejects.toBeDefined();
    // href should NOT change because we're already on /login
    expect(window.location.href).toBe("http://localhost/login");
  });
});

// ── Exported functions ────────────────────────────────────────────────────────

describe("API client — exported functions exist", () => {
  it("exports all expected API functions", async () => {
    const client = await import("../api/client");
    const expected = [
      "api",
      "loginUser",
      "fetchMe",
      "fetchDashboard",
      "fetchStrategies",
      "fetchDates",
      "startSimulation",
      "fetchSimulationResult",
      "startEvaluation",
      "fetchEvaluationResult",
      "cancelJob",
      "fetchUploadedModels",
    ];
    for (const fn of expected) {
      expect(fn in client, `Missing export: ${fn}`).toBe(true);
      expect(typeof (client as any)[fn], `${fn} should be a function`).toBe("function");
    }
  });
});
