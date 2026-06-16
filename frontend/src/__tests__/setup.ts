/**
 * Global test setup file.
 * Imported by vitest.config.ts via setupFiles.
 */
import "@testing-library/jest-dom";
import { vi, beforeEach, afterEach } from "vitest";

// Silence React act() warnings in tests — they are expected in test environments
const originalError = console.error.bind(console.error);
beforeEach(() => {
  console.error = (...args: unknown[]) => {
    const msg = String(args[0]);
    if (
      msg.includes("Warning: ReactDOM.render") ||
      msg.includes("act(")
    ) {
      return;
    }
    originalError(...args);
  };
});
afterEach(() => {
  console.error = originalError;
});

// Mock window.location — jsdom doesn't support navigation, and axios requires a valid URL
Object.defineProperty(window, "location", {
  writable: true,
  value: {
    pathname: "/",
    href: "http://localhost/",
    origin: "http://localhost",
    assign: vi.fn(),
    replace: vi.fn(),
  },
});

// Mock localStorage to guarantee it works in tests
const store: Record<string, string> = {};
const localStorageMock = {
  getItem: vi.fn((key: string) => store[key] || null),
  setItem: vi.fn((key: string, value: string) => {
    store[key] = String(value);
  }),
  removeItem: vi.fn((key: string) => {
    delete store[key];
  }),
  clear: vi.fn(() => {
    Object.keys(store).forEach((key) => delete store[key]);
  }),
  length: 0,
  key: vi.fn((index: number) => Object.keys(store)[index] || null),
};
Object.defineProperty(window, "localStorage", {
  value: localStorageMock,
  writable: true,
});
Object.defineProperty(globalThis, "localStorage", {
  value: localStorageMock,
  writable: true,
});

