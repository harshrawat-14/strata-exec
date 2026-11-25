/**
 * The public site with the engine offline: visitors can open every page, each shows static
 * content, and "Take the tour" starts a guided tour that walks across pages.
 */
import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import { render, screen, fireEvent, within } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppRoutes } from "../App";
import { TOUR_STEPS } from "../tour/steps";

function Where() {
  return <div data-testid="path">{useLocation().pathname}</div>;
}

function renderAt(path: string) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
        <Where />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeAll(() => {
  globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver;
  Element.prototype.scrollIntoView ??= () => {};
});

beforeEach(() => {
  localStorage.clear();
  // Engine offline: every request fails.
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));
});

describe("visitors can browse every page while the engine is offline", () => {
  it.each([
    ["/", "Explore"],
    ["/results", "Crash days won"],
    ["/evaluate", "RL agent IS"],
    ["/sweep", "precomputed GARCH sweep, 500 paths per point"],
    ["/upload", "Uploads need the engine, which is offline right now."],
    ["/learn", "How StrataExec"],
  ])("%s opens without a sign-in and shows its content", async (path, text) => {
    renderAt(path);
    expect(await screen.findByText(text, { exact: false })).toBeInTheDocument();
    expect(screen.getByTestId("path").textContent).toBe(path);
  });

  it("shows the engine-down notice on app pages", async () => {
    renderAt("/sweep");
    expect(await screen.findByText("The simulation engine is down")).toBeInTheDocument();
  });
});

describe("results page", () => {
  it("shows fixed headline numbers above the regime filter, which describes each regime", async () => {
    renderAt("/results");
    expect(await screen.findByText("8 / 8")).toBeInTheDocument();
    expect(screen.getByText("1 / 17")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /held-out/i })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: /^crash · 8$/i }));
    expect(screen.getByText(/the costliest third/i)).toBeInTheDocument();
    expect(screen.getByText("8 days")).toBeInTheDocument();
    expect(screen.getByText("8 / 8")).toBeInTheDocument(); // tiles do not move with the filter
  });
});

describe("guided tour", () => {
  // Regression: on the results page the banner's button used to navigate to /results, i.e. nowhere.
  it("starts from the engine-down banner", async () => {
    renderAt("/results");
    const banner = await screen.findByRole("status");
    fireEvent.click(within(banner).getByRole("button", { name: /take the tour/i }));
    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveAccessibleName(`Tour step 1 of ${TOUR_STEPS.length}: The headline`);
  });

  it("explains the regimes, links to Learn, and walks across pages", async () => {
    renderAt("/results");
    fireEvent.click(await screen.findByRole("button", { name: /start the tour/i }));
    fireEvent.click(await screen.findByRole("button", { name: /^next/i }));
    const regimes = await screen.findByRole("dialog");
    expect(within(regimes).getByText(/costliest third/i)).toBeInTheDocument();
    expect(within(regimes).getByRole("link", { name: /learn more/i })).toHaveAttribute("href", "/learn#regimes");

    const simStep = TOUR_STEPS.findIndex(s => s.route === "/simulator");
    for (let i = 1; i < simStep; i++) fireEvent.click(screen.getByRole("button", { name: /^next/i }));
    expect(await screen.findByText("Strategy simulator")).toBeInTheDocument();
    expect(screen.getByTestId("path").textContent).toBe("/simulator");

    fireEvent.click(screen.getByRole("button", { name: /close tour/i }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("opens straight into the tour from a ?tour=1 link", async () => {
    renderAt("/results?tour=1");
    expect(await screen.findByRole("dialog")).toHaveAccessibleName(/Tour step 1 of/);
  });

  it("starts from the login page too", async () => {
    renderAt("/login");
    fireEvent.click(await screen.findByRole("button", { name: /take the tour/i }));
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(screen.getByTestId("path").textContent).toBe("/results");
  });
});
