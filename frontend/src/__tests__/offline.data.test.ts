/** Static fallbacks used while the engine is offline. */
import { describe, it, expect, vi } from "vitest";

vi.mock("../api/client", () => ({
  fetchSimulationResult: vi.fn().mockRejectedValue(new Error("Network Error")),
}));

import { loadSimulationPreset, pinnedSweepResult, presetToResult } from "../api/offline";

describe("simulation presets", () => {
  it("fall back to the static copy when the API fails, shaped like /api/compare", async () => {
    const file = {
      preset_key: "gbm__0.20__0.001__50000__500",
      params: { model: "gbm" },
      results: { strategies: [{ strategy: "TWAP" }], price_path: [100, 101] },
    };
    const f = vi.fn().mockResolvedValue(new Response(JSON.stringify(file), { status: 200 }));
    const r = await loadSimulationPreset(file.preset_key, f as unknown as typeof fetch);
    expect(f).toHaveBeenCalledWith(`/demo/simulation/${file.preset_key}.json`);
    expect(r).toEqual(presetToResult(file as never));
    expect(r.price_path).toEqual([100, 101]);
    expect(r.params_used).toEqual({ model: "gbm" });
  });

  it("report a missing preset instead of rendering nothing", async () => {
    const f = vi.fn().mockResolvedValue(new Response("not found", { status: 404 }));
    await expect(loadSimulationPreset("nope", f as unknown as typeof fetch)).rejects.toThrow(/not found/);
  });
});

describe("pinned sweeps", () => {
  it("cover every dimension the sweep page offers, with 500 paths per point", () => {
    for (const dim of ["volatility", "horizon", "impact", "slices"]) {
      const r = pinnedSweepResult(dim);
      expect(r.grid_values.length).toBeGreaterThan(1);
      const key = dim === "slices" ? "trade_chunks" : dim;
      const rows = (r.sweep_data as Record<string, { NumPaths: number }[]>)[key];
      expect(rows.every(row => row.NumPaths === 500)).toBe(true);
    }
  });
});
