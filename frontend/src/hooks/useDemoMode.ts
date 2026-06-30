import { useQuery } from '@tanstack/react-query';

export interface DemoManifest {
  demo_mode: boolean;
  simulation_presets?: Array<{
    key: string;
    label: string;
    description: string;
    tags: string[];
    params_summary: {
      model: string;
      sigma: number;
      lambda: number;
      notional: number;
      horizon: number;
    };
  }>;
  evaluation_dates?: Array<{
    date: string;
    label: string;
    regime: string;
    is_test_date: boolean;
  }>;
  unavailable_features?: Array<{
    feature: string;
    message: string;
    run_locally: string;
  }>;
}

export function useDemoMode() {
  const { data: manifest } = useQuery<DemoManifest>({
    queryKey: ['demo-manifest'],
    queryFn: () =>
      fetch('/api/demo/manifest').then(r => r.json()),
    staleTime: Infinity, // manifest never changes during a session
  });

  const isDemoMode = manifest?.demo_mode ?? false;

  function isFeatureAvailable(feature: string): boolean {
    if (!isDemoMode) return true;
    const unavailable = manifest?.unavailable_features ?? [];
    return !unavailable.some(f => f.feature === feature);
  }

  function getRunLocallyInstructions(feature: string): string | null {
    if (!isDemoMode) return null;
    const info = manifest?.unavailable_features?.find(
      f => f.feature === feature
    );
    return info?.run_locally ?? null;
  }

  return {
    isDemoMode,
    manifest,
    isFeatureAvailable,
    getRunLocallyInstructions,
    simulationPresets: manifest?.simulation_presets ?? [],
    evaluationDates: manifest?.evaluation_dates ?? [],
  };
}
