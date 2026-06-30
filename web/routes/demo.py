from fastapi import APIRouter
from pathlib import Path
import json
import os

router = APIRouter()
DEMO_DIR = Path(__file__).parent.parent.parent / "demo_results"
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


@router.get("/api/demo/manifest")
async def get_demo_manifest():
    """
    Returns the demo mode manifest: available presets, dates,
    and which features require running locally.
    Used by frontend to populate dropdowns and show alerts.
    """
    if not DEMO_MODE:
        return {"demo_mode": False}

    manifest_path = DEMO_DIR / "manifest.json"
    if not manifest_path.exists():
        return {
            "demo_mode": True,
            "error": "manifest not found — run precompute script",
        }
    return json.loads(manifest_path.read_text())
