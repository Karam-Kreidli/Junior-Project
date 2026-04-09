import asyncio
import base64
import json
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_UAE = ROOT / "data_oz" / "uae_domain"
WORMS_CACHE = ROOT / "worms_cache.json"

# ── Species Setup ─────────────────────────────────────────────────────
OZFISH_SPECIES = ["SpiderFish", "TulesOneFish", "TulesTwoFish"]

UAE_SPECIES = [
    "Lethrinus nebulosus",
    "Lutjanus argentimaculatus",
    "Epinephelus coioides",
    "Cephalopholis hemistiktos",
    "Pomacanthus asfur",
    "Scomberomorus commerson",
    "Caranx sexfasciatus",
    "Platax teira",
    "Acanthurus sohal",
    "Stegostoma tigrinum",
]

# ── Global App State ──────────────────────────────────────────────────
app = FastAPI(title="BioReef API")

# Enable CORS for the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DashboardSettings(BaseModel):
    demo_mode: bool
    goggle_mode: str  # "raw" or "restored"
    refresh_rate: float


settings = DashboardSettings(
    demo_mode=True,
    goggle_mode="raw",
    refresh_rate=1.5
)


# ── Taxonomy Cache Loading ────────────────────────────────────────────
def load_taxonomy() -> Dict[str, Dict[str, str]]:
    taxonomy: Dict[str, Dict[str, str]] = {}
    if WORMS_CACHE.exists():
        with open(WORMS_CACHE, "r") as f:
            cache = json.load(f)
        for key, val in cache.items():
            if key.startswith("species:"):
                sp = key.replace("species:", "")
                taxonomy[sp] = val

    # Add placeholders for OzFish missing WoRMS identifiers
    for sp in OZFISH_SPECIES:
        if sp not in taxonomy:
            taxonomy[sp] = {
                "species": sp,
                "genus": "—",
                "family": "OzFish",
                "order": "—",
                "class": "—",
                "phylum": "Chordata",
                "kingdom": "Animalia",
            }
    return taxonomy


TAXONOMY_DB = load_taxonomy()


# ── Demo Engine ───────────────────────────────────────────────────────
def _get_files_with_ext(folder: Path, exts: set) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]


def img_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()


def generate_demo_frame() -> Dict[str, Any]:
    """Generates a mock inference payload."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Check valid folders in uae_domain
    folders = {}
    if DATA_UAE.exists():
        for sp in UAE_SPECIES:
            folder_name = sp.replace(" ", "_")
            folder = DATA_UAE / folder_name
            if folder.exists():
                folders[sp] = folder

    if not folders:
        # Fallback if no images found
        return {
            "image": None,
            "detections": [],
            "source_label": "No Data Found",
            "fps": 0.0,
            "vram": 0.0
        }

    # Pick 1-4 random species
    n_det = random.randint(1, min(4, len(folders)))
    detected_species = random.sample(list(folders.keys()), n_det)

    primary_sp = detected_species[0]
    images = _get_files_with_ext(folders[primary_sp], exts)
    b64_img = None
    if images:
        path = random.choice(images)
        try:
            img = Image.open(path).convert("RGB")
            # If in "restored" mode, we could simulate enhancement.
            # For demo, keeping the same logic but returning the image.
            b64_img = img_to_base64(img)
        except Exception:
            pass

    detections = []
    for sp in detected_species:
        conf = round(random.uniform(0.45, 0.98), 2)
        detections.append({"species": sp, "confidence": conf})

    # Sort descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "image": b64_img,
        "detections": detections,
        "source_label": primary_sp,
        "fps": round(random.uniform(18, 32), 1),
        "vram": round(random.uniform(2.1, 3.8), 2)
    }

# ── API Routes ────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "BioReef Engine"}


@app.get("/api/taxonomy")
def get_taxonomy():
    """Returns the full taxonomy dictionary."""
    return TAXONOMY_DB


@app.get("/api/feed/demo")
def get_demo_feed():
    """Returns the current frame, detections, and metrics."""
    if not settings.demo_mode:
        return {
            "image": None,
            "detections": [],
            "source_label": "Live Mode (Waiting for stream...)",
            "fps": 0.0,
            "vram": 0.0
        }
        
    return generate_demo_frame()


@app.get("/api/settings")
def get_settings():
    """Returns current active settings."""
    return settings.dict()


@app.post("/api/settings")
def update_settings(new_settings: DashboardSettings):
    """Updates active settings from the frontend."""
    global settings
    settings = new_settings
    return {"status": "success", "settings": settings.dict()}

if __name__ == "__main__":
    import uvicorn
    # Start on 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
