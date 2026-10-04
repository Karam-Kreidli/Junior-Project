# OzFish temporal clips — what arrived, verified 2026-09-12

Location: `Junior-Project/data/results/` — 2,191 station dirs, **10,313 clips, 218.7 GB**.

## Format (verified, not assumed)

Clips are `<station>.<ext>.<start>.mp4`: **120 frames, 25 fps, 1920x1080, MJPEG**, 4.8 s.

**The filename number is the window START, and the labelled frame sits at start+60 — the exact
centre. This holds for 10,313 / 10,313 clips (100%).**

So `A000001_L.avi.12838.mp4` is frames 12838-12957, and frame **12898** is the one carrying the
`frame_metadata.csv` annotation (`Caranx melampygus`, box 657,346,952,619 — confirmed decoded).

MJPEG needs the FFMPEG backend explicitly. `cv2.VideoCapture(path)` bare picks MSMF on Windows and
**fails silently** (`isOpened()` False). Use `cv2.VideoCapture(path, cv2.CAP_FFMPEG)`.

## They are NOT unlabeled

Every clip is centred on an existing human annotation, so each one inherits its species label from
`frame_metadata.csv` by exact join on `station` + `start+60`. No new taxonomy needed, which matters
because problems.md forbids in-house species labelling.

Better still: **49.8% of clips contain more than one labelled frame** (mean 2.45 labelled frames per
clip, mean temporal span 63 of 120 frames). Those are directly usable as tracking supervision
without any propagation step — real annotated motion, e.g.
`G000292_L.avi.10731` has labels at offsets 16,55,56,60,62,63,64,72,75,95,100,101,102.

## Coverage

| | count |
|---|---|
| clips | 10,313 (10,301 usable) |
| labelled frames at clip centres | 12,929 |
| labelled frames physically inside some clip window | 24,727 (38.4% of all 64,385) |
| annotation events (gap > 120 frames) with a clip | 9,234 / 31,407 (29.4%) |
| benchmark classes (>=20 rows) with >=1 clip | **341 / 343** |
| benchmark classes with >=10 clips | 222 |
| benchmark classes with >=20 clips | 148 |

Only 2 benchmark classes get no clip at all: *Pseudanthias evansi*, *Decapterus muroadsi*.

Sampling is roughly one clip per annotation event, not per row (median clip-centres / labelled-frames
per station = 0.19). The remaining 61.6% of labelled frames are single stills with no surrounding
video, same as before.

| group | dirs | clips | GB | bench classes >=1 | unique to group |
|---|---|---|---|---|---|
| A | 150 | 1,687 | 50.1 | 146 | 23 |
| B | 480 | 1,137 | 24.5 | 138 | 32 |
| E | 723 | 1,417 | 23.7 | 8 | 1 |
| G | 838 | 6,072 | 120.4 | **272** | **110** |

**G is the valuable group** — 59% of clips and 110 classes found nowhere else.

## Provenance

**Supplied by AIMS directly**, in response to our outreach. Not sourced from the Pawsey public
portal, and not published there. Pawsey (Greg) did reply and offered a bundled tape-staged
download; that offer is moot and was never taken up.

For the paper's data-availability statement: cite the OzFish DOI (10.25845/5e28f062c5097,
CC BY 3.0 AU) and note that the temporal clips were provided by AIMS on request. Do **not** direct
readers to `storage.pawsey.org.au` for this set — it isn't there.

## This supersedes the Pawsey download

The acquisition problem is **closed**; nothing further is needed from either party.

`download_priority.csv` listed 248 stations worth fetching. **247 of them are already covered**
(only `E000070_L` is not). All 124 stations in `download_manifest.csv` are covered.

Critically, G stations here run **1-432, including 715 dirs below station 370** — exactly the range
missing from Pawsey and the entire subject of the drafted AIMS email.

Consequences:
- Do not send the AIMS email asking for G000001-G000369. We have it.
- Do not send the Pawsey bulk-transfer request. ~470 GB of tape staging is no longer needed.
- `extract_bursts.py` and the burst manifests are obsolete for this purpose. Clips already are
  the burst windows, pre-cut.
- `Junior-Project/G000377_L.avi` (3.0 GB) is a leftover download test. Its 42 labelled frames are
  only 4-covered by clips, so it is not strictly redundant, but it is not worth 3 GB either.

## Data quality

- 12 truncated clips (258 bytes, fail to decode), all group B, listed in `manifests/bad_clips.txt`
  and flagged `ok=0` in `clip_index.csv`. None is re-fetchable from Pawsey (absent from the priority
  list). 0.12% loss, ignore them.
- 2.38% of `frame_metadata.csv` boxes fall outside 1920x1080 (some negative). Pre-existing dataset
  issue, not a clip problem. Clamp boxes before cropping.
- Stereo `_L`/`_R` are **not** redundant here, unlike the burst analysis: 901 of 903 two-sided
  stations have completely disjoint clip sets, because each side was sampled at its own annotation
  events. Treat them as independent footage.

## Artifacts written

- `manifests/clip_index.csv` — one row per clip: station, clip, start, center, n_label_frames,
  label_offsets, n_boxes, species, ok. Built by `build_clip_index.py`.
- `manifests/bad_clips.txt` — the 12 to exclude.
