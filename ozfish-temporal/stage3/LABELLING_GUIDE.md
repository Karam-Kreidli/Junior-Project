# Stage 3 labelling guide

How to label the OzFish clips in CVAT for the Stage 3 temporal model. Read once before your
first clip.

## Which clips

`stage3_clips/` has one folder per species. Each file starts with its labelling rank:
`001__G000228_L.avi.47513.mp4`. **Label each species in file-name order.** Order matters: the
first clips are the ones the model can be tested on.

- `001`–`050`: **round 1**, label these now (all of them if the species has fewer).
- `051`–`100`: rounds 2 and 3, only when told to.
- `101` and up: never label; these are for a baseline test only.

Progress for a species = the highest rank labelled. `clip_paths.csv` maps every file to its
station, rank, round and fold.

## The task

Each CVAT task is one clip. The clip's species is its folder name; a `RARE.txt` file in the
folder means it is a rare species (Rule 2 applies). The fish to follow are
pre-loaded as boxes on **frame 60**: these are fish an OzFish expert identified. Your job is
to follow each one, as a track, for as long as it is visible.

How many are pre-loaded is decided for you, never by the labeller:
- **Abundant species:** exactly one, picked at random when the expert boxed several.
- **Rare species:** all expert boxes, up to **5**, picked at random when there were more.

You are not identifying species. Every fish you label in a clip is the clip's species.

## CVAT setup

**Once, per person:** create a project. In the label constructor, open the **Raw** tab and paste
the contents of `stage3_clips/cvat_labels.json`. That gives two labels:
- `fish`: the tracks you draw. Attribute `source` (`expert` / `inferred`), set once per track.
- `reference`: the expert's box, pre-loaded. Never draw or edit these.

**Per clip:**
1. Create a task in the project. Name it exactly as the video file (e.g. `001__G000228_L.avi.47513`).
   Upload that `.mp4`. Leave image quality at 100 if you can.
2. Open the task, **Actions → Upload annotations**, format **CVAT 1.1**, and pick the `.xml`
   with the same name, next to the video.
3. Go to **frame 60** (CVAT counts from 0). A red `reference` box sits on the fish to follow.
   If it is clearly not on a fish, stop and report the clip; don't guess.
4. Draw a `fish` **Track** (not Shape) per Rule 1, starting where the fish first appears and
   passing through the reference box on frame 60.
5. When done, **Actions → Export task dataset**, format **CVAT for video 1.1**, and save the
   export next to the video as `<same name>_done.zip`. That export is what counts as labelled.

## Rule 1 — follow the fish, no frame limit

1. Start at the pre-loaded box on frame 60. Track **backwards** to the first frame the fish is
   visible, then **forwards** to the last.
2. Draw the box tight around the visible fish. Place keyframes as needed: CVAT interpolates
   between them; add a keyframe wherever the interpolated box slides off the fish.
3. Partly hidden but still clearly the same fish → keep tracking, tick **occluded**.
4. The fish leaves the frame, or you are no longer sure it is the same fish → mark the track
   **outside** and stop.
5. **Never pick a fish back up after it leaves.** A fish swimming back in may be a different
   individual. If it returns, leave it unlabelled.
6. Don't stretch a track through a stretch you're unsure about. A shorter correct track is
   worth more than a longer wrong one.

## Rule 2 — extra fish (rare species only)

Only in species folders that contain `RARE.txt`:

1. Every pre-loaded box is an expert fish (`source = expert`). Track each one per Rule 1.
2. You may also track **other** fish in the clip, but only if you can clearly see it is the
   same species. Set `source = inferred` on these.
   **Exception:** if 5 fish are pre-loaded, the clip is a school; don't add any more.
3. If in doubt, skip it. Close relatives can look almost identical (*Chromis*, *Lethrinus*,
   *Pentapodus*); "probably the same" is not enough.
4. Skip dense schools of look-alikes entirely.

In folders **without** `RARE.txt`, track only the pre-loaded fish. Do not add extra fish.

## Why these rules exist

- Test scores count each **clip** once (averaged over its fish), so a 26-fish school can't
  outweigh 25 other clips. Extra fish still help training, but past ~5 from one moment they add
  little and cost hours.

- `inferred` fish are used for training only, never for testing, and can be removed in one step
  if they turn out to hurt. That only works if the tag is set honestly.
- Re-acquiring a fish, or guessing a species, puts wrong labels into the data that nobody can
  find later.
- Track length varies from clip to clip; that is expected and handled in training.

## Record while you label (pilot clips only)

For the first ~100 clips, note per clip: minutes spent, and anything unusual (fish lost early,
dense school, bad visibility). This sets the time estimate for the rest of the job.
