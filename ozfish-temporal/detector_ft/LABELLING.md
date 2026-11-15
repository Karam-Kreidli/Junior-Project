# Detector labelling guide

We are retraining the fish detector. Each task is about 80 still frames from AIMS videos. Every
frame already has boxes drawn by two AI models. Your job is to make each frame's boxes
**complete and correct**.

The footage is AIMS private data. Do not share the frames outside the team or post them anywhere.

## Setup (once)

1. Start your local CVAT (Docker Desktop running, then http://localhost:8080).
2. Put the task folders you received next to `upload_tasks.py`, inside a folder named `cvat_tasks`.
3. Run `pip install cvat-sdk` (skip it if you already did this for Stage 3).
4. Run `python upload_tasks.py`. It asks for your CVAT login and creates one task per folder, in
   a project called `detector-ft` with the label `fish`, with the AI boxes already loaded.

## The rule that matters most: box EVERY fish

A fish without a box teaches the detector "this is not a fish". One missed fish does more damage
than a slightly loose box.

- Box every fish: every species, tiny and far away, blurry, half out of the frame, and partly
  hidden behind another fish or the bait arm.
- Zoom in (to about 200%) and scan the whole frame, including the dark edges and the far background.
  The AI misses small distant fish most often.
- If you can't tell whether something is a fish, box it if you would count it as one.

## Fixing the AI boxes

- Delete boxes on things that are not fish: rocks, coral, the bait arm or camera, shadows,
  sea urchins and other animals. Rays and sharks ARE fish: keep them.
- Delete duplicates: one box per fish.
- Make each box tight around the visible part of the fish, fins and tail included.
- Overlapping fish each get their own box, even when the boxes overlap.

## When a frame is too hard

If you cannot box every fish in a frame (a pile of fish, too murky), delete the frame: in the
annotation view, click the bin icon on the top bar ("Delete frame"). Never leave a frame
half-labelled. Very crowded frames have already been removed, so this should be rare.

## Sending your work back

For each finished task: open the task, then **Actions → Export task dataset**. Choose format
**CVAT for images 1.1**, leave "Save images" OFF, and name the file after the task
(e.g. `train_003.zip`). Send the zips back.

`test_` tasks are what every result is measured against. Take extra care on them.
