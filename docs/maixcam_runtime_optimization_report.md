# MaixCAM Runtime Debugging and Optimization Report

Updated: 2026-05-27

Scope: `MaixCAM_App/main.py`, `MaixCAM_App/mjpeg_server.py`, `MaixCAM_App/config.py`, `MaixCAM_App/config.json`

## 1. Executive Summary

During the MaixCAM face-recognition demo, the system initially became heavily laggy whenever a person entered the camera frame. The main symptoms were a stream that dropped to roughly 2-3 FPS, occasional VENC buffer issues, and a main loop blocked by AI inference, file I/O, and network fallback calls.

The fix was implemented in several stages:

1. Reduced SD card/file I/O inside the camera loop.
2. Decoupled MJPEG stream FPS from AI pipeline FPS.
3. Restarted/rebooted cleanly when needed to release stuck VENC buffers.
4. Added AI frame skipping and overlay caching.
5. Split ArcFace P3 recognition cadence from YOLO/v9 detection cadence.
6. Removed server fallback calls from the realtime recognition hot path.

Current result:

- The MaixCAM app runs stably after reboot.
- MJPEG stream returns frames at `http://10.154.36.1:8080/`.
- VENC errors in the final verification: `0`.
- Server fallback is no longer called repeatedly inside the camera loop.
- When server `10.154.36.100:8000` times out, camera and stream continue running.

## 2. Initial Symptoms

### 2.1 Lag When a Face Was Present

When no face was visible, camera/stream behavior looked more normal. Once a face entered the frame, the loop in `main.py` had to run the full sequence:

```text
YOLO face detection
-> v9 landmark detection
-> ArcFace P3 embedding
-> cache/server match
-> draw overlay
-> update MJPEG frame
```

Because all of these steps ran in the same thread as `cam.read()`, slow AI inference directly slowed camera capture. The stream only received new frames at the speed of the AI loop, which produced the 2-3 FPS feel.

### 2.2 MJPEG Stream Was Coupled to AI

Before the fix, the stream only received a new frame after the AI loop finished processing. If AI took 200-300 ms per frame, MJPEG also produced new frames at that slower cadence.

### 2.3 VENC Crash / Stuck Buffer

At times the log showed:

```text
CVI_VENC_GetStream failed with 0xc0078012
```

This is a MaixCAM encoder/camera-layer issue. During debugging, the most reliable recovery path was to kill the app, wait for buffers to be released, and reboot the device when a hot restart was not enough.

### 2.4 Network Timeout in the Hot Path

When local cache matching missed or cache/server state was not ready, `cache_mgr.match()` could fallback to server endpoint `/api/face/match`. In practice, server `http://10.154.36.100:8000` was timing out, which blocked the recognition loop until the HTTP timeout expired.

## 3. Root Cause

### 3.1 Too Much AI Work Per Frame

All three models are loaded at startup:

- `FACE_DET`: YOLOv8/YOLO11 face detector.
- `LM_MODEL`: v9 landmark model.
- `RECOG_MODEL`: ArcFace P3 model.

The code does not unload these models after each detection. The model objects remain alive in runtime until the app is killed or restarted. The issue was not model reloads from storage; the issue was calling all three models too frequently in the same camera loop.

### 3.2 One Thread Did Too Much Work

The main loop was responsible for everything:

```text
read camera
check flag/config files
sync cache when scheduled
detect face
run landmarks
extract embedding
match identity
draw overlay
push frame to MJPEG
```

Any slow step slowed the whole loop.

### 3.3 Server Fallback Was Not Suitable for the Realtime Loop

Network fallback is useful for accuracy and wider matching, but it should not run directly inside the camera loop. A 5-second HTTP timeout can visibly freeze the stream.

## 4. Changes Implemented

### 4.1 Reduced Per-Frame File I/O

Previously the app frequently checked files such as `active_flight.txt`, registration requests, clear flags, and sync flags. This was changed to throttle file I/O with `IO_CHECK_INTERVAL = 1.0`, so disk-based checks happen at most once per second.

Impact:

- Less SD card access in the hot path.
- Lower jitter caused by file I/O.

### 4.2 Decoupled MJPEG Stream Cadence From AI Cadence

`mjpeg_server.py` was updated so the stream has its own FPS limit, defaulting to 15 FPS. The server repeats the latest available frame if the AI pipeline has not produced a new one yet.

Added config:

```json
"stream_fps_limit": 15,
"stream_jpeg_quality": 70
```

Impact:

- MJPEG clients receive frames more steadily.
- Stream delivery is no longer directly tied to AI pipeline speed.

### 4.3 Added AI Frame Skipping

Added config:

```json
"ai_frame_interval": 3
```

In `main.py`, AI runs only when:

```python
frame_idx % AI_FRAME_INTERVAL == 0
```

`cam.read()` and `mjpeg_server.update_frame(img)` still run every loop. Frames that do not run AI redraw the cached overlay.

Impact:

- Fewer YOLO/v9/ArcFace calls.
- Camera reads continue more smoothly.
- Stream feels smoother.

### 4.4 Added Overlay Cache

Added `overlay_cache` to store:

- Face bounding box.
- Crop box.
- Landmarks.
- v9 score.
- Match/local/no-match result.

Config:

```json
"overlay_cache_ttl_sec": 1.0
```

If the current frame does not run AI, the app redraws the overlay from cache. If the cache exceeds its TTL, the overlay is dropped to avoid showing stale identity information for too long.

### 4.5 Split ArcFace P3 Cadence From YOLO/v9 Cadence

After basic frame skipping was stable, the next optimization was to avoid running ArcFace P3 every time YOLO/v9 ran.

Added config:

```json
"recognition_frame_interval": 9,
"recognition_cache_ttl_sec": 1.5
```

Current logic:

- YOLO/v9 runs according to `ai_frame_interval`.
- ArcFace P3 runs only when:
  - local face registration is active, or
  - no recognition cache exists, or
  - recognition cache has expired, or
  - `recognition_frame_interval` is reached.
- If P3 does not need to run, the app reuses the most recent recognition result for display.

Impact:

- Face box and landmarks are still updated reasonably often.
- P3 embedding and matching run less frequently.
- The heaviest load when a person stands in front of the camera is reduced.

### 4.6 Removed Server Fallback From the Hot Path

Inside the camera loop, the code changed from:

```python
cache_mgr.match(embedding, active_flight)
```

to:

```python
cache_mgr.match_local(embedding, active_flight)
```

Reason: `match()` tries local cache first, but falls back to `match_server()` on a miss. When the server times out, the camera loop becomes very slow.

After the change:

- The hot path only matches local cache/RAM cache.
- Server fallback is no longer called repeatedly while rendering camera frames.
- Final verification showed fallback log count = `0`.

## 5. Current Pipeline

The realtime pipeline now behaves like this:

```text
Each loop:
  cam.read()
  if I/O interval elapsed:
    read active_flight / sync flag / registration request

  if frame is not scheduled for AI:
    draw overlay_cache if still valid

  if frame is scheduled for AI:
    YOLO detect face
    if no face:
      reset EMA + recognition cache
      draw "No face"
    if face exists:
      crop by bbox
      run v9 landmarks
      smooth landmarks with EMA
      if score is valid:
        build lm_abs
        create new overlay_cache
        if recognition should run:
          crop 112x112
          run ArcFace P3 embedding
          match local cache or local face_db
          update recognition_cache
        if recognition can be skipped:
          redraw recognition_cache

  draw HUD
  disp.show if LCD is enabled
  mjpeg_server.update_frame(img)
  sleep 0.005
```

## 6. Current Runtime Configuration

In `MaixCAM_App/config.py` and `/root/config.json` on the device:

```json
{
  "stream_fps_limit": 15,
  "stream_jpeg_quality": 70,
  "ai_frame_interval": 3,
  "recognition_frame_interval": 9,
  "overlay_cache_ttl_sec": 1.0,
  "recognition_cache_ttl_sec": 1.5
}
```

Meaning:

- `ai_frame_interval = 3`: YOLO/v9 runs every 3 camera frames.
- `recognition_frame_interval = 9`: ArcFace P3 runs less often, every 9 camera frames or when the recognition cache expires.
- `overlay_cache_ttl_sec = 1.0`: face box/landmark overlay is kept for up to 1 second.
- `recognition_cache_ttl_sec = 1.5`: recognition result is reused for up to 1.5 seconds.

If faster recognition response is needed, reduce `recognition_frame_interval` to 6 or 3. If smoother streaming is more important, increase it to 12 or 15.

## 7. Deployment and Verification on MaixCAM

Uploaded files to `/root/`:

- `main.py`
- `mjpeg_server.py`
- `config.py`
- `config.json`

The device was restarted/rebooted when needed to release camera/VENC buffers.

Final verification:

```text
Process: python -u /root/main.py
PID: 622
VENC errors: 0
Port 8080: LISTEN
MJPEG probe: HTTP 200 OK, contains boundary --frame
Fallback log count: 0
```

Stream:

```text
http://10.154.36.1:8080/
```

Note: startup cache sync still times out against:

```text
http://10.154.36.100:8000
```

However, after removing server fallback from the hot path, this timeout no longer slows the realtime camera loop.

## 8. Remaining Risks and Open Work

### 8.1 Startup Sync Is Still Blocking

The app still performs cache sync at startup before entering the camera loop. If the server times out, startup is delayed. Recommended next steps:

- Move initial sync to a background thread, or
- start camera/stream first and sync later, or
- reduce sync timeout for edge demos.

### 8.2 No On-Device Timing Profiler Yet

Add timing logs for each stage:

```text
cam.read
YOLO
v9
P3
match_local
draw/update_frame
```

Once measured, interval values can be tuned based on real device timings instead of guesses.

### 8.3 Camera and AI Are Still in One Thread

The app still uses a single main loop. Frame skipping reduces load, but a stronger architecture would be:

```text
Camera/render thread:
  cam.read -> latest_frame -> stream

AI worker:
  latest_frame -> infer -> latest_overlay
```

This could improve smoothness further, but it needs careful testing with Maix Python, the camera object, and the NN runtime to avoid native crashes.

### 8.4 Server Fallback Should Become Async or Manual

Server fallback should not run in the camera loop. Better options:

- Background queue.
- Rate limit to once every few seconds.
- Trigger only after repeated local-cache misses.
- Manual/user-triggered fallback.

## 9. Conclusion

The debugging process showed that the bottleneck was not model unload/reload from storage. The real issue was too much heavy work inside the same camera loop. The implemented changes moved the pipeline from "run all 3 models every AI frame" to "run scheduled stages, reuse short-lived caches, and avoid network in the hot path."

The current state is suitable for demo:

- Stream is running.
- VENC is clean.
- AI frame skipping is enabled.
- ArcFace P3 has its own throttle.
- Server timeout no longer freezes the camera loop.

The next recommended improvements are background sync and a timing profiler before attempting deeper threading or model-level changes.
