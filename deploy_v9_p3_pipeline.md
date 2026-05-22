# Deploy V9 + ArcFace P3 Pipeline

## Goal

Run the hardware pipeline on MaixCAM using:

- `face_detect_v9.mud` for face refinement / landmark prediction
- `face_recognize_arcface_p3.mud` for 128D ArcFace embeddings
- Hybrid face database: load saved identities and allow runtime registration

## Runtime Pipeline

```text
Camera frame
→ YOLO face box proposal
→ v9 crop refinement + 5 landmarks
→ 112x112 recognition crop
→ ArcFace P3 embedding
→ L2 normalize
→ cosine distance match
→ overlay identity / distance / threshold
→ save/load face DB JSON
```

## Files To Update

1. `MaixCAM_App/main.py`
   - Replace v3 landmark model with v9.
   - Add P3 recognition model.
   - Add embedding extraction, matching, hybrid DB load/save.
   - Add marker-file registration flow for MaixCAM.

2. `scripts/export/export_recognize_onnx.py`
   - Export P3 checkpoint to ONNX under `models/exports`.
   - Create P3 `.mud` metadata for MaixCAM runtime.

3. `scripts/export/upload_to_maixcam.py`
   - Upload v9 `.mud/.cvimodel`, P3 `.mud/.cvimodel`, YOLO files, and app.

4. `scripts/export/compile_recognize_p3_vps.py`
   - Compile exported P3 ONNX to INT8 `.cvimodel` on VPS.

## Registration Control On Device

Create this file on MaixCAM:

```text
/root/register_name.txt
```

Content example:

```text
Kien
```

The app collects several embeddings, averages them, normalizes the prototype, saves it into:

```text
/root/face_db.json
```

Clear database by creating:

```text
/root/clear_face_db.flag
```
