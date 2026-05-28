# Face Similarity Fix: Vector Qua Gan Nhau

## Summary

Loi face similarity qua gan nhau khong den tu model ArcFace P3 bi collapse va cung khong den tu ma hoa/giai ma vector. Nguyen nhan chinh nam o pipeline sinh embedding:

- Web co fallback tao dummy/unit embedding khi ONNX load loi, lam nhieu khuon mat co vector giong nhau.
- Web va Edge dung crop vuong theo landmark thay vi 5-point similarity alignment theo chuan ArcFace/InsightFace.
- Score hien thi `1 - distance` de gay hieu nham nhu phan tram do giong.
- Threshold mac dinh khong dong nhat giua server, web va MaixCAM.
- Server chua chan vector dummy/sparse truoc khi luu/tim trong Qdrant.

Fix hien tai tap trung vao viec loai false positive va loai vector khong dang tin. Khong train lai model trong vong sua nay.

## Evidence

Ket qua dieu tra truoc khi sua:

| Hang muc | Ket qua |
| --- | --- |
| P3 chay truc tiep tren LFW | Same mean khoang `0.0167`, different mean khoang `0.0612` |
| Crypto AES-GCM/XTEA round-trip | Sai so vector gan `0` |
| Qdrant vector cu | 2 vector chi cach nhau `0.024654`, score cu `0.975346` |
| Crop cu | Nen nhieu different pairs xuong gan nguong |

Ket luan: vector luu vao Qdrant da qua gan tu dau do pipeline capture/alignment, khong phai do model hay crypto.

## Runtime Pipeline Sau Khi Sua

```text
Camera/Webcam frame
-> V9 face/landmark detection
-> 5-point similarity alignment ve ArcFace 112x112
-> ArcFace P3 embedding 128D
-> validate finite/dimension/norm/sparse
-> L2 normalize
-> Qdrant upsert/search hoac Edge local match
-> so sanh cosine distance voi threshold 0.020
```

## Web Changes

File chinh:

- `web_stage3/js/components/face-capture.js`
- `web_stage3/js/pages/checkin.js`
- `web_stage3/pages/register-face.html`
- `web_stage3/pages/kiosk.html`

Thay doi:

- Xoa fallback dummy/unit embedding. Neu ONNX V9/P3 khong load duoc thi capture se fail that va UI bi disable.
- `_alignFaceWithLandmarks` dung 5 diem landmark de estimate similarity transform va warp ve crop `112x112`.
- Register/kiosk khong tiep tuc khi model load fail.
- Check-in khong hien thi `1 - distance` nhu phan tram giong. UI dung trang thai match/no-match; distance/threshold chi nen dung cho debug.

## Server And Vector Guard

File chinh:

- `server/vector_service.py`
- `server/face_routes.py`
- `server/admin_routes.py`

Validation embedding hien tai:

- Phai dung `128D`.
- Moi gia tri phai finite.
- Norm phai hop le, khong gan zero.
- Vector duoc L2-normalize truoc khi luu/tim.
- Reject vector sparse/dummy, vi du `[1, 0, 0, ...]`.

Threshold:

- Server threshold mac dinh: `0.020`.
- Client gui threshold cao hon se bi clamp ve toi da `0.020`.
- API `/api/face/match` van tuong thich nguoc, nhung response co them effective threshold de debug.

Admin cleanup:

- Xoa face registration qua admin route se xoa dung Qdrant point, khong chi reset database flag.

## Edge/MaixCAM Changes

File chinh:

- `MaixCAM_App/main.py`
- `MaixCAM_App/config.py`
- `config.json` neu duoc sync len thiet bi/runtime

Thay doi:

- Threshold mac dinh dong bo ve `0.020`.
- `make_recognition_crop` dung ArcFace reference points thay cho crop vuong.
- Uu tien native `image.affine` neu runtime Maix ho tro.
- Neu native affine loi, fallback sang deterministic pixel warp.
- Neu alignment fail thi bo frame, khong quay lai crop vuong cu.

## Data Cleanup Required

Embedding cu sinh tu crop/fallback cu khong con dang tin. Sau khi deploy fix:

1. Xoa cac vector face demo/cu trong Qdrant/cache.
2. Reset trang thai `face_registered` cho booking lien quan.
3. Register lai khuon mat bang pipeline moi.
4. Khong nang threshold de giu vector cu, vi se lam tang false positive.

Trong lan sua nay, Qdrant demo da duoc cleanup va booking demo `2`, `3` da reset de bat buoc re-register.

## Test Checklist

### Model Calibration

- P3 direct tren LFW van phai co same mean thap hon different mean ro rang.
- Aligned crop moi khong duoc nen many different pairs xuong duoi threshold `0.020`.

### Web

- Gia lap ONNX load fail.
- Capture/register/kiosk phai bi chan.
- Khong co request register/match nao duoc gui voi dummy embedding.

### Crypto

- AES-GCM round-trip vector sai so gan `0`.
- XTEA round-trip vector sai so gan `0`.

### API/Qdrant

- Register vector dense hop le thanh cong.
- Vector dummy `[1,0,0...]` bi reject `422`.
- Exact self-match pass.
- Cross-match fail voi threshold `0.020`.
- Client gui threshold lon hon `0.020` bi clamp.

### Edge Simulation

- Sync cache thanh cong.
- Decrypt embedding thanh cong.
- Exact local match pass.
- Cross local match fail.
- HUD/overlay dung distance va threshold, khong dung phan tram `1 - distance`.

## Verification Snapshot

Da chay cac kiem tra sau trong qua trinh sua:

```powershell
python -m py_compile server\vector_service.py server\face_routes.py server\admin_routes.py MaixCAM_App\config.py MaixCAM_App\main.py scripts\tests\test_face_vector_guards.py
node --check web_stage3\js\components\face-capture.js
node --check web_stage3\js\pages\checkin.js
.venv-tpu310\Scripts\python.exe scripts\tests\test_face_vector_guards.py
```

Ket qua calibration nhanh:

| Test | Ket qua |
| --- | --- |
| LFW same mean | `~0.0167` |
| LFW different mean | `~0.0612` |
| New aligned crop under `0.020` | `7/3160` different pairs |
| Old crop under `0.020` | `169/3160` different pairs |

## Operational Notes

- Restart server dang chay sau khi deploy code moi. Neu port cu van chay process cu, API van co hanh vi cu.
- Re-register khuon mat sau khi deploy. Vector cu khong nen tai su dung.
- Deploy lai `MaixCAM_App/main.py` va config len thiet bi MaixCAM, sau do restart app tren thiet bi.
- Khi can tune threshold, dung calibration data that. Khong tang threshold thu cong chi de lam pass mot case bi false reject.

## Files Changed

```text
MaixCAM_App/config.py
MaixCAM_App/main.py
server/admin_routes.py
server/face_routes.py
server/vector_service.py
web_stage3/js/components/face-capture.js
web_stage3/js/pages/checkin.js
web_stage3/pages/kiosk.html
web_stage3/pages/register-face.html
scripts/tests/test_face_vector_guards.py
```
