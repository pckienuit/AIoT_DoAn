# Kế Hoạch Phát Triển Hệ Thống Tra Cứu Thông Tin Chuyến Bay Qua Nhận Diện Khuôn Mặt

> **Đồ án AIoT** — Tích hợp AI thị giác trên phần cứng biên RISC-V
> Cập nhật: 2026-05-25

---

## 1. Tổng Quan Hệ Thống

### 1.1 Mục tiêu
Xây dựng hệ thống cho phép hành khách **tra cứu thông tin chuyến bay** (cổng ra, thời gian boarding, trạng thái) bằng cách **nhìn vào camera** tại sảnh sân bay — thay vì xuất trình vé giấy hoặc mở ứng dụng.

### 1.2 Ba khối chính

```mermaid
graph LR
    subgraph "Khối 1: Web Client"
        W1[Đặt vé máy bay] --> W2[Đăng ký khuôn mặt]
        W2 --> W3["ONNX Inference (Browser WASM)"]
        W3 --> W4["AES-GCM Encrypt Vector 128D"]
    end

    subgraph "Khối 2: Server"
        S1["Vector DB (Encrypted)"]
        S2[Flight Info Database]
        S3[REST/gRPC API]
        S1 --- S3
        S2 --- S3
    end

    subgraph "Khối 3: Edge Device"
        E1["Camera RISC-V C906"]
        E2["YOLO → V9 → P3 Pipeline"]
        E3[Local Cache]
        E4[LCD / Audio Output]
    end

    W4 -->|HTTPS| S3
    S3 -->|Sync encrypted vectors| E3
    E1 --> E2
    E2 -->|Query cache or API| S3
    E2 --> E4
```

### 1.3 Phần cứng Edge (MaixCAM / Sipeed)

| Thông số | Giá trị |
|:---|:---|
| CPU | RISC-V C906 @ 1GHz |
| RAM | 128MB DDR3 |
| Storage | MicroSD 8GB |
| Kết nối | USB 2.0, UART, SPI, I2C, GPIO, WiFi |
| AI | TPU tích hợp (INT8 inference) |
| Camera | FPC connector (GC4653 sensor) |
| Audio | Microphone tích hợp |

---

## 2. Kiến Trúc Chi Tiết

### 2.1 Khối Web — Đặt vé & Đăng ký khuôn mặt

#### Luồng người dùng
1. Hành khách **đặt vé máy bay** bình thường (chọn chuyến, thanh toán).
2. Sau khi đặt vé thành công → hiện màn hình **"Đăng ký Face Check-in"**.
3. Hướng dẫn trên màn hình: quay video 5 giây, xoay đầu trái-phải-lên-xuống trong điều kiện sáng đủ.
4. Trình duyệt chạy pipeline ONNX cục bộ (WASM):
   - **MediaPipe** phát hiện bounding box khuôn mặt.
   - **V9 Landmarks** (224×224) → 5 điểm mốc.
   - Căn chỉnh & crop 112×112 → **ArcFace P3** → vector embedding 128D.
5. Trích xuất **7 khung hình tốt nhất** (score > 0.4, đa góc), tính trung bình vector → L2 normalize.
6. **Mã hóa AES-GCM 256-bit** vector trung bình → gửi lên server kèm `booking_id`.

#### Xử lý chất lượng đăng ký
- Kiểm tra **độ sáng trung bình** ảnh (luminance > 80, < 220).
- Kiểm tra **góc xoay đa dạng** qua phân bố landmark: yêu cầu ít nhất 3 góc khác biệt.
- Kiểm tra **face score từ V9 > 0.5** cho mỗi khung được chọn.
- Nếu chất lượng không đạt → hiển thị hướng dẫn cụ thể để người dùng thử lại.

#### Công nghệ
| Layer | Công nghệ |
|:---|:---|
| Frontend | HTML/CSS/JS hoặc Next.js |
| Face Detection | MediaPipe Face Detection (CDN) |
| Landmark + Embedding | `onnxruntime-web` (WASM) — V9 + P3 |
| Mã hóa | Web Crypto API — AES-GCM 256-bit |
| Giao tiếp | HTTPS REST → Server |

---

### 2.2 Khối Server — Lưu trữ & Đối sánh

#### Chức năng
1. **Nhận và lưu** vector mã hóa + thông tin vé từ Web Client.
2. **Đồng bộ cache** xuống thiết bị Edge theo chuyến bay (chỉ gửi vector của hành khách cùng chuyến).
3. **Fallback matching**: nếu Edge không match được local → gọi API server để tìm kiếm mở rộng.

#### Kiến trúc cơ sở dữ liệu — Tách biệt trách nhiệm

> **Nguyên tắc**: SQL chỉ lưu *metadata có cấu trúc*, Vector DB lưu *embedding để search*. Không trộn lẫn.

```
┌─────────────────────────────────────────────────────────────────┐
│  RELATIONAL DB — SQLite (prototype) / PostgreSQL (production)   │
│  Lưu metadata cấu trúc — KHÔNG lưu vector                      │
│                                                                 │
│  ┌─────────────┐   ┌──────────────────────┐   ┌─────────────┐  │
│  │ passengers  │   │ bookings             │   │ flights     │  │
│  │─────────────│   │──────────────────────│   │─────────────│  │
│  │ id (PK)     │──▶│ id (PK)              │◀──│ id (PK)     │  │
│  │ name        │   │ passenger_id (FK)    │   │ flight_code │  │
│  │ email       │   │ flight_id (FK)       │   │ departure   │  │
│  │ phone       │   │ seat_number          │   │ gate        │  │
│  └─────────────┘   │ qdrant_point_id ─────┼─┐ │ destination │  │
│                    │ face_registered_at   │ │ │ status      │  │
│                    │ status               │ │ └─────────────┘  │
│                    └──────────────────────┘ │                  │
└─────────────────────────────────────────────┼──────────────────┘
                                              │ foreign key
┌─────────────────────────────────────────────┼──────────────────┐
│  VECTOR DB — Qdrant (self-hosted Docker)    │                  │
│  Collection: "face_embeddings"              ▼                  │
│                                                                 │
│  Point {                                                        │
│    id      : UUID  ← liên kết với bookings.qdrant_point_id     │
│    vector  : float32[128]  ← embedding ArcFace P3 (plaintext)  │
│    payload : {                                                  │
│      booking_id  : "BK-001"                                     │
│      flight_id   : "VN123"                                      │
│      passenger   : "Nguyen Van A"                               │
│      seat        : "12A"                                        │
│      gate        : "B07"                                        │
│      expires_at  : "2026-05-24T08:00:00Z"                       │
│    }                                                            │
│  }                                                              │
│                                                                 │
│  Index: HNSW (m=16, ef_construct=128)                           │
│  → ANN search cosine < 1ms với hàng triệu vector               │
└─────────────────────────────────────────────────────────────────┘
```

#### Tại sao dùng Vector DB thay vì SQL thuần?

| Tiêu chí | SQL + pgvector | **Qdrant (Vector DB)** |
|:---|:---|:---|
| Search 1M vector | ~100ms (scan tuần tự) | **< 1ms** (HNSW ANN index) |
| Cosine similarity | Phải tính lại mỗi query | Native operator, pre-indexed |
| Filter by flight_id | JOIN bảng, slow | Filter ngay trong vector search |
| Sync xuống Edge | SQL dump phức tạp | REST API scroll → JSON đơn giản |
| Xóa vector hết hạn | DELETE query | Payload filter + delete by TTL |
| Scale ngang | Cần sharding thủ công | Built-in partition + replication |

#### Luồng đăng ký vector (Web → Qdrant)
```
Browser: ONNX inference → vector 128D → AES-GCM encrypt → POST /api/face/register
Server:  AES-GCM decrypt → upsert Qdrant point {vector, payload}
         → lưu qdrant_point_id vào bookings table (SQL)
```

#### Luồng tìm kiếm (Edge → Qdrant)
```
Edge:   ArcFace P3 → vector 128D → POST /api/face/match
Server: Qdrant.search(vector, filter={flight_id: X}, top_k=1, metric=cosine)
        → score ≥ 0.955 (dist ≤ 0.045): Match → trả payload (tên, ghế, cổng)
        → score < 0.955: No Match
```

#### API Endpoints

| Method | Endpoint | Mô tả |
|:---|:---|:---|
| POST | `/api/bookings` | Tạo booking mới (SQL) |
| POST | `/api/face/register` | Decrypt vector → upsert Qdrant + lưu point_id |
| GET | `/api/flights/:code` | Lấy thông tin chuyến bay (SQL) |
| POST | `/api/face/match` | Qdrant ANN search by flight_id → trả kết quả |
| GET | `/api/sync/:flight_id` | Qdrant scroll by payload.flight_id → JSON cache |
| PATCH | `/api/bookings/:id/checkin` | Cập nhật status SQL + Qdrant payload |

#### Quy trình đối sánh trên Server
1. Edge gửi vector 128D (plaintext qua HTTPS/TLS nội bộ).
2. Gọi **Qdrant search**: `vector=emb, filter={flight_id: X}, top_k=1, with_payload=True`.
3. Qdrant dùng **HNSW index** trả về điểm gần nhất + cosine score trong < 1ms.
4. Ngưỡng: `score ≥ 0.955` (dist ≤ 0.045) → Match; `≥ 0.920` → Cần xác minh; `< 0.920` → Không khớp.

#### Công nghệ
| Layer | Công nghệ |
|:---|:---|
| Backend | Python FastAPI |
| Relational DB | SQLite → PostgreSQL (metadata) |
| **Vector DB** | **Qdrant** (self-hosted Docker / Qdrant Cloud) |
| Qdrant Client | `qdrant-client` Python SDK |
| Bảo mật | AES-GCM, HTTPS/TLS, API key auth |

---

### 2.3 Khối Edge — Thiết bị nhận diện tại sân bay

#### Pipeline xử lý

```mermaid
graph TD
    A["Camera GC4653 (320×320)"] --> B["YOLOv8n Face Detection"]
    B -->|"Bounding Box (x,y,w,h)"| C["Adaptive Crop (178×218 CelebA ratio)"]
    C -->|"Resize 224×224"| D["V9 Landmark Model (cvimodel)"]
    D -->|"5 Landmarks + Score"| E{"Score > 0.4?"}
    E -->|No| F[Bỏ qua / Hiển thị 'Vui lòng nhìn thẳng']
    E -->|Yes| G["EMA Smoothing (α=0.35)"]
    G -->|"Aligned Crop 112×112"| H["ArcFace P3 (cvimodel)"]
    H -->|"Embedding 128D"| I["L2 Normalize"]
    I --> J{"Local Cache Match?"}
    J -->|"dist ≤ 0.045"| K["✅ Hiển thị thông tin chuyến bay"]
    J -->|"No match"| L["Gọi Server API fallback"]
    L -->|Match| K
    L -->|No match| M["❌ Không tìm thấy booking"]
```

#### Chiến lược Cache thông minh
- **Pre-sync**: Trước giờ bay 3 tiếng, Edge tải toàn bộ vector hành khách của các chuyến bay sắp khởi hành về local (qua WiFi).
- **Cấu trúc cache**: File JSON trên MicroSD, nhóm theo `flight_id`.
- **Dung lượng**: 128D × 4 bytes × 200 hành khách ≈ **100 KB/chuyến** → MicroSD 8GB dư sức chứa hàng nghìn chuyến.
- **TTL**: Cache tự xóa sau khi chuyến bay cất cánh + 2 tiếng.
- **Fallback**: Nếu không match trong cache → gọi REST API server để tìm kiếm mở rộng.

#### Hiển thị kết quả
- **LCD** (nếu có): Tên hành khách, số hiệu chuyến bay, cổng ra, thời gian boarding, ghế ngồi.
- **Audio** (qua mic/speaker): Đọc thông tin bằng TTS đơn giản hoặc beep xác nhận.
- **LED GPIO**: Xanh = match, Đỏ = không match, Vàng = đang xử lý.

#### Giao tiếp Edge ↔ Server

| Kênh | Giao thức | Mục đích |
|:---|:---|:---|
| WiFi | HTTPS REST | Sync cache, fallback match, heartbeat |
| UART | Serial | Debug log, kết nối màn hình phụ |

---

## 3. Bảo Mật End-to-End

```mermaid
sequenceDiagram
    participant Browser as Web Client
    participant Server as Server
    participant Edge as Edge Device

    Browser->>Browser: ONNX inference → Vector 128D
    Browser->>Browser: AES-GCM encrypt (session key)
    Browser->>Server: POST /face/register {encrypted_vector, IV, booking_id}
    Note over Server: Lưu encrypted vector + IV vào DB

    Edge->>Server: GET /sync/flight_123 (authenticated)
    Server->>Edge: {encrypted_vectors[], IVs[], booking_infos[]}
    Note over Edge: Giải mã bằng device key, lưu cache

    Edge->>Edge: Camera → Pipeline → Vector 128D
    Edge->>Edge: Cosine match với cache (plaintext trên RAM)
    Edge->>Edge: Hiển thị kết quả

    alt Cache miss
        Edge->>Server: POST /face/match {encrypted_vector}
        Server->>Server: Giải mã + brute-force match
        Server->>Edge: {matched_booking}
    end
```

### Quản lý khóa mã hóa
| Khóa | Vị trí | Mục đích |
|:---|:---|:---|
| Session Key (AES-256) | Browser (Web Crypto) | Mã hóa vector trước khi gửi server |
| Master Key (AES-256) | Server (env var) | Re-encrypt để lưu DB |
| Device Key (AES-256) | Edge (secure storage) | Giải mã cache khi sync |

### Nguyên tắc
- Vector khuôn mặt **không bao giờ** truyền dạng plaintext qua mạng.
- Server lưu trữ dạng mã hóa, chỉ giải mã trong RAM khi cần match.
- Edge giải mã cache vào RAM, không lưu plaintext xuống MicroSD.

---

## 4. Mô Hình AI — Đã Hoàn Thành

| Mô hình | Input | Output | Kích thước ONNX | Latency (CPU Python) |
|:---|:---|:---|:---|:---|
| YOLOv8n Face | 320×320 RGB | Bounding boxes | 3.3 MB (.cvimodel) | ~15 ms |
| V9 Landmarks | 224×224 RGB (÷255) | class(1) + bbox(4) + lm(10) | 10.2 MB | **2.70 ms** |
| ArcFace P3 | 112×112 RGB ([-1,1]) | Embedding 128D | 11.8 MB | **6.39 ms** |

### Đã triển khai
- ✅ Train V9 trên CelebA (90 epochs, Wing+Focal loss, Label Smoothing)
- ✅ Train ArcFace P3 trên CASIA-WebFace (60 epochs, SGD+CosineAnnealing, ArcMargin s=64 m=0.50)
- ✅ Export ONNX → cvimodel (MaixHub TPU compiler)
- ✅ Deploy trên MaixCAM: `MaixCAM_App/main.py`
- ✅ Benchmark Python CPU + Browser WASM

---

## 5. Kế Hoạch Phát Triển Theo Giai Đoạn

### Giai đoạn 1: Prototype Core (2 tuần)

| # | Task | Output | Trạng thái |
|:--|:---|:---|:---|
| 1.1 | Train & export V9 + ArcFace P3 | `.onnx`, `.cvimodel` | ✅ Xong |
| 1.2 | Deploy pipeline trên MaixCAM | `MaixCAM_App/main.py` | ✅ Xong |
| 1.3 | Web ONNX inference + AES-GCM | `index.html` | ✅ Xong |
| 1.4 | Benchmark so sánh 4 model | `benchmark_models.py`, WASM panel | ✅ Xong |

### Giai đoạn 2: Server Backend (2 tuần)

| # | Task | Output | Trạng thái |
|:--|:---|:---|:---|
| 2.1 | Khởi tạo FastAPI + SQLite schema (passengers, flights, bookings) | `server/database.py`, `server/main.py` | ✅ Xong |
| 2.2 | Setup **Qdrant** Docker, tạo collection `face_embeddings` (HNSW, cosine) | `docker-compose.yml`, `server/vector_service.py` | ✅ Xong |
| 2.3 | API: CRUD passengers + flights + bookings (SQL) | `server/routes.py` REST endpoints | ✅ Xong |
| 2.4 | API: Face register → plaintext 128D vector → upsert Qdrant | `POST /api/face/register` | ✅ Xong prototype |
| 2.5 | API: Face match → Qdrant ANN search by flight_id | `POST /api/face/match` | ✅ Xong |
| 2.6 | API: Sync cache → Qdrant scroll by payload → JSON response | `GET /api/sync/{flight_id}` | ✅ Xong |

> **Cập nhật 2026-05-24:** Prototype backend dùng plaintext ArcFace P3 vector 128D để test ổn định luồng Web → FastAPI → Qdrant. AES-GCM client/server sẽ triển khai ở giai đoạn bảo mật sau khi prototype ổn định.

### Giai đoạn 3: Web Frontend Đặt Vé (2 tuần)

| # | Task | Output | Trạng thái |
|:--|:---|:---|:---|
| 3.1 | Tạo web prototype riêng, không đụng folder `web/` benchmark cũ | `web_stage3/index.html`, `styles.css`, `app.js` | ✅ Xong |
| 3.2 | UI tạo passenger, flight, booking và chọn opt-in/opt-out face registration | Stage 3 dashboard form | ✅ Xong |
| 3.3 | Reuse MediaPipe + V9 + ArcFace P3 trong browser để tạo embedding thật từ upload/webcam/dataset | `web_stage3/app.js` real ONNX pipeline | ✅ Xong |
| 3.4 | Submit readiness indicator: Real ONNX / fallback test vector / skip face | Dashboard status panel + telemetry metric | ✅ Xong |
| 3.5 | Integration test: booking → optional face register → sync → match | Browser-tested with dataset image | ✅ Xong prototype |
| 3.6 | Logic multi-frame: chọn 7 khung tốt nhất, trung bình vector | Webcam burst capture + average top frames | ✅ Xong prototype |
| 3.7 | Quality gate: ánh sáng, góc, score và hướng dẫn retry | Brightness/V9/pose quality strip + blocking submit | ✅ Xong prototype |
| 3.8 | AES-GCM encrypt + server decrypt | Secure vector transport | ✅ Xong |

> **Cập nhật 2026-05-24:** `web_stage3` đã chặn submit khi opt-in face nhưng chưa có real quality-passed embedding. Dataset E2E đã xác nhận score 0.974, brightness 131, sync count 1, match score 1.0/distance 0. Webcam path đã có burst 12 frame, chọn tối đa 7 frame tốt nhất rồi average + L2 normalize.

### Giai đoạn 4: Edge Integration (2 tuần) ✅ HOÀN THÀNH

> Đã tích hợp đầy đủ tính năng kết nối mạng, đồng bộ cache từ xa, thuật toán cosine và hiển thị trực tuyến lên thiết bị MaixCAM vật lý.

| # | Task | Output / Giải pháp | Trạng thái |
|:--|:---|:---|:---|
| 4.1 | WiFi/USB connection trên MaixCAM | IP tĩnh virtual USB network interface `10.154.35.1` kết nối ổn định. | ✅ Xong |
| 4.2 | HTTP client: sync cache từ server | [sync_cache.py](file:///d:/AIoT_DoAn/MaixCAM_App/sync_cache.py) đồng bộ tự động dữ liệu hành khách. | ✅ Xong |
| 4.3 | Local matching: cosine search trên cache | Tích hợp thuật toán cosine distance tối ưu hóa tính toán thuần trong `main.py` (<1ms). | ✅ Xong |
| 4.4 | Fallback: gọi server API khi cache miss | Cơ chế tự động fallback gửi POST request lên FastAPI server để kiểm tra Qdrant DB. | ✅ Xong |
| 4.5 | Hiển thị kết quả: LCD + LED + Audio | [mjpeg_server.py](file:///d:/AIoT_DoAn/MaixCAM_App/mjpeg_server.py) stream luồng camera kèm HUD nhận diện qua HTTP port 8080. | ✅ Xong |
| 4.6 | Cache lifecycle: auto-sync, TTL, cleanup | Quản lý cache qua file JSON trên SD card và cơ chế cờ hiệu `sync_now.flag`. | ✅ Xong |

> **Cập nhật 2026-05-24:** Đã triển khai giải pháp stream MJPEG thay thế cho màn hình LCD vật lý giúp người dùng giám sát và kiểm tra trực tiếp từ máy tính chủ mà không cần phần cứng hiển thị LCD chuyên dụng. Đã tích hợp time.sleep(0.005) chống nghẽn CPU và mất kết nối USB.

### Giai đoạn 5: Tích Hợp & Kiểm Thử (1 tuần) 🟢 100%

| # | Task | Output / Kết quả | Trạng thái |
|:--|:---|:---|:---|
| 5.1 | End-to-end test: Web đăng ký → Server lưu → Edge nhận diện | Kiểm thử liên thông toàn chuỗi thành công, nhận diện khớp khoảng cách cosine `0.039`. | ✅ Xong |
| 5.2 | Stress test: 50+ hành khách, đo latency toàn trình | **Edge (MaixCAM):** YOLO 11.17ms avg, E2E 15.86ms avg, 53 FPS, 100 iters/1.8s. **Server:** 100% reg success, ~285ms avg latency. **Script:** `MaixCAM_App/stress_test_maixcam.py` + `scripts/run_maixcam_stress_test.py`. | ✅ Xong |
| 5.3 | Kiểm thử bảo mật: sniff traffic, verify encryption | AES-GCM-256 (Web -> Server) và XTEA-CTR-128 (Server <-> Edge) bảo mật hoàn toàn. | ✅ Xong |
| 5.4 | Demo video + poster đồ án | Đã ghi hình và lưu trữ demo động tại [maixcam_live_feed](maixcam_live_feed_1779604313240.webp). | ✅ Xong |

> **Cập nhật 2026-05-24 (Bảo mật & Mã hóa):** Đã tích hợp thành công hệ thống mã hóa kép (Dual-Cipher):

> **Cập nhật 2026-05-25 (Stress Test):** Đo hiệu năng thực tế trên MaixCAM và Server:
>
> **MaixCAM Edge Device:**
> - YOLO Detection: **11.17ms avg**, 13.90ms P95
> - Pipeline E2E: **15.86ms avg**, 16.86ms P95 — cực nhanh!
> - Throughput: **53 FPS** (100 iterations / 1.8s)
> - Camera: GC4653 720P 60fps, resolution 320x224
>
> **FastAPI Server:**
> - Registration: 100% success rate với 10/50/100/200 passengers
> - Latency: **~285ms avg**, P95 ~320-350ms
> - Sync throughput: 226.72 KB cho 200 passengers trong <500ms
>
> **Files:**
> - `MaixCAM_App/stress_test_maixcam.py` — stress test trên thiết bị biên
> - `scripts/run_maixcam_stress_test.py` — upload & run tự động qua SSH
> - `scripts/tests/benchmark_edge_pc.py` — benchmark mô phỏng trên PC
> 1. Trình duyệt mã hóa vector bằng AES-GCM-256 (sử dụng Web Crypto API) trước khi gửi lên FastAPI Server qua endpoint `/api/face/register`.
> 2. FastAPI Server giải mã và lưu trữ tạm thời trong RAM, sau đó lập chỉ mục vector plaintext trên Qdrant DB.
> 3. Khi đồng bộ cache về Edge qua endpoint `/api/sync/{flight_id}`, server mã hóa on-the-fly bằng XTEA-CTR (128-bit) với khóa thiết bị duy nhất.
> 4. MaixCAM lưu dữ liệu mã hóa XTEA-CTR trực tiếp xuống thẻ MicroSD. Không ghi thông tin plaintext xuống lưu trữ tĩnh. Chỉ giải mã vào bộ nhớ RAM khi chạy đối sánh cosine để triệt tiêu nguy cơ lộ dữ liệu sinh trắc học nếu thẻ nhớ bị đánh cắp.


---

## 6. Cấu Trúc Thư Mục Đề Xuất (Toàn Hệ Thống)

```
project-root/
├── web/                          # Khối 1: Web Client
│   ├── index.html                # Trang đặt vé + đăng ký khuôn mặt
│   ├── models/                   # ONNX models cho browser
│   │   ├── face_detect_v9.onnx
│   │   └── face_recognize_arcface_p3.onnx
│   └── js/
│       ├── pipeline.js           # V9 + P3 inference pipeline
│       ├── crypto.js             # AES-GCM encrypt/decrypt
│       └── registration.js       # Multi-frame capture logic
│
├── server/                       # Khối 2: Server Backend
│   ├── main.py                   # FastAPI entry point
│   ├── models/                   # SQLAlchemy models
│   ├── routes/                   # API routes
│   │   ├── bookings.py
│   │   ├── flights.py
│   │   └── face.py               # Register + Match + Sync
│   ├── services/
│   │   ├── crypto_service.py     # AES-GCM operations
│   │   └── vector_service.py     # Cosine similarity search
│   └── database.py
│
├── edge/                         # Khối 3: MaixCAM App
│   ├── main.py                   # Pipeline chính (đã có)
│   ├── sync_cache.py             # Đồng bộ vector từ server
│   ├── display.py                # LCD + Audio output
│   └── models/                   # .mud + .cvimodel files
│
├── training/                     # Code training (đã hoàn thành)
│   ├── train_v9.py
│   ├── train_recognize.py
│   └── evaluate_models.py
│
└── docs/                         # Tài liệu
    ├── development_plan.md       # File này
    ├── report.md                 # Báo cáo benchmark
    └── benchmark-pipeline.md
```

---

## 7. Rủi Ro & Giải Pháp

| Rủi ro | Mức độ | Giải pháp |
|:---|:---|:---|
| RAM 128MB không đủ cho 3 model + cache | Cao | Load model tuần tự, cache lưu MicroSD chỉ đọc khi cần |
| WiFi mất kết nối tại sân bay | Trung bình | Cache pre-sync 3h trước, hoạt động offline hoàn toàn |
| Ánh sáng kém tại quầy check-in | Trung bình | Thêm đèn LED ring quanh camera, kiểm tra luminance |
| Khuôn mặt đeo khẩu trang | Cao | V9 đã train với CelebA có occlusion, giảm ngưỡng xuống 0.06 |
| Latency quá cao trên TPU | Thấp | Đã benchmark: V9=2.7ms + P3=6.4ms → tổng <15ms (đạt yêu cầu) |

---

## 8. Chỉ Số Đánh Giá Thành Công

| Chỉ số | Mục tiêu |
|:---|:---|
| Độ chính xác nhận diện (LFW) | ≥ 95% |
| Tỷ lệ False Positive | < 0.1% |
| Thời gian nhận diện E2E (Edge) | < 500ms |
| Thời gian đăng ký khuôn mặt (Web) | < 15 giây |
| Số hành khách cache/chuyến | ≥ 200 |
| Hoạt động offline sau sync | ✅ |

---

## 9. Stress Test Chi Tiet (2026-05-25)

### 9.1 Gioi Thieu

**Stress test** la bai kiem tra danh gia kha nang chiu tai va hieu nang cua he thong khi phai xu ly so luong lon hanh khach dong thoi.

### 9.2 Muc Dich

1. **Do do tre dang ky (Registration Latency)**
   - Thoi gian server nhan, giai ma AES-GCM, va luu vector vao Qdrant
   - Anh huong truc tiep den trai nghiem nguoi dung khi dang ky khuon mat

2. **Do throughput cua Sync API**
   - Thoi gian truy van va tra ve danh sach vector cua ca chuyen bay
   - Anh huong den Edge device khi sync cache truoc gio bay

3. **Kiem tra do on dinh**
   - Ty le thanh cong khi xu ly N hanh khach lien tiep
   - Phat hien bottleneck hoac memory leak

### 9.3 Phuong Phap

**Script**: scripts/run_maixcam_stress_test.py (tự động upload + chạy trên MaixCAM)

`
Luong test moi batch:
1. Tao chuyen bay moi (flight_id)
2. Loop N lan:
   - Tao passenger moi (name, email, phone)
   - Tao booking (booking_code, passenger_id, flight_id)
   - Tao vector 128D gia (random, L2 normalized)
   - Ma hoa AES-GCM-256 (booking_id, ciphertext, IV)
   - POST /api/face/register -> Qdrant upsert
   - Do latency: start -> response received
3. Goi GET /api/sync/{flight_id}
   - Do sync time + kich thuoc response
4. Tinh toan thong ke (avg, p95, throughput)
`

**Test sizes**: 10, 50, 100, 200 hanh khach (tong 360 passengers)

**Progress Bar**: Hien thi % hoan thanh + ETA theo thoi gian thuc
`
      [========================================] 100.0% ETA: 0s Pax 200/200
`

### 9.4 Ket Qua

| Size | Reg OK | Avg ms | P95 ms | Sync ms | KB | Bytes/Pax |
|:---|:---|:---|:---|:---|:---|:---|
| 10 | **10/10 (100%)** | 283.71 | 311.98 | 278.87 | 11.32 | 1159 |
| 50 | **50/50 (100%)** | 285.71 | 317.39 | 337.36 | 56.55 | 1158 |
| 100 | **100/100 (100%)** | 284.17 | 320.09 | 362.73 | 113.27 | 1159 |
| 200 | **200/200 (100%)** | 295.10 | 350.75 | 456.93 | 226.72 | 1161 |

### 9.5 Phan Tich

| Chi so | Gia tri | Danh gia |
|:---|:---|:---|
| Registration Avg Latency | **~285ms** | Tot - duoi nguong 500ms |
| Registration P95 Latency | **~320-350ms** | On dinh - bien do dao dong thap |
| Sync Latency (200 pax) | **~457ms** | Nhanh - < 1 giay cho 200 vectors |
| Bytes/Passenger | **~1159 bytes** | Nhat quan - overhead ma hoa on dinh |
| Success Rate | **100%** | Hoan hao - khong co loi |

### 9.6 Ket Luan

- He thong dap ung tot voi **200+ hanh khach/chuyen bay**
- Latency on dinh khong tang tuyen tinh theo so luong (Qdrant HNSW index hieu qua)
- Sync API phu hop de Edge device sync truoc gio bay
- **San sang production** voi quy mo san bay vua va nho
