# Giải Pháp Kỹ Thuật: Xử Lý Đa Vé Cho Một Khuôn Mặt (Multi-Booking Edge Case)

Trong kiến trúc hệ thống hiện tại của dự án AIoT Flight Face Lookup, mô hình 1-1 (1 khuôn mặt = 1 vé) đang được áp dụng. Hàm `match_local` trên thiết bị biên MaixCAM hoặc API `search_face_embedding` trên Server chỉ trả về 1 kết quả duy nhất (bằng cách lấy kết quả có khoảng cách cosine nhỏ nhất và dừng lại).

Tuy nhiên, trong môi trường sân bay thực tế, việc một khuôn mặt khớp với nhiều vé rất dễ xảy ra. Tài liệu này mô tả chi tiết các tình huống thực tế và đề xuất kiến trúc xử lý cho đồ án/giai đoạn phát triển tiếp theo.

---

## 1. Các Kịch Bản Thực Tế (Edge Cases)

### Kịch bản A (Hợp lệ): Hành khách bay nối chuyến (Transit)
Một người có 2 chặng bay trong ngày (VD: chặng 1 lúc 8h sáng, chặng 2 lúc 14h chiều). Cả 2 vé đều được đăng ký khuôn mặt của người này.
- Khi người này soi mặt ở Kiosk Sảnh Chung (Main Hall - Kiosk không giới hạn trong 1 `flight_id` cụ thể), hệ thống cần tìm kiếm trên toàn bộ cơ sở dữ liệu và trả về cả 2 vé hợp lệ.

### Kịch bản B (Bad Practice): Đăng ký hộ gia đình
Một hành khách thao tác đặt vé cho cả nhà (4 vé trong cùng 1 `flight_id`). Nhưng ở bước đăng ký Face Check-in, người này vô tình (hoặc cố ý) dùng khuôn mặt của chính mình liên tục cho cả 4 vé.
- Kết quả: Cơ sở dữ liệu Vector (Qdrant) chứa 4 vector cực kỳ giống nhau, liên kết với 4 ghế ngồi khác nhau (VD: 12A, 12B, 12C, 12D).
- Hệ thống hiện tại sẽ chỉ nhận diện ngẫu nhiên 1 trong 4 ghế này và hiển thị ra màn hình, gây khó hiểu cho người dùng và nhân viên kiểm soát.

---

## 2. Đề Xuất Thay Đổi Kiến Trúc Kỹ Thuật

Để xử lý triệt để vấn đề này, hệ thống cần được nâng cấp ở cả 3 khối: Server API, Local Cache Sync (MaixCAM), và UI/UX hiển thị.

### 2.1 Cải tiến API Đối Sánh (`/api/face/match`) trên Backend

**Vấn đề:** Tham số `limit=1` đang chặn việc hiển thị các kết quả hợp lệ khác trong Qdrant.
**Triển khai:**
- Tăng tham số giới hạn tìm kiếm trong thuật toán HNSW của Qdrant lên `limit=5`.
- Sàng lọc lại tập kết quả bằng ngưỡng `threshold` (mặc định là `0.045`). 
- Chuyển cấu trúc trả về từ một đối tượng JSON `booking` duy nhất sang một mảng `matches: []`.

**Mã nguồn dự kiến trong `server/face_routes.py`:**
```python
@router.post("/face/match")
def match_face(payload: FaceMatchRequest) -> dict[str, Any]:
    # ... logic giải mã vector ...

    try:
        # Thay đổi từ limit=1 thành limit=5
        results = search_face_embedding(vector, payload.flight_id, limit=5)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    matched_bookings = []
    for hit in results:
        score = float(hit.score)
        distance = 1.0 - score
        if distance <= payload.threshold:
            booking = get_booking(int(hit.payload["booking_id"]))
            matched_bookings.append({
                "distance": distance,
                "score": score,
                "point_id": str(hit.id),
                "booking": booking
            })

    if not matched_bookings:
        return {"matched": False, "matches": [], "threshold": payload.threshold}

    # Sắp xếp danh sách ưu tiên người có độ giống cao nhất lên đầu
    matched_bookings.sort(key=lambda x: x["distance"])

    return {
        "matched": True,
        "matches": matched_bookings,
        "threshold": payload.threshold
    }
```

### 2.2 Cải tiến Edge Cache Match (`MaixCAM_App/sync_cache.py`)

Tại thiết bị biên (Edge Device), logic đối sánh Cosine Distance `match_local` hiện tại đang dừng lại khi tìm được biến `best_dist`. Thuật toán này cần được sửa đổi để duyệt qua toàn bộ mảng cache, thu thập tất cả những vector đạt chuẩn.

**Mã nguồn dự kiến trong `sync_cache.py`:**
```python
def match_local(self, embedding: list, flight_id: int) -> dict | None:
    cache = self._read_cache(flight_id)
    if cache is None:
        return None

    query = _l2_normalize([float(v) for v in embedding])
    
    matches = []
    for item in cache.get("items", []):
        ref = item.get("embedding")
        if not ref:
            continue
            
        dist = _cosine_distance(query, ref)
        if dist <= self.threshold:
            matches.append({
                "distance": dist,
                "score": 1.0 - dist,
                "payload": item.get("payload", {}),
                "point_id": item.get("point_id"),
            })
            
    if not matches:
        return None
        
    # Sắp xếp các kết quả
    matches.sort(key=lambda x: x["distance"])
    
    return {
        "source": "cache",
        "matches": matches,
        "primary_distance": matches[0]["distance"]
    }
```

### 2.3 Giao Diện UI/UX Trên Kiosk MaixCAM

Do hạn chế về phần cứng Edge (màn hình độ phân giải thấp, dòng chữ hiển thị thông qua thư viện `image.draw_string`), không thể in ra một bảng danh sách dài các vé. Giao diện cần được thiết kế thông minh:

**A. Kiosk Sảnh Chung (Không khóa chuyến bay - Dành cho người Transit):**
- **Hiển thị Carousel tự động:** Thiết bị MaixCAM duy trì một biến bộ đếm thời gian (dựa trên `time.time()`). Nếu mảng kết quả có 2 vé, cứ sau 3 giây màn hình sẽ chuyển giao diện hiển thị giữa Vé 1 và Vé 2.
- Dưới chân màn hình (HUD) cần có chấm tròn báo hiệu số lượng vé (VD: `(1/2)` và `(2/2)`).

**B. Kiosk Tại Cổng Lên Máy Bay (Khóa theo 1 flight_id - Lỗi Đăng Ký Hộ):**
- Màn hình sẽ gộp chung thông tin thành một "Group Ticket". Nhóm tên hành khách và tất cả các số ghế để dễ dàng đọc nhanh.
- Ví dụ cách hiển thị bằng MicroPython trên MaixCAM:
```python
if len(matches) > 1:
    # Gộp tên và các số ghế
    primary_name = matches[0]["payload"]["passenger_name"]
    all_seats = [m["payload"]["seat_number"] for m in matches]
    seat_text = ", ".join(all_seats)
    
    img.draw_string(x, max(0, y - 30), f"{primary_name} (+{len(matches)-1})", image.COLOR_GREEN)
    img.draw_string(x, max(0, y - 15), f"Seats: {seat_text}", image.COLOR_YELLOW)
else:
    # Hiển thị 1 vé như bình thường
```

---

## 3. Khuyến Nghị: Áp Dụng Quality Gate Bảo Mật Nghiệp Vụ

Dù phương pháp gộp vé (như 2.3 B) giúp luồng vận hành trơn tru hơn, nguyên lý cơ bản của sinh trắc học là định danh 1:1. Tốt nhất là **chặn hoàn toàn việc đăng ký 1 khuôn mặt cho nhiều người khác nhau trên cùng một chuyến bay**.

Nên triển khai một "Quality Gate" ngay trên Server trước khi cập nhật Vector DB:

**Trong `POST /api/face/register` của `server/face_routes.py`:**
```python
# Trước khi gọi upsert_face_embedding(...)
existing_results = search_face_embedding(vector, booking["flight_id"], limit=1)

if existing_results:
    best_match = existing_results[0]
    distance = 1.0 - float(best_match.score)
    
    # Ngưỡng nghiêm ngặt (0.045) cho thấy đây có thể là cùng 1 người
    if distance <= 0.045:
        raise HTTPException(
            status_code=409, 
            detail="Khuôn mặt này đã được sử dụng cho một hành khách khác trong cùng chuyến bay. Vui lòng không đăng ký hộ."
        )

# Nếu vượt qua Quality Gate, tiến hành lưu bình thường
upsert_face_embedding(point_id, vector, face_payload)
```

Điều này ép buộc người dùng phải tự đưa mặt vào lúc đăng ký, giữ cho cơ sở dữ liệu Vector phân tán cực kỳ sạch và độ tin cậy của hệ thống đạt mức cao nhất.
