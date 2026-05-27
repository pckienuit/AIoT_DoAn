# Chống Trùng Vé Theo Chuyến Bay Và Ghế

Tài liệu này mô tả lỗi tạo nhiều `booking_code` cho cùng một `flight_id` và `seat_number`, cách server đã xử lý, và cách kiểm tra sau khi sửa.

## Vấn Đề

Trong màn hình **Vé của tôi**, có thể xuất hiện nhiều vé khác `booking_code` nhưng cùng:

- Mã chuyến bay.
- Ngày giờ bay.
- Số ghế.
- Hành khách.

Ví dụ đã ghi nhận trong DB:

| flight_id | seat_number | booking_code |
|---|---|---|
| `2521` | `3E` | `9E1361FD` |
| `2521` | `3E` | `A1FF8551` |

## Nguyên Nhân

Endpoint `POST /api/bookings` đã có kiểm tra ghế bị chiếm, nhưng kiểm tra này chưa đủ chắc trong trường hợp có request gần như đồng thời:

1. Request A kiểm tra ghế `3E`: chưa thấy ai giữ.
2. Request B cũng kiểm tra ghế `3E`: chưa thấy ai giữ.
3. Cả hai request cùng `INSERT` booking.
4. Server sinh hai `booking_code` khác nhau cho cùng một ghế.

Đây là race condition ở tầng ghi dữ liệu. UI chỉ hiển thị dữ liệu server trả về nên không phải nguyên nhân gốc.

## Mục Tiêu Đúng

Với các booking còn hiệu lực, mỗi cặp sau chỉ được tồn tại một lần:

```text
flight_id + seat_number
```

Các trạng thái được xem là không còn giữ ghế:

- `cancelled`
- `refunded`

Các trạng thái vẫn giữ ghế:

- `confirmed`
- `checked_in`
- Các trạng thái active khác nếu được thêm sau này.

## Cách Server Đang Chặn Duplicate

Logic chính nằm trong `server/routes.py`.

### Chuẩn hóa mã ghế

Ghế được parse và chuẩn hóa trước khi lưu:

```text
"3e" -> "3E"
"3e, 4a" -> "3E, 4A"
```

Điều này tránh việc `3e` và `3E` bị xem là hai ghế khác nhau.

### Khóa ghi khi tạo hoặc đổi ghế

Khi dùng SQLite, server gọi `BEGIN IMMEDIATE` trước khi kiểm tra ghế và trước khi ghi booking. Nhờ vậy các request ghi booking được xếp hàng, request sau không thể đọc trạng thái cũ trong lúc request trước chưa commit.

Luồng tạo booking mới:

1. Lấy thông tin chuyến bay.
2. Parse và chuẩn hóa `seat_number`.
3. Mở transaction ghi.
4. Kiểm tra lại `available_seats` trong transaction.
5. Kiểm tra ghế active đã bị chiếm chưa.
6. Tạo passenger nếu cần.
7. Insert booking.
8. Trừ `available_seats`.
9. Commit.

Nếu ghế đã có booking active, API trả:

```json
{
  "detail": "Seat already taken: 3E"
}
```

với HTTP status `409 Conflict`.

### Đổi ghế cũng dùng cùng kiểm tra

Endpoint đổi ghế cũng kiểm tra duplicate trong transaction và bỏ qua chính booking hiện tại khi so sánh. Vì vậy đổi từ ghế A sang ghế B sẽ bị chặn nếu ghế B đang được booking khác giữ.

## Guard Ở Schema

`server/database.py` có thêm unique index cho SQLite DB sạch:

```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_bookings_active_flight_seat_unique
ON bookings(flight_id, seat_number)
WHERE seat_number IS NOT NULL
  AND trim(seat_number) != ''
  AND status NOT IN ('cancelled', 'refunded');
```

Index này là lớp bảo vệ cuối cùng ở DB. Nếu DB hiện tại đã có dữ liệu trùng, index sẽ chưa bật được và server sẽ in cảnh báo:

```text
[WARN] bookings active seat uniqueness not enabled; clean duplicate active seats first (...)
```

Cảnh báo này không làm server dừng chạy. Sau khi cleanup duplicate cũ, chạy lại server để index được tạo.

## Cleanup Dữ Liệu Trùng Cũ

Trước khi bật được unique index trên DB hiện tại, cần xử lý các booking active đang trùng ghế.

Câu query tìm duplicate:

```sql
SELECT
  flight_id,
  upper(trim(seat_number)) AS seat,
  COUNT(*) AS count,
  group_concat(booking_code) AS booking_codes
FROM bookings
WHERE seat_number IS NOT NULL
  AND trim(seat_number) != ''
  AND status NOT IN ('cancelled', 'refunded')
GROUP BY flight_id, upper(trim(seat_number))
HAVING COUNT(*) > 1;
```

Với mỗi nhóm duplicate, chọn một booking để giữ lại. Các booking còn lại nên được cập nhật thành `cancelled` hoặc `refunded` thay vì xóa thẳng, để vẫn giữ lịch sử giao dịch.

Ví dụ:

```sql
UPDATE bookings
SET status = 'cancelled',
    updated_at = datetime('now')
WHERE booking_code = 'A1FF8551';
```

Sau khi cleanup, khởi động lại server hoặc chạy `init_db()` để unique index được tạo.

## Cách Kiểm Tra

Kiểm tra cú pháp Python:

```powershell
python -m py_compile server\routes.py server\database.py
```

Kiểm tra DB còn duplicate active hay không:

```powershell
@'
import sqlite3

conn = sqlite3.connect("server/data/prototype.db")
conn.row_factory = sqlite3.Row
rows = conn.execute("""
    SELECT flight_id, upper(trim(seat_number)) seat, COUNT(*) c, group_concat(booking_code) codes
    FROM bookings
    WHERE seat_number IS NOT NULL
      AND trim(seat_number) != ''
      AND status NOT IN ('cancelled','refunded')
    GROUP BY flight_id, upper(trim(seat_number))
    HAVING COUNT(*) > 1
""").fetchall()
print([dict(r) for r in rows])
conn.close()
'@ | python -
```

Kết quả mong muốn:

```text
[]
```

Kiểm tra API bằng thao tác thực tế:

1. Chọn một chuyến bay.
2. Đặt ghế `3E` và thanh toán hoặc tạo booking.
3. Thử đặt lại chính ghế `3E` cho cùng chuyến.
4. API phải trả `409 Conflict`, UI không được tạo thêm vé mới.

## File Liên Quan

- `server/routes.py`: logic tạo booking, đổi ghế, kiểm tra ghế đã bị giữ.
- `server/database.py`: schema bookings, context manager DB, unique index guard.
- `web_stage3/pages/my-tickets.html`: màn hình hiển thị danh sách vé.
- `web_stage3/pages/seat-map.html`: luồng chọn ghế trước khi thanh toán.
