# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- **Record hạng 1** (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `468`, `cab`, `1`, `0.510915`, `ImageNet-1K`.

- **Record này mô tả toàn ảnh như thế nào?**
  Vì `task: "image_classification"`, record không gắn với vùng ảnh cụ thể nào (không có `bbox`) mà gán một nhãn duy nhất cho toàn bộ khung ảnh 640×428. Model kết luận: nội dung tổng thể của ảnh `traffic` có khả năng cao nhất (~51%) thuộc lớp `cab` (xe taxi). Đây là kết luận về **cả bức ảnh**, không phải về một vật thể riêng lẻ trong ảnh.

- **Ai định nghĩa class list mà checkpoint có thể dự đoán?**
  Danh sách lớp không do người dùng hay dữ liệu evidence quyết định, mà do **bộ dữ liệu huấn luyện gốc** — `taxonomy_name: "ImageNet-1K"` — quy định sẵn 1000 lớp cố định. Checkpoint `yolo11n-cls.pt` được huấn luyện trên taxonomy đó nên chỉ có thể trả về một trong các `class_id`/`class_name` đã định sẵn (ví dụ ở sample `traffic`: `cab`, `minibus`, `police_van`, `recreational_vehicle`, `streetcar`; ở sample `kitchen`: `gong`, `dining_table`, `restaurant`, `lumbermill`, `bakery`).

- **Vì sao cần giữ cả ID, tên lớp và tên taxonomy?**
  - `class_id` (số nguyên, ví dụ `468`): khóa ổn định để đối chiếu bằng máy.
  - `class_name` (ví dụ `"cab"`): để người đọc hiểu ngay.
  - `taxonomy_name` (`"ImageNet-1K"`): xác định **ngữ cảnh định danh** — vì cùng một `class_id` có thể mang nghĩa khác nhau ở taxonomy khác (ví dụ `COCO-80` dùng trong phần 2 và 3 của báo cáo này). Thiếu trường này, con số ID trở nên vô nghĩa hoặc dễ hiểu sai.

- **Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?**
  Dữ liệu cho thấy mỗi ảnh có 5 record (rank 1–5), không chỉ 1 nhãn. Ví dụ sample `kitchen`: rank 1 `gong` (0.42) sát nút với rank 2 `dining_table` (0.07) — chênh lệch điểm số lớn nhưng vẫn không có nhãn nào áp đảo tuyệt đối. Guideline cần quy định: (1) có chấp nhận chỉ lấy top-1 hay phải xem cả top-5 khi ảnh có nhiều chủ thể; (2) ngưỡng score nào được coi là "đủ tin cậy" để chốt nhãn; (3) nếu cần định vị riêng từng vật thể, phải chuyển sang object detection/instance segmentation (phần 2, 3) thay vì dùng classification cấp ảnh.

- **Vì sao model score không phải ground truth?**
  `score` (ví dụ `0.510915` cho `cab`) chỉ là xác suất do model tự ước lượng (softmax), không phải nhãn đã được con người kiểm chứng. Bằng chứng rõ nhất: điểm số top-1 của sample `kitchen` chỉ **0.420192** — dưới 50%, nghĩa là chính model cũng không chắc chắn — nên không thể coi đây là sự thật mà chỉ là một dự đoán cần đối chiếu với ground truth thực tế.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- **Một record** (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): `"person"`, `0.912625`, `[385.33, 69.24, 498.92, 348.92]`, `113.58`, `279.68`.

- **Diễn giải vị trí box bằng lời:**
  Ảnh `kitchen` có kích thước 640×427px. Box của `person` có góc trên-trái tại `(385.33, 69.24)` và góc dưới-phải tại `(498.92, 348.92)`. Vì `x_min ≈ 60%` và `x_max ≈ 78%` chiều rộng ảnh, vật thể nằm ở **phía bên phải** khung hình; `y_min ≈ 16%` và `y_max ≈ 82%` chiều cao ảnh, box trải gần như suốt chiều dọc ảnh (chỉ chừa khoảng trống nhỏ ở trên và dưới) — phù hợp với hình dáng một người đứng, cao 279.68px và rộng 113.58px.

- **So sánh số prediction ở hai threshold:**
  File `detection_predictions.json` được sinh ra với **một** threshold duy nhất, `score_threshold: 0.35`, cho ra **11 record** ở sample `kitchen`. Lọc lại chính tập điểm số đó ở ngưỡng cao hơn (`0.60`) chỉ còn **6 record** (loại bỏ 5 box có score trong khoảng 0.38–0.50: hai `bowl`, một `bowl` khác, hai `cup`). Lưu ý: không thể tái tạo số liệu ở ngưỡng thấp hơn (`0.20`) từ file này, vì quá trình sinh dữ liệu đã lọc bỏ mọi box có score dưới 0.35 ngay từ lúc chạy `predict()` — những box đó chưa từng được ghi lại nên không thể khôi phục.

- **Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?**
  Từ 0.35 → 0.60: số box giảm từ 11 xuống 6 (giảm ~45%) — khối lượng reviewer cần xem giảm đáng kể, nhưng đánh đổi là loại bỏ luôn các box tin cậy trung bình (0.38–0.50) như `cup`, `bowl` — có thể là vật thể thật nhưng bị mất (giảm recall/độ bao phủ). Ngược lại, nếu hạ threshold xuống 0.20 (không có trong dữ liệu nhưng theo nguyên lý), số box sẽ tăng thêm, bắt được nhiều vật thể mờ/nhỏ hơn nhưng reviewer phải xem nhiều box nhiễu hơn.

- **Đề xuất một quy tắc box chặt:**
  Box được coi là "chặt" khi biên box sát mép thật của vật thể ở cả 4 cạnh, không chừa khoảng nền dư thừa lớn, và không cắt mất một phần vật thể đang nằm trong khung ảnh. Ví dụ box `person` ở trên có `x_min = 385.33` rất gần mép thật bên trái của người trong ảnh (không lấn nhiều vào vùng bếp phía sau) — đây là dấu hiệu của box chặt tốt.

- **Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?**
  Dữ liệu không có trường nào đánh dấu occlusion/truncation (ví dụ hai record `bowl` có tọa độ rất gần nhau: `[155.02, 168.67, 182.38, 184.18]` và `[155.76, 113.58, 173.98, 130.13]` — có thể là các bát xếp chồng/gần nhau, khó phân định ranh giới). Guideline cần quyết định: (1) có cần gắn cờ riêng (`occluded`, `truncated`) hay không; (2) box nên vẽ theo phần nhìn thấy được hay ước lượng cả phần bị che; (3) ngưỡng che khuất nào (ví dụ >50% diện tích) thì phải escalation cho reviewer quyết định thay vì chấp nhận tự động.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- **Một record** (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): `"kitchen-001"`, `"person"`, `0.899318`, `348 điểm`, một vài điểm đầu: `[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]` (bám sát viền vai/đầu của người, đi từng bước 1px một để dò theo đường viền thật của vật thể).

- **Polygon bổ sung chi tiết gì so với box?**
  Box (`bbox_xyxy`) chỉ là hình chữ nhật bao ngoài — 4 số tọa độ. Polygon (`polygon_xy`) gồm **348 điểm** nối liền theo đúng đường viền thật (contour) của vật thể, nên biểu diễn được hình dạng chính xác (vai, đầu, tay, dáng đứng...) thay vì chỉ một khung hình chữ nhật thô chứa cả vùng nền xung quanh vật thể.

- **`instance_id` dùng để làm gì và không phải loại ID nào?**
  `instance_id` (ví dụ `"kitchen-001"`) dùng để **phân biệt từng thực thể vật lý riêng lẻ** được phát hiện trong một sample — ví dụ nếu ảnh có 2 người, mỗi người sẽ có `instance_id` khác nhau dù cùng `class_name: "person"`. Nó **không phải** là: `class_id` (không định danh loại/lớp, mà định danh từng cá thể); không phải ID toàn cục xuyên suốt dataset hay khóa chính cơ sở dữ liệu; và không đảm bảo cùng một vật thể sẽ giữ nguyên `instance_id` ở lần chạy khác hoặc ảnh khác.

- **Đề xuất một quy tắc biên mask:**
  Biên mask (polygon) nên bám sát cạnh thật của vật thể ở mức pixel, không cắt xén mất phần vật thể, không bao gồm pixel nền, và đường viền nên mượt (không có điểm zigzag/nhiễu bất thường) trong khi vẫn giữ đúng hình dạng tự nhiên — ví dụ 348 điểm của `person` ở trên di chuyển từng bước rất nhỏ (~1px) để bám theo đường viền thật.

- **Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?**
  Khi hai vật thể chạm/che nhau (ví dụ các `bowl` gần nhau đã thấy ở phần detection), guideline cần quyết định: (1) hai mask có được phép chồng lấn (overlap) ở vùng tiếp xúc hay phải phân chia ranh giới rạch ròi; (2) với phần vật thể bị vật khác che khuất, mask có vẽ theo phần nhìn thấy được hay ước lượng luôn phần bị che; (3) ngưỡng độ mờ/nhiễu nào (ví dụ biên không rõ do ánh sáng, chuyển động) cần escalation cho người review thay vì annotator tự quyết.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn lớp cho toàn ảnh (theo taxonomy cố định, ví dụ ImageNet-1K) | Ảnh có nhiều chủ thể (traffic có cả xe buýt, xe con, xe tải) nhưng chỉ được gán 1 nhãn; điểm top-1 có thể thấp (`kitchen`: 0.42) cho thấy sự mơ hồ | Chọn nhãn phù hợp nhất mô tả nội dung tổng thể của ảnh theo danh sách lớp cho phép | So sánh nhãn được chọn với top-k dự đoán của model, kiểm tra ảnh có thực sự đơn nghĩa hay cần loại khỏi tập dữ liệu |
| Phát hiện vật thể | Bounding box `xyxy` (pixel) + `class_id` cho từng vật thể | Vật thể nhỏ/gần nhau khó tách box (2 `bowl` cạnh nhau); box lỏng bao cả nền xung quanh; box bị mất khi score dưới threshold | Vẽ box chặt sát từng vật thể, gán đúng lớp, đánh dấu occlusion/truncation nếu có | Kiểm tra box có chặt không, có bỏ sót/box thừa không, đối chiếu với ngưỡng score đã chọn |
| Instance segmentation | Polygon/mask theo từng instance (pixel-level) + `instance_id` | Biên mờ giữa các vật thể tiếp xúc/che khuất nhau; polygon nhiều điểm dễ bị nhiễu ở vùng viền phức tạp | Vẽ mask bám sát viền thật từng cá thể, gán `instance_id` riêng biệt, xử lý vùng che khuất theo guideline | Kiểm tra mask có ôm sát biên không, các instance có bị lẫn/chồng lấn sai không, `instance_id` có nhất quán không |

## 5. An toàn dữ liệu

- **Một quy tắc bảo vệ dữ liệu:** Không đưa họ tên, MSSV, email, số điện thoại hay bất kỳ thông tin định danh cá nhân nào vào trong báo cáo, output hoặc metadata của file nộp — chỉ giữ lại dữ liệu evidence (JSON, ảnh minh họa) đúng phạm vi bài thực hành.
- **Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:** người phụ trách/giảng viên hướng dẫn bài thực hành (qua kênh chính thức của môn học, ví dụ VLearn hoặc email giảng viên), đồng thời không tiếp tục xử lý hay chia sẻ thêm dữ liệu đó.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
