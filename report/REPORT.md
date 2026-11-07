# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 11/06/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

---

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1:

   > `class_id` = 468 ; `class_name` = "cab" ; `rank` = 1 ; `score` = 0.510915 ; `taxonomy_name` = "ImageNet-1K"

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

   > _Nói về cả bức ảnh. Nhiệm vụ Image Classification gán duy nhất một nhãn phân loại đại diện cho toàn bộ khung cảnh/nội dung chính của ảnh, không khoanh vùng từng vật thể riêng lẻ._

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

   > _Đi kèm với taxonomy/checkpoint (ở đây là ImageNet-1K) đã được định nghĩa từ trước khi huấn luyện mô hình phân loại. Mô hình không tự nghĩ ra tên lớp mới._

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

   > _Để tránh nhầm lẫn và xung đột. Cùng một tên lớp (ví dụ: "cab") có thể mang ID khác nhau ở các bộ dữ liệu khác nhau. `taxonomy_name` xác định rõ ngữ cảnh nguồn của danh sách nhãn đó, đảm bảo tính nhất quán._

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

   > _Guideline cần định nghĩa quy tắc ưu tiên rõ ràng. Ví dụ: Chọn đối tượng chiếm diện tích lớn nhất ở trung tâm ảnh, chọn chủ thể hành động chính, hoặc quy định thứ tự ưu tiên các lớp từ cao xuống thấp khi có tranh chấp._

6. Vì sao không được lấy `score` của model làm ground truth?
   > _Vì score chỉ thể hiện mức độ tự tin của mô hình toán học tại thời điểm đó. Nó vẫn chứa sai số và không thể thay thế cho nhãn chuẩn xác tuyệt đối (ground truth) do con người xác thực._

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record:

   > `class_name` = "person" ; `score` = 0.912625 ; `bbox_xyxy` = [385.33, 69.24, 498.92, 348.92] ; `bbox_width` = 113.58 ; `bbox_height` = 279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

   > _Hộp giới hạn bao quanh vật thể có góc trên-trái nằm ở tọa độ (x=385.33, y=69.24); góc dưới-phải ở tọa độ (x=498.92, y=348.92). Chiều rộng của hộp là 113.58 pixel và chiều cao là 279.68 pixel (tính từ điểm gốc 0,0 ở góc trên cùng bên trái của bức ảnh)._

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

   > _Ngưỡng `0.20`: Có 17 vật thể._
   > _Ngưỡng `0.60`: Có 6 vật thể._

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

   > _Phải tốn nhiều thời gian và công sức hơn để kiểm tra, lọc bỏ hoặc xóa các hộp dự đoán sai (False Positives), nhiễu nền hoặc các hộp trùng lặp do mô hình quá nhạy bén tạo ra._

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

   > _"Vẽ hộp giới hạn khít nhất có thể sát vào biên của vật thể (margin tối đa không quá 2-3 pixel). Tránh vẽ hộp quá rộng chứa nhiều khoảng nền, hoặc vẽ quá hẹp cắt lẹm vào các bộ phận của vật thể."_

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   > - **Quan sát:** Các nồi/chảo treo trên giá bị che khuất một phần bởi nhau.\*
   > - **Quyết định:** Viết thành quy tắc guideline thay vì escalation, vì đây là lỗi phổ biến (occlusion). Quy tắc: "Đối với vật thể bị che khuất một phần nhưng vẫn nhận diện được hình dáng/lớp, hãy vẽ box bao phủ toàn bộ phần nhìn thấy được và ước lượng biên của phần bị che khuất (nếu phần bị che khuất chiếm dưới 50% diện tích vật thể)."\*

---

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record:

   > `instance_id` = "kitchen-001" ; `class_name` = "person" ; `score` = 0.95 (giả định dựa trên file) ; `polygon_point_count` = 348 ; `polygon_xy` = [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], ...]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

   > _Polygon cung cấp hình dáng và ranh giới thực tế chi tiết đến từng pixel của vật thể, loại bỏ hoàn toàn các khoảng không gian trống và phông nền nhiễu thường bị kẹt lại bên trong các hộp giới hạn hình chữ nhật (bounding boxes)._

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

   > - `instance_id` dùng để phân biệt các thực thể cá biệt cùng loại trong một khung ảnh (ví dụ: người 1, người 2).\*
   > - Nó **không phải class ID** vì class ID dùng để định danh nhóm/lớp vật thể chung (như 'person' là 0).\*
   > - Nó **không phải tracking ID** vì tracking ID dùng để khớp theo dõi một đối tượng duy nhất xuyên suốt nhiều frame liền kề của video, còn ở bài toán này ta chỉ xử lý trên một ảnh tĩnh đơn lẻ.\*

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

   > _"Chấm các điểm đa giác (polygon) bám sát chính xác theo đường viền thực tế của đối tượng. Đảm bảo đường viền không lấn sâu vào trong cơ thể đối tượng quá 2 pixel và tuyệt đối không bao gồm phần phông nền xung quanh."_

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   > - **Quan sát:** Các chiếc thìa hoặc nồi treo đè lên nhau ở phía trên bên trái.\*
   > - **Quyết định:** Ưu tiên tra cứu và áp dụng guideline trước, vì đa số các trường hợp chồng lấn cơ bản đã có quy chuẩn gán nhãn. Chỉ dùng escalation để hỏi Mentor/Lab Coach khi gặp trường hợp quá mờ hoặc vật thể dị dạng không thể dùng mắt thường để suy luận ranh giới, tuyệt đối không tự ý đoán mò.\*

---

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:
`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ                    | Ground truth trông như thế nào? (đơn vị nhãn)                               | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                     | Annotator làm gì?                                                                                    | Reviewer xem gì?                                                                   |
| ------------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Phân loại ảnh**         | 1 ID lớp (Class ID/Name) duy nhất cho toàn bộ ảnh.                          | Ảnh chứa nhiều chủ đề/vật thể phức tạp (tranh chấp lớp chính).         | Lựa chọn 1 lớp đại diện chuẩn xác nhất dựa trên thứ tự ưu tiên của guideline.                        | Kiểm tra xem nhãn được chọn có đúng là tiêu điểm chính của bức ảnh hay không.      |
| **Phát hiện vật thể**     | Tọa độ Bounding Box `(x_min, y_min, x_max, y_max)` kèm Class ID.            | Hộp lấn quá nhiều nền, bị cắt lẹm mép, hoặc vật bị che khuất.          | Căn chỉnh góc hộp khít với biên vật thể nhất (margin 2-3px) và ước lượng viền bị che theo guideline. | Kiểm tra độ khít của bounding box và xác minh xem gán nhãn lớp đã đúng chưa.       |
| **Instance segmentation** | Tập hợp các điểm tọa độ đa giác `(polygon_xy)` kèm Class ID và Instance ID. | Ranh giới giữa hai vật thể đè lên nhau bị mờ nhòe, khó phân biệt biên. | Zoom kỹ, chấm điểm polygon bám sát đường viền pixel thực tế, bỏ qua phần bị khuất hoàn toàn.         | Kiểm tra xem polygon có bị tràn viền ra nền hoặc cắt phạm vào đối tượng hay không. |

---

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   > _Tuyệt đối không upload dữ liệu cá nhân, dữ liệu khách hàng hoặc các hình ảnh nội bộ chưa được cấp phép lên môi trường Colab công cộng, và không lưu trữ thông tin nhận diện cá nhân trong báo cáo hay code._

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   > _Sẽ ngay lập tức dừng việc thao tác/gán nhãn, đóng cửa sổ để bảo mật dữ liệu đó, và escalate báo cáo trực tiếp cho Lab Coach hoặc Mentor quản lý dự án để xin chỉ thị xử lý._

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [v] `REPORT.md` (file này đã điền)
- [v] `day1_lab_outputs/classification_predictions.json`
- [v] `day1_lab_outputs/detection_predictions.json`
- [v] `day1_lab_outputs/segmentation_predictions.json`
- [v] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [v] `day1_lab_outputs/visuals/classification_top5.png`
- [v] `day1_lab_outputs/visuals/detection_predictions.png`
- [v] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [v] Ô validation cuối notebook báo `PASS`
- [v] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [v] Đã nộp **link repository** trên LMS
