# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 11/06/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

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

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Nói về cả bức ảnh. Nhiệm vụ Image Classification gán duy nhất một nhãn phân loại đại diện cho toàn bộ khung cảnh nội dung chính của ảnh, không khoanh vùng từng vật thể riêng lẻ.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Đi kèm với taxonomy/checkpoint ImageNet-1K đã được định nghĩa từ trước khi huấn luyện mô hình phân loại này, mô hình không tự nghĩ ra tên lớp mới.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   Để tránh nhầm lẫn. Cùng một tên lớp ví dụ: cab có thể có ID khác nhau ở các bộ dữ liệu khác nhau, taxonomy_name xác định rõ ngữ cảnh nguồn của danh sách nhãn đó, đảm bảo tính nhất quán dữ liệu.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Guideline cần định nghĩa quy tắc ưu tiên rõ ràng ví dụ: Chọn đối tượng chiếm diện tích lớn nhất ở trung tâm ảnh, hoặc chọn chủ thể hành động chính, hoặc quy định thứ tự ưu tiên các lớp từ cao xuống thấp khi có tranh chấp.
6. Vì sao không được lấy `score` của model làm ground truth?
   Vì score chỉ thể hiện khả năng dự đoán của mô hình toán học tại thời điểm đó, nó vẫn chứa sai số và không thể thay thế cho nhãn chuẩn xác tuyệt đối do con người xác thực.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   Hộp giới hạn bao quanh vật thể có góc trên-trên trái nằm ở tọa độ x=385.33 pixel, y=69.24 pixel; góc dưới-phải ở tọa độ x=498.92 pixel, y=348.92 pixel. Chiều rộng của hộp là 113.58 pixel và chiều cao là 279.68 pixel tính từ điểm gốc (385.33,69.24) ở góc trên bên trái của ảnh.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   Ngưỡng 0.20: Có 17 vật thể.
   Ngưỡng 0.60: Có 6 vật thể.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   Phải tốn nhiều thời gian và công sức hơn để kiểm tra, lọc bỏ hoặc xóa rất nhiều hộp dự đoán sai, nhiễu hoặc trùng lặp do mô hình nhạy bén quá mức tạo ra.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   "Vẽ hộp giới hạn khít nhất có thể sát vào biên của vật thể (Margin tối đa không quá 2-3 pixel). Tránh vẽ hộp quá rộng chứa nhiều khoảng trống trống hoặc cắt lẹm vào bộ phận của vật thể."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Quan sát: Các nồi/chảo treo trên giá bị che khuất một phần bởi nhau.
   Quy tắc: "Đối với vật thể bị che khuất một phần nhưng vẫn nhận diện được lớp: Vẽ bao phủ toàn bộ phần nhìn thấy được và ước lượng phần bị che khuất nếu phần bị che khuất chiếm dưới 50% diện tích vật thể."

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001; `class_name`="person" ; `score`="kitchen-001"; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[
   446.0,
   70.0
   ],
   [
   445.0,
   71.0
   ],
   [
   444.0,
   71.0
   ],
   [
   443.0,
   72.0
   ],
   [
   442.0,
   72.0
   ]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   Polygon cho biết hình dạng ranh giới thực tế chi tiết từng pixel của vật thể thay vì chỉ là một khung hình chữ nhật bao quanh bao gồm cả các khoảng trống và nền nhiễu xung quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   instance_id dùng để phân biệt các thực thể đơn lẻ cùng loại trong một khung ảnh ví dụ: người 1, người 2.
   Nó không phải class ID vì class ID dùng để định danh nhóm lớp như 'person' là 0.
   Nó không phải tracking ID vì tracking ID dùng để khớp một đối tượng duy nhất xuyên suốt nhiều frame của video, còn ở đây chỉ là một ảnh tĩnh đơn lẻ.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   "Vẽ đa giác (polygon) ôm sát và chính xác theo đường viền thực tế của đối tượng. Không cắt lẹm vào trong cơ thể đối tượng quá 2 pixel và không lấy lấn ra ngoài phần phông nền xung quanh."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   Các chiếc thìa (spoon) hoặc nồi treo đè lên nhau ở phía trên bên trái.
   Em sẽ ưu tiên tra cứu guideline trước. Vì hầu hết các trường hợp chồng lấn (occlusion) cơ bản đã được định nghĩa trong quy chuẩn gán nhãn (ví dụ: vẽ phần nhìn thấy được, bỏ qua phần khuất hoàn toàn). Nếu gặp trường hợp quá mờ hoặc dị dạng không thể đoán nổi lớp thì mới dùng escalation để hỏi Mentor/Lab Coach chứ không tự ý đoán mò.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --------------------- | --------------------------------------------- | -------------------------------------------------- | ----------------- | ---------------- |
| Phân loại ảnh         |                                               |                                                    |                   |                  |
| Phát hiện vật thể     |                                               |                                                    |                   |                  |
| Instance segmentation |                                               |                                                    |                   |                  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Tuyệt đối không upload dữ liệu cá nhân, dữ liệu khách hàng hoặc các hình ảnh nội bộ chưa được cho phép lên môi trường Colab công cộng và không lưu trữ thông tin cá nhân trong báo cáo.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Sẽ ngay lập tức dừng công việc gán nhãn, bảo mật dữ liệu đó và báo cáo trực tiếp cho Lab Coach hoặc Mentor quản lý trực tiếp dự án để được xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
