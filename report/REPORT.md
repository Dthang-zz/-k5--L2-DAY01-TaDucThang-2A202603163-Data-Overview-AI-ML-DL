# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

\**Ngày chạy:*11/06/2026\*

\**Runtime Colab:*CPU\*

\*_Python / PyTorch / Ultralytics:_
Python: 3.13.18
PyTorch: 2.11.0+cpu
Ultralytics: 8.4.145 \*

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

\**Thay đổi so với notebook nguồn:*trong notebook không có dữ liệu của kitchen \* Không / mô tả rõ thay đổi

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

1. Chép record hạng 1: `class_id`= 468 ; `class_name`=cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`=ImageNet-1K.

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

- record này nói về **cả bức ảnh**. Vì yolo11n-cls.pt là mô hình thuộc nhóm Image Classification.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

- đi kèm taxonomy/checkpoit.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

- vì ba trường hợp này có ba nhiệm vụ khác nhau
  - class_id: Mã số định danh
  - class_name: Tên lớp mà mô hình dự đoán cab
  - taxonomy_name: Tên chuẩn hóa theo hệ thống phân loại

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

- guideline cần nêu rõ: chủ thể chính, mức độ chiếm ưu thế của chủ thể, ngữ cảnh, thứ tự ưu tiên, trường hợp khác không xác định

6. Vì sao không được lấy `score` của model làm ground truth?

- Vì score và ground truth là hai khái niệm khác hoàn toàn nhau.

* score là điểm số do mô hình ai dự đoán
* ground truth là nhãn thực tế được xác định thông qua quá trình gán nhãn, dựa trên guideline và dữ liệu

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record:
   `class_name`=person ; `score`=091 ; `bbox_xyxy`=[1090, 247, 1414, 1030]; `bbox_width`=324 ; `bbox_height`=783

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

- bbox_xyxy có 4 giá trị:
  - x1: Tọa độ X của góc trên bên trái.
  - y1: Tọa độ Y của góc trên bên trái.
  - x2: Tọa độ X của góc dưới bên phải.
  - y2: Tọa độ Y của góc dưới bên phải.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - Ngưỡng 1: confidence >= 0.20
   - Ngưỡng 2: confidence >= 0.60

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

- Hạ ngưỡng giúp tăng khả năng giữ lại các vật thể mà mô hình chưa tự tin, nhưng đồng thời làm tăng khối lượng công việc kiểm tra thủ công.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

- Quy định:
  - Hộp phải ôm sát vật thể, không bao gồm quá nhiều vùng nền xung quanh.
  - Góc trên bên trái và góc dưới bên phải phải xác định chính xác theo tọa độ pixel.
  - Không bỏ sót các phần nhìn thấy của vật thể.
  - Không gộp nhiều vật thể riêng biệt vào cùng một Bounding Box.
  - Với vật thể bị che khuất, chỉ vẽ theo phần nhìn thấy, không tự suy đoán phần bị che.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

- Có vấn đề chưa chắc chắn:

* Phần cơ thể nhìn thấy đã đủ để xác định chắc chắn đây là một người hay chưa?
* Bounding Box cần bao quanh phần cơ thể nhìn thấy hay phải ước lượng toàn bộ cơ thể bị khuất?

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

- không có thông số kitchen trong notebook.

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

- Bounding Box: Cho biết vật thể nằm trong khu vực hình chữ nhật nào.
- Polygon: Cho biết đường biên và hình dạng thực tế của vật thể.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

- instance_id dùng để phân biệt từng cá thể vật thể riêng biệt trong cùng một ảnh, kể cả khi chúng có cùng class_id và class_name.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

- Quy tắc:
  - Đường biên phải bám sát đường viền thực tế của vật thể.
  - Chỉ gán nhãn phần vật thể nhìn thấy trong ảnh.
  - Không vẽ Mask bao gồm vùng nền hoặc vật thể khác.
  - Khi nhiều vật thể cùng chạm nhau, phải phân tách Mask theo đường biên của từng vật thể.
  - Khi vật thể bị che khuất, chỉ vẽ phần nhìn thấy, không tự suy đoán hoặc vẽ phần bị che.
  - Mỗi vật thể riêng biệt phải có một Mask và một instance_id riêng.
  - Những vùng biên không thể xác định rõ phải được đánh dấu để reviewer kiểm tra hoặc chuyển Lab Coach xử lý theo quy trình escalation
- Nguyên tắc: Một vật thể – một Mask – một instance_id.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

- Vùng tiếp xúc giữa chiếc bát màu hồng và mặt bàn ăn ở góc dưới bên trái ảnh. Chọn guideline.

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
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

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
