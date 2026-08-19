# Table Structure Recognition cho bảng không viền — Giải pháp Production

> Tài liệu kỹ thuật tổng hợp các hướng tiếp cận, chi phí tài nguyên và lộ trình triển khai
> cho bài toán nhận dạng cấu trúc bảng (TSR) tổng quát, trọng tâm là **bảng không viền
> (wireless/borderless)** và **bảng lai** có merged cell, rowspan, colspan.
>
> Ưu tiên: mã nguồn mở dùng được cho mục đích thương mại. Các phương án trả phí được
> liệt kê riêng để tham khảo.
>
> Cập nhật: 08/2026

---

## Mục lục

- [0. Chẩn đoán vấn đề](#0-chẩn-đoán-vấn-đề)
- [PHẦN I — Giải pháp trong hệ sinh thái PaddlePaddle](#phần-i--giải-pháp-trong-hệ-sinh-thái-paddlepaddle)
- [PHẦN II — Giải pháp ngoài PaddlePaddle](#phần-ii--giải-pháp-ngoài-paddlepaddle)
- [PHẦN III — VLM nhỏ làm fallback](#phần-iii--vlm-nhỏ-làm-fallback)
- [PHẦN IV — Tài nguyên production tổng hợp](#phần-iv--tài-nguyên-production-tổng-hợp)
- [PHẦN V — Giấy phép & phương án trả phí](#phần-v--giấy-phép--phương-án-trả-phí)
- [PHẦN VI — Đánh giá & metric](#phần-vi--đánh-giá--metric)
- [PHẦN VII — Lộ trình triển khai](#phần-vii--lộ-trình-triển-khai)
- [Phụ lục](#phụ-lục)

---

## 0. Chẩn đoán vấn đề

### 0.1 Failure mode cốt lõi

Lỗi phổ biến nhất trong pipeline TSR cho bảng không viền là **một cell nhiều dòng bị tách
thành nhiều hàng**, kéo theo hiệu ứng domino làm hỏng toàn bộ bảng phía dưới.

Đây là failure mode đã được ghi nhận trong literature. Bài đánh giá TDATR (2026) phân loại
các ca khó trên iFLYTAB-full và xác định loại lỗi hàng đầu là **"boundary confusion"**:
trong bảng không viền chứa text nhiều dòng, model không phân biệt được khoảng cách giữa
các dòng chữ (line spacing) với ranh giới phân tách cell (cell delimiter).

### 0.2 Nguyên nhân gốc rễ — ba tầng

**Tầng 1 — Sai primitive.** Cell bounding box là đối tượng khó dự đoán nhất trong bảng
không viền vì ranh giới của nó *không tồn tại trong ảnh*. Cell rỗng đặc biệt khó vì không
có bất kỳ đặc trưng thị giác nào. Literature đã chuyển hướng: nhận diện **đường phân tách**
(separator) hiệu quả hơn detect trực tiếp vùng hàng/cột, và hiệu quả hơn nhiều so với
detect từng cell.

**Tầng 2 — Thiếu thông tin.** Câu hỏi "hai dòng chữ này thuộc một cell hay hai cell" là
**bài toán không xác định về mặt thuần thị giác**. Cùng một khoảng cách pixel có thể là
line spacing trong cell, hoặc ranh giới giữa hai hàng. Con người phân biệt được vì đọc
nội dung. Một model thuần vision không có thông tin đó → tồn tại **trần chính xác** không
thể vượt qua bằng cách fine-tune thêm.

**Tầng 3 — Kiến trúc chuỗi cam kết cứng.** Mỗi tầng ra quyết định rời rạc mà tầng sau
không được phép xét lại. Một lỗi cục bộ trở thành lỗi toàn cục.

### 0.3 Ba mục tiêu khác nhau — cần chọn đúng

| Mục tiêu | Khả thi? | Ghi chú |
|---|---|---|
| TEDS-Struct trung bình ≥ 95% trên tập wireless | Khả thi | SOTA hiện đã ở vùng này trên benchmark chuẩn |
| Tỉ lệ bảng đúng **hoàn toàn** (exact match) ≥ 95% | Chưa hệ thống nào công bố đạt | Không nên cam kết cho bảng không viền tổng quát |
| ≥ 95% bảng đạt ngưỡng chất lượng, phần còn lại **được phát hiện** và route sang review | Khả thi | **Đây là mục tiêu production đúng đắn** |

Khoảng cách giữa mục tiêu 2 và 3 nằm hoàn toàn ở **tầng verification**, không nằm ở model.
Đầu tư vào verification cho phép đạt mục tiêu 3 với chất lượng model thấp hơn đáng kể.

### 0.4 Nguyên tắc thiết kế xuyên suốt tài liệu

1. **Dự đoán separator, không dự đoán cell box.**
2. **Đưa tín hiệu text vào quyết định merge** — phá trần thông tin ở tầng 2.
3. **Mọi tầng phải xuất confidence**, không chỉ hard decision.
4. **Có ràng buộc toàn cục** (grid hợp lệ, hàng đều nhau) thay vì ghép cục bộ tuần tự.
5. **Có tầng verification độc lập** với model sinh ra kết quả.

---

# PHẦN I — Giải pháp trong hệ sinh thái PaddlePaddle

Ưu điểm giữ Paddle: đã có hạ tầng, PP-OCRv5 chất lượng cao cho tiếng Việt, toàn bộ
Apache 2.0, deploy đã chạy ổn. Phần này liệt kê các cải tiến **không cần rời hệ sinh thái**.

---

## I.1 Nâng cấp lên Table Recognition V2 / PP-StructureV3

### Bối cảnh

Nếu đang dùng PP-Structure v1 với SLANet / SLANet_plus end-to-end, đây là nguyên nhân
chính. Pipeline v2 có kiến trúc khác hẳn:

```
Table Classification (PP-LCNet_x1_0_table_cls)  →  wired / wireless
        ↓
Table Structure Recognition (SLANeXt_wired | SLANeXt_wireless)
        ↓
Table Cell Detection (RT-DETR-L_wired_table_cell_det | RT-DETR-L_wireless_table_cell_det)
        ↓
Fusion + OCR matching  →  HTML
```

SLANeXt train trọng số riêng cho bảng có viền và không viền, cải thiện đáng kể so với
SLANet ở cả hai loại.

### Cách bật

```python
from paddleocr import TableRecognitionPipelineV2

pipeline = TableRecognitionPipelineV2(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    wired_table_structure_recognition_model_name="SLANeXt_wired",
    wireless_table_structure_recognition_model_name="SLANeXt_wireless",
    wired_table_cells_detection_model_name="RT-DETR-L_wired_table_cell_det",
    wireless_table_cells_detection_model_name="RT-DETR-L_wireless_table_cell_det",
)

output = pipeline.predict(
    "table.png",
    use_e2e_wired_table_rec_model=False,      # False = dùng cell detection
    use_e2e_wireless_table_rec_model=False,   # True  = dùng e2e SLANeXt
)
```

> **Lưu ý:** đặt `use_e2e_*=True` sẽ **vô hiệu hóa** cell detection model và dùng thẳng
> structure model sinh HTML. Với bảng không viền, khuyến nghị `False` để tận dụng
> cell detection, trừ khi benchmark nội bộ cho kết quả ngược lại.

### Điểm yếu cần vá: bảng lai

Classifier chỉ chọn **một** nhánh cho cả bảng. Với bảng lai (header có viền, body không
viền — rất phổ biến trong hóa đơn, báo cáo), đây là điểm gãy.

**Vá 1 — Chạy song song, chọn theo consistency score:**

```python
result_wired    = predict_with_branch(img, branch="wired")
result_wireless = predict_with_branch(img, branch="wireless")

score_w  = consistency_score(result_wired, ocr_boxes)
score_wl = consistency_score(result_wireless, ocr_boxes)

final = result_wired if score_w > score_wl else result_wireless
```

Chi phí gấp đôi ở tầng structure nhưng RT-DETR-L vẫn rẻ. Xem [I.4](#i4-multi-hypothesis--reranking)
cho hàm `consistency_score`.

**Vá 2 — Phân vùng theo line coverage:**

```python
import cv2, numpy as np

def line_coverage_map(gray, min_len_ratio=0.3):
    """Trả về mask đường kẻ ngang và dọc."""
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, -2)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w*min_len_ratio), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h*min_len_ratio)))
    horiz = cv2.dilate(cv2.erode(bw, hk), hk)
    vert  = cv2.dilate(cv2.erode(bw, vk), vk)
    return horiz, vert
```

Chia bảng thành các dải ngang, tính line coverage từng dải. Dải coverage cao → nhánh wired,
dải còn lại → nhánh wireless, rồi merge grid theo trục cột chung.

**Vá 3 — Dùng đường dọc làm ràng buộc cứng.** Rất nhiều bảng "không viền" thực chất có
đầy đủ đường **dọc** nhưng thiếu đường **ngang**. Đây là ca dễ nhất mà pipeline hay bỏ lỡ:
detect đường dọc bằng morphology → chốt cứng biên cột → chỉ còn phải giải bài toán 1D
phân hàng. Nên có nhánh riêng cho ca này.

### Chi phí

| Hạng mục | Ước tính |
|---|---|
| Công triển khai | 3–5 ngày |
| Thay đổi hạ tầng | Không |
| Tăng latency | +30–60% (thêm cell det + classification) |
| Tăng VRAM | +~1.5 GB (RT-DETR-L ×2 nếu chạy song song) |

---

## I.2 Đổi HTML sang OTSL cho model im2seq

### Vấn đề với HTML làm target

SLANet/SLANeXt sinh chuỗi token HTML. HTML là ngôn ngữ tồi cho việc này:

- Vocabulary lớn (28+ token), chuỗi dài → nhiều cơ hội sai.
- Không đảm bảo tính chữ nhật: model có thể sinh chuỗi **hợp lệ cú pháp** nhưng các hàng
  có số cột khác nhau → bạn phải vá bằng rules.
- Không có cơ chế phát hiện lỗi trong lúc decode.

### OTSL (Optimized Table Structure Language)

OTSL giảm số token xuống **5** (HTML cần 28+) và rút ngắn độ dài chuỗi còn khoảng **một
nửa**. Quan trọng hơn: độ chính xác model cải thiện đáng kể, thời gian inference giảm một
nửa, và **cấu trúc sinh ra luôn đúng cú pháp** — loại bỏ phần lớn nhu cầu hậu xử lý.

OTSL mô tả bảng dựa trên lưới 2D nguyên tử, cho phép phát hiện và sửa lỗi ngay trong quá
trình sinh chuỗi. MinerU2.5 đã chọn OTSL làm target VLM vì lý do này rồi convert sang HTML
ở bước cuối.

Bộ token:

| Token | Nghĩa |
|---|---|
| `C` | Cell mới (có hoặc không có nội dung) |
| `L` | Nối sang trái (colspan) |
| `U` | Nối lên trên (rowspan) |
| `X` | Nối chéo (2D span) |
| `NL` | Xuống dòng mới |

Vì mỗi hàng có **độ dài token cố định** bằng số cột, lỗi "hàng lệch số cột" bị loại bỏ về
mặt cấu trúc.

### Cách triển khai trên Paddle

1. Viết converter `HTML ↔ OTSL` (khoảng 200 dòng Python, deterministic hai chiều).
2. Convert lại toàn bộ label của tập train (PubTabNet, synthetic, data nội bộ).
3. Đổi vocabulary và `max_seq_len` trong config SLANeXt (giảm được ~50%).
4. Retrain / fine-tune.
5. Convert output OTSL → HTML ở inference. **Yêu cầu output HTML của bạn vẫn được đáp ứng.**

Thêm validation trong lúc decode:

```python
def otsl_valid_next_tokens(grid_state, n_cols):
    """Trả về mask token hợp lệ tại bước tiếp theo — dùng cho constrained decoding."""
    col = grid_state.current_col
    valid = {"C"}
    if col > 0 and grid_state.left_is_cell_or_L():
        valid.add("L")
    if grid_state.row > 0 and grid_state.above_is_cell_or_U():
        valid.add("U")
    if col > 0 and grid_state.row > 0 and grid_state.can_x():
        valid.add("X")
    if col == n_cols - 1:
        valid = {"NL"}
    return valid
```

Áp mask này lên logits khi decode → **không thể sinh ra cấu trúc sai**.

### Chi phí

| Hạng mục | Ước tính |
|---|---|
| Công triển khai | 1–2 tuần (converter + retrain) |
| Rủi ro | Thấp — thay đổi cục bộ, có thể A/B |
| Lợi ích latency | Giảm ~40–50% thời gian decode |
| Lợi ích accuracy | Cải thiện có ý nghĩa, đặc biệt bảng dài |

> **Đây là cải tiến ROI cao nhất nếu bạn muốn giữ nguyên kiến trúc im2seq.**

---

## I.3 Merge head đa mô thức — phá trần thông tin

### Lý do

Xem [0.2 tầng 2](#02-nguyên-nhân-gốc-rễ--ba-tầng). Bài toán "một cell hay hai cell" không
giải được bằng vision đơn thuần. Literature xác nhận: SEM (v1) fuse feature vision + text
cho mỗi grid và đạt độ chính xác cao hơn nhờ feature text. UniTabNet thêm Vision Guider và
Language Guider, đạt SOTA trên iFLYTAB và **vượt SEMv3 rõ rệt trên iFLYTAB-DP** — tập con
các bảng có mô tả dài, được chọn theo tiêu chí có nhiều text trong cell.

iFLYTAB-DP chính là bài toán của bạn được đóng gói thành benchmark.

### Thiết kế cụ thể — không cần thay kiến trúc

Giữ nguyên pipeline Paddle, thêm một **merge classifier** hậu xử lý quyết định gộp các
text line / cell dọc trục y.

**Đầu vào:** cặp (block trên, block dưới) cùng cột.

**Feature (khoảng 25–35 chiều):**

```python
def pair_features(top, bottom, table_ctx):
    """
    top, bottom: dict có bbox, text, và thuộc tính font nếu có
    table_ctx:   thống kê toàn bảng (median line height, trục cột RANSAC, ...)
    """
    mlh = table_ctx.median_line_height          # chuẩn hóa quan trọng nhất
    gap = bottom.y0 - top.y1

    return {
        # --- Hình học chuẩn hóa ---
        "gap_norm":          gap / mlh,
        "gap_vs_row_median": gap / table_ctx.median_row_gap,
        "h_ratio":           top.height / bottom.height,
        "left_align_diff":   abs(top.x0 - bottom.x0) / mlh,
        "right_align_diff":  abs(top.x1 - bottom.x1) / mlh,
        "x_overlap_ratio":   x_overlap(top, bottom) / min(top.w, bottom.w),
        "top_fills_cell":    top.width / table_ctx.col_width(top),   # dấu hiệu wrap
        "bottom_shorter":    bottom.width < top.width * 0.9,

        # --- Ngữ cảnh toàn bảng (chống domino) ---
        "n_lines_other_cols_in_band": table_ctx.count_lines_other_cols(top, bottom),
        "band_is_single_line":       table_ctx.other_cols_single_line(top, bottom),

        # --- Text (phá trần thông tin) ---
        "top_ends_punct":     top.text.rstrip().endswith((".", ";", ":", "!", "?")),
        "bottom_starts_lower": bottom.text[:1].islower(),
        "bottom_starts_conj":  bottom.text.split()[:1] in CONTINUATION_WORDS,
        "top_is_numeric":      is_numeric(top.text),
        "bottom_is_numeric":   is_numeric(bottom.text),
        "type_match":          data_type(top.text) == data_type(bottom.text),
        "top_unclosed_paren":  count_unclosed(top.text) > 0,
        "bottom_in_paren":     bottom.text.strip().startswith("("),
        "top_len_ratio":       len(top.text) / max(len(bottom.text), 1),

        # --- Thị giác cục bộ ---
        "line_pixel_between":  table_ctx.has_horizontal_line_between(top, bottom),
        "ink_density_gap":     table_ctx.ink_density_in_gap(top, bottom),
        "font_size_match":     abs(top.font_size - bottom.font_size) < 0.5,
        "italic_match":        top.italic == bottom.italic,
        "bold_match":          top.bold == bottom.bold,
    }
```

**Ba feature quan trọng nhất, theo kinh nghiệm:**

1. `gap_norm` — chuẩn hóa theo **median line height của chính bảng đó**, không phải giá trị
   tuyệt đối. Đây là chuẩn hóa hay bị quên nhất và ảnh hưởng lớn nhất.
2. `band_is_single_line` — nếu các cột *khác* trong cùng dải y chỉ có một dòng, thì hai
   dòng ở cột này gần như chắc chắn thuộc cùng một cell. Đây là ràng buộc toàn cục mạnh.
3. `bottom_starts_lower` — dấu hiệu wrap gần như tuyệt đối trong văn bản Latin và tiếng Việt.

**Model:** LightGBM hoặc XGBoost là đủ. Vài nghìn mẫu train. Xuất **xác suất**, không phải
hard decision — cần cho reranking ở I.4.

### Sinh nhãn tự động

Từ synthetic data render bằng HTML/DOM, nhãn `same-cell` lấy được miễn phí: hai text line
thuộc cùng `<td>` → label = 1.

### Chi phí

| Hạng mục | Ước tính |
|---|---|
| Công triển khai | 1–1.5 tuần |
| Tài nguyên inference | Không đáng kể (<5 ms/bảng, CPU) |
| Rủi ro | Rất thấp — module độc lập, dễ rollback |

> **Đây là thay đổi đơn lẻ có ROI cao nhất cho failure mode cụ thể của bạn.**

---

## I.4 Multi-hypothesis + reranking

Không cần train model mới. Sinh K bảng ứng viên rồi chấm điểm.

### Nguồn giả thuyết

```python
hypotheses = []
for theta in [-1.0, -0.5, 0.0, 0.5, 1.0]:          # góc xoay
    for branch in ["wired", "wireless"]:            # nhánh Paddle
        for thr in [0.3, 0.5, 0.7]:                 # ngưỡng cell det
            hypotheses.append(run_pipeline(img, theta, branch, thr))
```

Thực tế nên giới hạn K ≈ 6–10 để kiểm soát latency. Chạy song song bằng batch inference.

### Hàm chấm điểm (domain-agnostic)

```python
def consistency_score(table, ocr_boxes, img):
    s = {}
    # 1. Quan trọng nhất: OCR box nằm trọn trong đúng MỘT cell
    s["ocr_containment"] = frac_boxes_fully_inside_exactly_one_cell(table, ocr_boxes)

    # 2. Không có đường lưới cắt ngang chữ
    s["no_text_cut"] = 1 - frac_cells_whose_border_crosses_ink(table, img)

    # 3. Đồng nhất kiểu dữ liệu theo cột
    s["col_type_purity"] = mean(column_type_entropy_inverse(table))

    # 4. Đều đặn chiều cao hàng
    s["row_height_regularity"] = 1 - cv(row_heights(table))

    # 5. Đối xứng hàng: số cell không rỗng mỗi hàng ổn định
    s["row_fill_consistency"] = 1 - frac_rows_deviating_from_mode(table)

    # 6. Tỉ lệ cell rỗng hợp lý
    s["empty_ratio_penalty"] = penalty_if_empty_ratio_extreme(table)

    # 7. Căn lề: phương sai vị trí mép trái / dấu thập phân trong cột
    s["alignment_variance"] = 1 - mean_column_alignment_variance(table)

    return weighted_sum(s, WEIGHTS)
```

**Ghi chú về `col_type_purity`:** đây là feature tổng quát hơn nhiều người tưởng. Mọi bảng
thật đều có cột đồng nhất về kiểu (số / ngày / text / mã). Một cột lẫn 90% số với 10% text
gần như luôn là dấu hiệu lệch cột. Không cần biết đó là hóa đơn, báo cáo hay bảng khoa học.

**Học trọng số:** dùng golden set + structured hinge loss, hoặc đơn giản là logistic
regression trên cặp (ứng viên đúng, ứng viên sai). LightGBM ranker cũng phù hợp.

### Chi phí

| Hạng mục | Ước tính |
|---|---|
| Công triển khai | 1 tuần |
| Tăng latency | ×K (giảm được bằng batching) |
| Tăng VRAM | Không (tuần tự) hoặc ×K (song song) |
| Lợi ích | Thường +2–5 điểm TEDS, không cần retrain |

---

## I.5 Ràng buộc toàn cục bằng CP-SAT

Thay rules cố định bằng bài toán tối ưu có ràng buộc. Đây là chỗ **chấm dứt hiệu ứng domino**.

```python
from ortools.sat.python import cp_model

def solve_table_assignment(cells, n_rows_max, n_cols_max, scores):
    m = cp_model.CpModel()

    r0 = [m.NewIntVar(0, n_rows_max-1, f"r0_{i}") for i in range(len(cells))]
    c0 = [m.NewIntVar(0, n_cols_max-1, f"c0_{i}") for i in range(len(cells))]
    rs = [m.NewIntVar(1, n_rows_max,   f"rs_{i}") for i in range(len(cells))]
    cs = [m.NewIntVar(1, n_cols_max,   f"cs_{i}") for i in range(len(cells))]

    # --- Ràng buộc cứng ---
    for i in range(len(cells)):
        m.Add(r0[i] + rs[i] <= n_rows_max)
        m.Add(c0[i] + cs[i] <= n_cols_max)

    # Không chồng lấn (dùng NoOverlap2D)
    x_iv = [m.NewIntervalVar(c0[i], cs[i], c0[i]+cs[i], f"x_{i}") for i in range(len(cells))]
    y_iv = [m.NewIntervalVar(r0[i], rs[i], r0[i]+rs[i], f"y_{i}") for i in range(len(cells))]
    m.AddNoOverlap2D(x_iv, y_iv)

    # Bảo toàn thứ tự hình học: A hoàn toàn trên B  =>  row(A) <= row(B)
    for i, j in geometric_above_pairs(cells):
        m.Add(r0[i] <= r0[j])

    # --- Hàm mục tiêu mềm ---
    cost_terms = []
    for i in range(len(cells)):
        cost_terms.append(assignment_cost(i, r0[i], c0[i], scores))   # linearize
    m.Minimize(sum(cost_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 0.5
    solver.Solve(m)
    return extract_solution(solver, r0, c0, rs, cs)
```

**Giá trị:** một detection sai chỉ chịu phạt cục bộ thay vì lệch mọi thứ phía sau. Output
luôn là bảng hợp lệ. Bảng vài trăm cell giải trong <100 ms.

**Chi phí:** 1.5–2 tuần triển khai, +50–100 ms/bảng trên CPU.

---

## I.6 Kế hoạch dữ liệu cho Paddle

### Dataset công khai

| Dataset | Quy mô | Đặc điểm | Giấy phép |
|---|---|---|---|
| **iFLYTAB** | Lớn | 4 tập con: Wired-Digital, Wired-Camera, **Wireless-Digital, Wireless-Camera** | Kiểm tra repo SEMv2 |
| **PubTabNet** | 500k train / 9k val | Bài báo khoa học, HTML + bbox | CDLA-Permissive |
| **FinTabNet** | ~113k | Báo cáo tài chính, chủ yếu **không viền** | CDLA-Permissive |
| **PubTables-1M** | ~948k | Annotation đầy đủ row/column/spanning cell + **cell rỗng** | CDLA-Permissive 2.0 |
| **SciTSR / SciTSR-COMP** | 15k / 716 | SciTSR-COMP toàn bảng có span cell | MIT |
| **WTW** | — | Bảng có viền, ảnh chụp tự nhiên | Kiểm tra |

> **iFLYTAB là dataset quan trọng nhất** — nó được thiết kế riêng cho bài toán này. Tác giả
> nêu rõ WTW chỉ tập trung bảng có viền, trong khi phân tích bảng không viền khó hơn tương
> đối do thiếu tín hiệu thị giác để phân định cell, hàng và cột.
>
> Bốn tập con của iFLYTAB chính là cách bạn nên tổ chức metric nội bộ. Nếu đang báo cáo
> một con số TEDS duy nhất, bạn đang tự che mắt.

**Về giấy phép dataset:** PubTabNet, FinTabNet dùng CDLA-Permissive; PubTables-1M dùng
CDLA-Permissive 2.0. CDLA-Permissive cho phép sử dụng thương mại. Tuy nhiên **cần đọc kỹ
điều khoản gốc và tham vấn pháp lý** trước khi dùng cho sản phẩm thương mại — đặc biệt về
nghĩa vụ attribution và điều khoản với "Results" (output của model train trên data đó).

### Synthetic data — vấn đề nằm ở phân phối, không phải số lượng

Checklist bắt buộc cho tập synthetic:

- [ ] **Cell nhiều dòng (wrap)** — over-sample mạnh, mục tiêu ≥ 30% số cell body.
- [ ] **Line spacing trong cell ≈ row spacing** — chính là vùng mơ hồ. Nếu synthetic luôn
      có row spacing rộng rãi, model học một ngưỡng khoảng cách và ngưỡng đó sẽ vỡ trên
      dữ liệu thật. **Đây là lỗi phân phối phổ biến nhất.**
- [ ] Cell rỗng, đặc biệt rỗng thành cụm và rỗng ở cột đầu.
- [ ] Bảng lai: viền một phần, chỉ đường dọc, chỉ viền header, chỉ đường phân nhóm.
- [ ] Merged cell: rowspan, colspan, và span 2D.
- [ ] Header nhiều tầng.
- [ ] Nghiêng ±3°, keystone nhẹ, cong nhẹ.
- [ ] Nhiễu scan, watermark, JPEG artifact, độ phân giải thấp.
- [ ] Bảng dài (>50 hàng) để kiểm tra giới hạn sequence length.
- [ ] Đa ngôn ngữ nếu cần: tiếng Việt có dấu làm tăng chiều cao dòng.

**Pipeline sinh:**

```
HTML/CSS template (randomized) 
    → headless Chrome / WeasyPrint → PDF 
    → pdf2image → PNG
    → DOM query để lấy bbox từng <td>, <tr>, và từng text line
    → xuất label: OTSL/HTML + cell bbox + separator + nhãn same-cell
```

Ưu điểm: mọi nhãn (kể cả `same-cell` cho merge head ở I.3) đều lấy được miễn phí từ DOM.

### Hard negative mining

Chạy model hiện tại trên toàn bộ dữ liệu thật chưa gán nhãn, dùng `consistency_score`
(I.4) để lọc ra các mẫu model không chắc chắn, **chỉ gán nhãn nhóm đó**. Hiệu quả trên mỗi
nhãn cao hơn nhiều lần so với gán nhãn ngẫu nhiên.

---

## I.7 Tài nguyên production — nhánh Paddle

**Giả định:** GPU NVIDIA L4 (24 GB) hoặc T4 (16 GB), FP16, ảnh bảng đã crop ~1000×1400 px.
Các con số dưới đây là **ước tính** dựa trên kích thước kiến trúc và benchmark công bố;
cần đo lại trên hạ tầng thực tế.

| Module | Params (xấp xỉ) | VRAM (FP16) | Latency GPU | Latency CPU |
|---|---|---|---|---|
| PP-LCNet_x1_0_table_cls | ~3 M | ~50 MB | ~3 ms | ~15 ms |
| SLANet_plus | ~9 M | ~150 MB | ~30 ms | ~300 ms |
| SLANeXt_wired/wireless | ~25–40 M | ~300 MB mỗi model | ~50–80 ms | ~600 ms |
| RT-DETR-L cell det | ~32 M | ~600 MB | ~40–60 ms | ~800 ms |
| PP-OCRv5 det + rec | ~20 M | ~500 MB | ~80–150 ms | ~1–2 s |
| Merge head (LightGBM) | — | — | <5 ms | <5 ms |
| CP-SAT solver | — | — | — | 50–100 ms |

**Cấu hình pipeline v2 đầy đủ (1 worker):**

| Chỉ số | Ước tính |
|---|---|
| VRAM thường trú | ~2.5–3.5 GB |
| Latency/bảng (GPU) | ~250–400 ms |
| Throughput 1×L4, batch 8 | ~15–25 bảng/s |
| Throughput 1×T4, batch 4 | ~8–12 bảng/s |
| CPU-only (8 vCPU) | ~0.5–1 bảng/s |

**Nếu bật multi-hypothesis (K=6):**

| Chỉ số | Ước tính |
|---|---|
| VRAM | Không đổi nếu chạy tuần tự; ~5–6 GB nếu batch song song |
| Latency/bảng | ~800 ms – 1.2 s |
| Throughput 1×L4 | ~4–8 bảng/s |

**Chi phí cloud tham khảo** (giá on-demand, cần kiểm tra lại vì biến động):

| Instance | Giá/giờ (~) | Throughput (bảng/s) | Chi phí/1M bảng |
|---|---|---|---|
| 1×T4 (g4dn.xlarge) | $0.50–0.60 | ~10 | ~$14–17 |
| 1×L4 (g6.xlarge) | $0.80–1.00 | ~20 | ~$11–14 |
| 1×A10G (g5.xlarge) | $1.00–1.20 | ~25 | ~$11–13 |
| CPU 8 vCPU (c6i.2xlarge) | $0.34 | ~0.7 | ~$135 |

> Kết luận: với TSR thuần, GPU rẻ hơn CPU khoảng **10×** trên mỗi đơn vị công việc. Chỉ
> chọn CPU nếu volume rất thấp (<10k bảng/ngày) hoặc có ràng buộc on-premise.

---

# PHẦN II — Giải pháp ngoài PaddlePaddle

Phần này dành cho trường hợp bạn sẵn sàng thay module TSR bằng kiến trúc khác. Lưu ý:
**bạn không cần bỏ toàn bộ Paddle** — PP-OCRv5 vẫn là lựa chọn tốt cho text detection và
recognition tiếng Việt. Chỉ thay tầng structure.

---

## II.1 Split-and-Merge — khuyến nghị mạnh nhất

### Nguyên lý

Thay vì detect N cell, dự đoán tập **đường phân tách** hàng và cột → sinh lưới mịn →
merge module quyết định các ô lưới nào gộp thành cell logic.

```
Ảnh bảng
   ↓
[SPLIT]  Regress separator hàng + cột  →  lưới mịn (fine grid)
   ↓
[EMBED]  Feature cho từng ô lưới (vision + optional text)
   ↓
[MERGE]  Classify từng cặp ô kề nhau: gộp / không gộp
   ↓
Cấu trúc logic → HTML
```

**Tại sao giải đúng bài toán của bạn:**

| Vấn đề | Cell detection | Split-and-Merge |
|---|---|---|
| Bài toán | 2D, N đối tượng, không ràng buộc | 1D, ít đối tượng, ràng buộc thứ tự mạnh |
| Cell rỗng | Không có visual feature → thường miss | Có sẵn từ lưới |
| Cell nhiều dòng bị tách | Lỗi detection **không phục hồi được** | Một dự đoán merge=True, có supervision trực tiếp |
| Lan lỗi | Domino | Cục bộ |
| Span | Suy từ hình học bằng rules | Chính là cơ chế merge |

### Các biến thể theo thứ tự ưu tiên

#### SEMv3 — SOTA cho bảng không viền

SEMv3 giới thiệu module **Keypoint Offset Regression (KOR)** regress trực tiếp offset của
đường phân tách so với các keypoint proposal, cùng một tập **merge action** định nghĩa cấu
trúc bảng dựa trên grid.

Đạt SOTA trên ICDAR-2019 cTDaR Historical, WTW và iFLYTAB. **Ablation cho thấy module split
KOR cải thiện đáng kể riêng trên bảng không viền** — đúng phân khúc bạn đang thua.

#### SEMv2 — điểm khởi đầu thực dụng

Có **code công khai** (`github.com/ZZR8066/SEMv2`) kèm evaluation code (TEDS + F1-Measure)
và dataset iFLYTAB. Đây là baseline chạy được trong vài ngày.

> **Cảnh báo giấy phép:** cần kiểm tra file LICENSE của repo trước khi dùng thương mại.
> Nhiều repo học thuật Trung Quốc không ghi rõ license, mặc định là **all rights reserved**.
> Nếu không có license rõ ràng, hãy dùng repo để **học kiến trúc và tự implement lại**,
> hoặc liên hệ tác giả xin phép. Đây là rủi ro pháp lý thật, không phải hình thức.

#### RobusTabNet

Dùng spatial CNN dự đoán đường phân tách chia bảng thành lưới, rồi Grid CNN merge để khôi
phục spanning cell. Kiến trúc đơn giản, dễ tự implement từ paper.

#### TSRFormer / SepFormer — hướng DETR

TSRFormer dùng split module dựa trên **SepRETR** để regress đường phân tách, cộng relation
network để merge spanning cell. Cách regress này đạt độ chính xác cao hơn các phương pháp
segmentation **mà không cần module heuristic chuyển mask thành đường**.

SepFormer là cách tiếp cận **coarse-to-fine**, dự đoán separator từ đoạn đơn đến dải, với
**loss góc bổ sung** ở giai đoạn thô.

> **Loss góc chính là lời giải cho vấn đề nghiêng 1–2°.** Model học separator *nghiêng*
> thay vì bạn phải deskew trước. Bỏ được cả một tầng lỗi tích lũy.
>
> Đây là lý do tôi khuyên **bỏ deskew như một bước tiền xử lý riêng** — deskew là phép biến
> đổi có mất mát (resample làm nhòe chữ nhỏ) và sai số ước lượng góc vẫn còn nguyên đó.
> Hãy để model chịu trách nhiệm về hình học.

### Nhược điểm cần biết trước

Split-and-merge không hoàn hảo. Chính bài TDATR chỉ ra **SEMv3 dễ nhầm lẫn separator với
khoảng trắng giữa các từ**. Nghĩa là bạn đổi lỗi "tách nhầm hàng" lấy lỗi "tách nhầm cột"
ở bảng có khoảng cách từ rộng.

**Giảm thiểu:** thêm ràng buộc số cột toàn cục (số cột phải nhất quán giữa các hàng — vốn
đã có trong biểu diễn OTSL), và thêm feature text vào split head cho trục dọc (khoảng trắng
giữa hai từ trong cùng cell có đặc trưng khác khoảng trắng giữa hai cột: hai bên cùng kiểu
dữ liệu, độ rộng khe không nhất quán qua các hàng).

### Chi phí triển khai

| Hạng mục | Ước tính |
|---|---|
| Dựng baseline từ code SEMv2 | 1–2 tuần |
| Tự implement từ paper (nếu vướng license) | 4–6 tuần |
| Train từ đầu trên iFLYTAB + synthetic | 3–7 ngày trên 1×A100 hoặc 2×L4 |
| Fine-tune từ checkpoint | 1–2 ngày |

---

## II.2 Table Transformer (TATR) — lựa chọn an toàn nhất về pháp lý

### Đặc điểm

- Repo: `github.com/microsoft/table-transformer`
- **Giấy phép MIT** — an toàn tuyệt đối cho thương mại. Đây là điểm mạnh lớn nhất.
- Kiến trúc DETR: detect **row, column, spanning cell, header** như các đối tượng riêng biệt,
  rồi lấy giao điểm để suy ra cell.
- Train trên PubTables-1M (CDLA-Permissive 2.0), có annotation đầy đủ cho cả **cell rỗng**.

### Vì sao phù hợp với bảng không viền

TATR **không dự đoán đường kẻ** — nó dự đoán vùng hàng và vùng cột như object. Bảng không
viền vẫn có vùng hàng/cột rõ ràng về mặt ngữ nghĩa dù không có đường kẻ. Do đó nó không bị
phụ thuộc vào tín hiệu border.

Cell rỗng được xử lý tự nhiên vì cell = giao của row × column, không cần detect riêng.

### Điểm yếu

- Spanning cell được detect như một class riêng và độ chính xác thấp hơn row/column.
  Bảng nhiều span sẽ yếu.
- Train chủ yếu trên bài báo khoa học (PubTables-1M) và tài chính (FinTabNet.c) → domain
  gap với tài liệu tiếng Việt, hóa đơn, biểu mẫu hành chính. **Bắt buộc fine-tune.**
- Không robust với ảnh nghiêng/cong — cần deskew hoặc augment mạnh khi fine-tune.

### Chiến lược lai đáng thử

**TATR cho row/column + Split-Merge cho merge + PP-OCRv5 cho text.** TATR cho ranh giới
hàng/cột ổn định (bài toán dễ), tầng merge riêng xử lý span và multi-line (bài toán khó).
Tách hai bài toán ra giúp debug và fine-tune độc lập.

### Chi phí

| Hạng mục | Ước tính |
|---|---|
| Params | ~28 M (DETR-R18) |
| VRAM FP16 | ~800 MB – 1.2 GB |
| Latency GPU (L4) | ~60–100 ms/bảng |
| Latency CPU | ~1.5–2.5 s/bảng |
| Fine-tune trên 20k mẫu | ~1–2 ngày trên 1×A100 |
| Công tích hợp | 1 tuần |

---

## II.3 Docling + TableFormer (IBM)

### Đặc điểm

- Package: `docling` — **MIT license**, self-contained.
- Powered by DocLayNet (layout) và **TableFormer** (table structure).
- Chạy hiệu quả trên **commodity hardware trong ngân sách tài nguyên nhỏ**.
- Pipeline nạp page image ở 72 dpi, xử lý được trên **một CPU với latency dưới một giây**.
- TableFormer dùng **custom structure token language** (chính là OTSL).

### Vì sao đáng cân nhắc

TableFormer xử lý được nhiều đặc tính bảng khó: **viền một phần hoặc không viền, cell rỗng,
hàng/cột rỗng, cell span, và phân cấp trên cả column-heading lẫn row-heading**. Đây là mô
tả sát với bài toán của bạn.

Ngoài ra, các bounding box dự đoán được **giao với text token từ PDF** để gom thành đơn vị
hoàn chỉnh — nghĩa là Docling khai thác text layer khi có, đúng nguyên tắc "PDF số thì
đừng OCR".

### Điểm mạnh riêng cho production

- **MIT + package hoàn chỉnh** = thời gian tích hợp ngắn nhất trong toàn bộ tài liệu này.
- CPU-friendly → có thể chạy on-premise không cần GPU nếu volume vừa phải.
- Có sẵn xuất Markdown/JSON/HTML.

### Điểm yếu

- Khó fine-tune sâu hơn so với tự train SEMv2/TATR (pipeline đóng gói kín hơn).
- Tối ưu cho tài liệu tiếng Anh; cần kiểm tra chất lượng trên tài liệu tiếng Việt.
- Chất lượng OCR nội tại kém hơn PP-OCRv5 cho tiếng Việt → nên dùng **chỉ tầng TableFormer**
  và ghép với PP-OCRv5.

### Chi phí

| Hạng mục | Ước tính |
|---|---|
| Params TableFormer | ~20–30 M |
| VRAM FP16 | ~600 MB – 1 GB |
| Latency CPU | <1 s/trang (theo công bố, ở 72 dpi) |
| Latency GPU | ~50–80 ms/bảng |
| Công tích hợp | **3–5 ngày** (nhanh nhất) |

---

## II.4 UniTable / UniTabNet — hướng đa mô thức

### UniTable

Framework thống nhất cho table recognition với self-supervised pretraining. Kết quả công bố
trên FinTabNet: **UniTable Large đạt S-TEDS 98.89**, so với TableFormer 96.80 và OTSL 95.90.

Đạt SOTA trên 4/5 benchmark bảng lớn nhất.

> **Lưu ý về con số:** FinTabNet là bảng tài chính bố cục khá đều đặn. Con số 98.89 không
> tự động chuyển sang bảng không viền dạng biểu mẫu/hóa đơn. Vẫn cần benchmark nội bộ.

### UniTabNet

Bridging Vision và Language Models. Có **Vision Guider** và **Language Guider**. Đạt SOTA
mới trên iFLYTAB, và **vượt SEMv3 rõ rệt trên iFLYTAB-DP** (tập bảng có mô tả dài, nhiều
text trong cell).

Đây là bằng chứng trực tiếp nhất trong literature cho luận điểm ở [0.2 tầng 2](#02-nguyên-nhân-gốc-rễ--ba-tầng):
model đa mô thức vượt model thuần thị giác chính xác ở phân khúc cell nhiều dòng.

### Cân nhắc

- Kiến trúc autoregressive → chậm hơn split-and-merge, và có giới hạn max decoding length.
  UniTabNet trên WTW cho **precision cao nhưng recall thấp, chủ yếu do giới hạn độ dài
  decode** — vấn đề với bảng dài.
- Cần kiểm tra license của từng repo.

### Khi nào chọn

Chọn nếu bảng của bạn **không quá dài** (<40 hàng) và **nhiều text dài trong cell**. Nếu
bảng dài, ưu tiên split-and-merge (non-autoregressive, không có giới hạn độ dài).

---

## II.5 Kiến trúc lai tự xây — khuyến nghị dài hạn

Nếu có 2–3 tháng, đây là kiến trúc tôi đề xuất tự xây, kết hợp điểm mạnh của tất cả:

```
                    ┌─────────────────────────────────┐
                    │  Ảnh bảng (KHÔNG deskew trước)  │
                    └────────────┬────────────────────┘
                                 ↓
              ┌──────────────────┴──────────────────┐
              ↓                                     ↓
   ┌──────────────────────┐            ┌────────────────────────┐
   │ PP-OCRv5 text det    │            │ Separator regression   │
   │ + rec (giữ Paddle)   │            │ backbone (SepFormer-   │
   │ → text lines + text  │            │ style, có angle loss)  │
   └──────────┬───────────┘            └───────────┬────────────┘
              │                                    │
              │            ┌───────────────────────┘
              ↓            ↓
   ┌─────────────────────────────────────────────┐
   │  MERGE HEAD ĐA MÔ THỨC                      │
   │  vision feature + geometry + TEXT feature   │
   │  → P(same_cell), P(same_row), P(same_col)   │
   └──────────────────┬──────────────────────────┘
                      ↓
   ┌─────────────────────────────────────────────┐
   │  CP-SAT / ILP solver                        │
   │  ràng buộc cứng + chi phí mềm từ xác suất   │
   │  → cấu trúc luôn hợp lệ                     │
   └──────────────────┬──────────────────────────┘
                      ↓
   ┌─────────────────────────────────────────────┐
   │  OTSL (constrained) → HTML                  │
   └──────────────────┬──────────────────────────┘
                      ↓
   ┌─────────────────────────────────────────────┐
   │  VERIFICATION: render-back + consistency    │
   │  → confidence score                         │
   └──────────────────┬──────────────────────────┘
                      ↓
          score < ngưỡng ? → VLM fallback (Phần III)
                      ↓
                  Kết quả cuối
```

**Nguyên tắc:** mỗi tầng xuất **xác suất**, tầng cuối là **solver có ràng buộc**, và có
**verification độc lập**. Không tầng nào ra quyết định cứng mà tầng sau không sửa được.

### Chi phí

| Hạng mục | Ước tính |
|---|---|
| Thời gian phát triển | 2–3 tháng (1–2 kỹ sư) |
| Params tổng | ~50–70 M |
| VRAM FP16 | ~2–2.5 GB (chưa tính OCR) |
| Latency GPU | ~150–250 ms/bảng |
| Rủi ro | Trung bình — nhưng mỗi module test được độc lập |

---

## II.6 Bảng so sánh nhanh Phần II

| Giải pháp | License | Chất lượng wireless | Span | Tốc độ | Công tích hợp | Rủi ro |
|---|---|---|---|---|---|---|
| **Docling / TableFormer** | MIT ✅ | Tốt | Tốt | Nhanh (CPU OK) | 3–5 ngày | Thấp |
| **Table Transformer** | MIT ✅ | Tốt | Trung bình | Trung bình | 1 tuần | Thấp |
| **SEMv2 (code)** | ⚠️ Kiểm tra | Rất tốt | Tốt | Nhanh | 1–2 tuần | Pháp lý |
| **SEMv3 (paper)** | Tự implement | Rất tốt (SOTA) | Tốt | Rất nhanh | 4–6 tuần | Trung bình |
| **SepFormer/TSRFormer** | Tự implement | Rất tốt | Tốt | Nhanh | 4–6 tuần | Trung bình |
| **UniTable/UniTabNet** | ⚠️ Kiểm tra | Rất tốt | Rất tốt | Chậm (AR) | 2–3 tuần | Bảng dài |
| **Kiến trúc lai tự xây** | Tự sở hữu ✅ | Cao nhất | Cao nhất | Nhanh | 2–3 tháng | Trung bình |

> **Khuyến nghị:** bắt đầu bằng **Docling/TableFormer** làm baseline so sánh (nhanh, MIT,
> rủi ro thấp) song song với việc nâng cấp nhánh Paddle. Sau khi có số liệu, quyết định
> có đầu tư vào split-and-merge hay không.

---

## II.7 Chi phí chuyển đổi khỏi Paddle

| Hạng mục | Tác động |
|---|---|
| **Runtime** | Paddle → PyTorch. Nếu giữ PP-OCRv5, phải chạy **cả hai runtime** → +1.5–2 GB RAM, tăng độ phức tạp Docker image |
| **Serving** | Cần đổi từ PaddleServing/FastDeploy sang TorchServe/Triton/FastAPI |
| **Quantization** | PyTorch có hệ sinh thái tốt hơn (ONNX, TensorRT). Có thể là **lợi ích** |
| **Team** | Hầu hết kỹ sư quen PyTorch hơn → dễ tuyển và bảo trì |
| **Rollback** | Nên giữ nhánh Paddle chạy song song trong 1–2 tháng để so sánh production |

**Phương án giảm rủi ro:** export cả hai về **ONNX** và serve bằng ONNX Runtime hoặc Triton.
Khi đó runtime thống nhất, và bạn có thể A/B test hai kiến trúc trong cùng một service.

```bash
# Paddle → ONNX
paddle2onnx --model_dir ./slanext_wireless_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file slanext.onnx --opset_version 16

# PyTorch → ONNX
torch.onnx.export(model, dummy_input, "tatr.onnx", opset_version=16)
```

Ước tính: ONNX Runtime + TensorRT EP cho **tăng tốc 1.5–3×** so với runtime gốc, và giảm
VRAM ~20–30%. Đáng làm bất kể chọn kiến trúc nào.

---

# PHẦN III — VLM nhỏ làm fallback

## III.1 Định vị: fallback, không phải thay thế

Mối lo về hallucination là chính đáng. Nhưng nó có thể được **kiểm soát về mặt kỹ thuật**
nếu VLM được đặt đúng vị trí.

**Nguyên tắc cốt lõi: VLM chỉ được phép quyết định CẤU TRÚC, không được sinh NỘI DUNG.**

```
Pipeline chính (TSR)  →  cấu trúc + confidence
                              ↓
                     confidence < ngưỡng?
                              ↓ có
              VLM chỉ sinh khung cấu trúc (OTSL)
                              ↓
        Nội dung điền từ OCR box đã có, KHÔNG lấy từ VLM
                              ↓
              Verification: mọi OCR box phải được dùng đúng 1 lần
```

Với thiết kế này, VLM **không thể bịa số**, vì nó không được phép sinh ra chữ số nào cả.
Nó chỉ trả lời "bảng này có 9 cột, hàng thứ 3 có một cell span 2 cột" — và câu trả lời đó
được kiểm chứng bằng việc thử điền OCR box vào.

## III.2 Ba lớp phòng thủ chống hallucination

### Lớp 1 — Constrained decoding sang OTSL

Bắt VLM sinh **OTSL thay vì HTML**. Vì OTSL chỉ có 5 token và mỗi hàng có độ dài cố định,
bạn có thể áp **grammar-constrained decoding** (vLLM guided decoding, Outlines, XGrammar):

```python
# vLLM guided decoding với grammar
from vllm import SamplingParams

otsl_grammar = r"""
root    ::= row+
row     ::= cell{N} "NL"
cell    ::= "C" | "L" | "U" | "X"
"""

params = SamplingParams(
    temperature=0.0,
    guided_decoding=GuidedDecodingParams(grammar=otsl_grammar),
)
```

Kết quả: **model không thể sinh ra cấu trúc sai cú pháp**, và không thể sinh ra text.
Đây là biện pháp mạnh nhất và rẻ nhất.

### Lớp 2 — Đối chiếu bắt buộc với OCR

```python
def fill_and_verify(otsl_structure, ocr_boxes, cell_regions):
    assignment = assign_boxes_to_cells(ocr_boxes, cell_regions)

    checks = {
        # Mọi OCR box phải được dùng đúng 1 lần
        "all_boxes_used":   len(assignment.unused) == 0,
        "no_double_use":    len(assignment.duplicated) == 0,
        # Không có cell chứa box của cell khác
        "no_cross_cell":    assignment.cross_cell_count == 0,
        # Số cell không rỗng khớp kỳ vọng
        "fill_ratio_sane":  0.3 < assignment.fill_ratio < 1.0,
    }
    return all(checks.values()), checks
```

Nếu VLM trả cấu trúc mà OCR box không điền vừa, cấu trúc đó bị **loại**, không phải được
chấp nhận. Đây là điểm khác biệt then chốt so với dùng VLM end-to-end.

### Lớp 3 — Consensus giữa hai nguồn

Nếu pipeline TSR và VLM cho **cùng** cấu trúc → độ tin cậy rất cao, auto-accept.
Nếu khác nhau → cả hai đều đáng ngờ → human review. Không tự động chọn VLM chỉ vì nó "mới hơn".

## III.3 Thiết kế router

```python
def route(img, table_region):
    result = tsr_pipeline(img, table_region)
    conf   = verification_score(result, ocr_boxes, img)   # xem I.4 + III.2

    if conf >= 0.92:
        return result, "auto_accept"

    if conf >= 0.70:
        vlm_struct = vlm_structure_only(img)              # OTSL, constrained
        filled, ok = fill_and_verify(vlm_struct, ocr_boxes, ...)
        if ok and structures_agree(result, vlm_struct):
            return filled, "vlm_confirmed"
        if ok and verification_score(filled, ...) > conf + 0.1:
            return filled, "vlm_override"
        return result, "human_review"

    return result, "human_review"
```

**Tỉ lệ kỳ vọng (ước tính, cần đo thực tế):**

| Nhánh | Tỉ lệ traffic | Chi phí tương đối |
|---|---|---|
| auto_accept | 70–85% | 1× |
| vlm_confirmed / vlm_override | 10–20% | 5–15× |
| human_review | 3–10% | 500–2000× |

Việc chỉ gọi VLM cho 10–20% traffic khiến chi phí trung bình chỉ tăng ~1.5–3×, thay vì
5–15× nếu dùng VLM cho mọi bảng.

## III.4 Ứng viên VLM mã nguồn mở

| Model | Params | License | Ghi chú |
|---|---|---|---|
| **PaddleOCR-VL-1.6** | 0.9 B | **Apache 2.0** ✅ | 96.33% OmniDocBench v1.6; mạnh về table; cùng hệ sinh thái Paddle |
| **PaddleOCR-VL-1.5** | 0.9 B | **Apache 2.0** ✅ | 94.5% OmniDocBench v1.5; hỗ trợ localization hình dạng bất quy tắc |
| **dots.ocr** | ~1.7 B | Kiểm tra | Được benchmark cùng nhóm |
| **DeepSeek-OCR / v2** | — | Kiểm tra | Kiến trúc MoE, hợp batch lớn |
| **GOT-OCR 2.0** | ~0.6 B | Kiểm tra | Nhỏ nhất |
| **Granite-Docling** | — | Kiểm tra | Từ IBM, cùng hệ Docling |
| **MinerU2.5** | — | Kiểm tra | Dùng OTSL làm target — rất phù hợp thiết kế ở III.2 |

**Khuyến nghị: PaddleOCR-VL-1.6.**

Lý do:
1. **Apache 2.0** — an toàn thương mại, xác nhận được.
2. Chỉ 0.9 B → chi phí fallback thấp.
3. Cùng hệ sinh thái Paddle, giảm chi phí vận hành.
4. Kiến trúc **tương thích ngược hoàn toàn với 1.5, zero adaptation cost**, nên nâng cấp
   phiên bản sau này không tốn công.
5. Có bản chạy trên **vLLM/SGLang**, đã được tối ưu batch.

> **Lưu ý quan trọng:** PaddleOCR-VL series cung cấp **ít thông tin tọa độ chi tiết hơn**
> PP-StructureV3 — PP-StructureV3 mới là nhánh cho tọa độ cell chi tiết. Với thiết kế
> "VLM chỉ sinh cấu trúc, OCR điền nội dung" ở III.2, điều này không thành vấn đề, nhưng
> cần biết để không kỳ vọng sai.

## III.5 Tài nguyên cho nhánh VLM

**Giả định:** PaddleOCR-VL 0.9B, FP16, vLLM backend, ảnh bảng crop ~1000×1400.

| Chỉ số | Ước tính |
|---|---|
| Trọng số FP16 | ~1.8 GB |
| VRAM tối thiểu (KV cache nhỏ) | ~5–6 GB |
| VRAM khuyến nghị (batch tốt) | ~10–14 GB |
| Latency 1 bảng (batch=1, L4) | ~0.8–1.5 s |
| Latency 1 bảng (batch=16, L4) | ~150–300 ms amortized |
| Throughput 1×L4 (batch 16) | ~4–8 bảng/s |
| Throughput 1×A100 (batch 64) | ~20–40 bảng/s |

**Quantization:**

| Định dạng | VRAM | Tốc độ | Chất lượng |
|---|---|---|---|
| FP16 | ~1.8 GB | 1× | Chuẩn |
| INT8 (W8A8) | ~1.0 GB | ~1.3–1.6× | Gần như không giảm |
| INT4 (AWQ/GPTQ) | ~0.6 GB | ~1.8–2.5× | Giảm nhẹ, cần đo |

Với model 0.9 B, INT8 là điểm cân bằng tốt. INT4 chỉ nên dùng nếu cần chạy nhiều model
trên cùng GPU.

**Chi phí cloud cho nhánh fallback** (giả sử 15% traffic):

| Volume tổng | Bảng qua VLM | GPU cần | Chi phí/tháng (~) |
|---|---|---|---|
| 100k bảng/ngày | 15k/ngày | 1×L4 chia sẻ | $50–100 |
| 1M bảng/ngày | 150k/ngày | 1×L4 dedicated | $600–750 |
| 10M bảng/ngày | 1.5M/ngày | 2–3×A100 | $4,000–7,000 |

> So sánh: nếu dùng VLM cho **toàn bộ** traffic ở mức 1M bảng/ngày, cần ~5–7×L4 →
> $3,500–5,000/tháng. Router tiết kiệm được khoảng **80%**.

## III.6 Ý tưởng nâng cao: distill VLM thành model nhỏ

Thay vì gọi VLM ở runtime, dùng VLM để **sinh nhãn**:

1. Chạy PaddleOCR-VL-1.6 trên 200k–500k bảng thật chưa gán nhãn.
2. Lọc chỉ giữ output **đã pass verification** (III.2) — đây là bộ lọc chất lượng cực mạnh
   và tự động.
3. Dùng bộ nhãn đó fine-tune model TSR nhỏ (SEMv3/TATR).

Kết quả: bạn "hút" được năng lực của VLM vào một model 30 M params chạy 100 ms, và
**hoàn toàn không có hallucination ở runtime**. Chi phí VLM chỉ phát sinh một lần trong
giai đoạn sinh dữ liệu.

Đây có thể là hướng có ROI cao nhất trong toàn bộ tài liệu nếu bạn có volume dữ liệu thật lớn.

**Chi phí:** sinh nhãn cho 300k bảng trên 1×A100 ≈ 3–5 ngày ≈ $200–400 (spot instance).

---

# PHẦN IV — Tài nguyên production tổng hợp

## IV.1 Bảng so sánh tài nguyên toàn bộ phương án

| Phương án | Params | VRAM FP16 | Latency GPU | Latency CPU | Throughput 1×L4 |
|---|---|---|---|---|---|
| SLANet_plus (hiện tại) | ~9 M | 150 MB | 30 ms | 300 ms | ~60/s |
| PP-StructureV3 table v2 | ~70 M | 2.5–3.5 GB | 250–400 ms | 3–5 s | 15–25/s |
| + multi-hypothesis K=6 | — | 5–6 GB | 800 ms–1.2 s | — | 4–8/s |
| Docling / TableFormer | ~25 M | 0.6–1 GB | 50–80 ms | <1 s | ~30–50/s |
| Table Transformer | ~28 M | 0.8–1.2 GB | 60–100 ms | 1.5–2.5 s | ~25–40/s |
| SEMv3-style | ~35 M | 1–1.5 GB | 40–70 ms | 1–2 s | ~30–50/s |
| Kiến trúc lai (II.5) | ~60 M | 2–2.5 GB | 150–250 ms | 3–4 s | ~15–25/s |
| PaddleOCR-VL 0.9B | 0.9 B | 5–14 GB | 0.8–1.5 s (b=1) | Không khả thi | 4–8/s |

> Latency chưa bao gồm OCR text (PP-OCRv5: +80–150 ms GPU).
> Tất cả là **ước tính** dựa trên kích thước kiến trúc và benchmark công bố — **phải đo lại**
> trên ảnh và hạ tầng thực tế của bạn.

## IV.2 Ba cấu hình production tham khảo

### Cấu hình A — Tiết kiệm (< 50k bảng/ngày)

```
1× T4 (16 GB) hoặc CPU 8 vCPU
├── PP-OCRv5 (det + rec)
├── Docling/TableFormer HOẶC PP-StructureV3
├── Merge head (LightGBM, CPU)
└── Verification (CPU)
Không có nhánh VLM. Confidence thấp → human review.
```

| Chỉ số | Giá trị |
|---|---|
| VRAM | ~4 GB |
| Throughput | ~8–15 bảng/s |
| Chi phí hạ tầng | $350–450/tháng |

### Cấu hình B — Cân bằng (50k–1M bảng/ngày) ⭐ khuyến nghị

```
2× L4 (24 GB) — tách vai trò
├── GPU 1: TSR pipeline
│   ├── PP-OCRv5
│   ├── Split-and-Merge hoặc PP-StructureV3
│   ├── Merge head đa mô thức
│   ├── CP-SAT solver (CPU)
│   └── Verification + render-back
└── GPU 2: VLM fallback
    └── PaddleOCR-VL-1.6 (INT8, vLLM, batch 16)

Redis queue giữa hai tầng, batch động cho VLM.
```

| Chỉ số | Giá trị |
|---|---|
| VRAM | GPU1 ~6 GB, GPU2 ~12 GB |
| Throughput | ~20–35 bảng/s (85% qua nhánh nhanh) |
| Chi phí hạ tầng | $1,200–1,600/tháng |
| Human review | ~3–8% |

### Cấu hình C — Quy mô lớn (> 1M bảng/ngày)

```
Autoscaling group
├── 4–8× L4: TSR workers (stateless, HPA theo queue depth)
├── 2–3× A100 40GB: VLM pool (vLLM, continuous batching)
├── Redis / Kafka: hàng đợi
├── S3: lưu ảnh + output để tích lũy training data
└── Label Studio / CVAT: human review loop
```

| Chỉ số | Giá trị |
|---|---|
| Throughput | 150–300 bảng/s |
| Chi phí hạ tầng | $8,000–15,000/tháng (on-demand), $3,000–6,000 (reserved/spot) |

## IV.3 Tối ưu chi phí

| Kỹ thuật | Tiết kiệm | Công sức |
|---|---|---|
| ONNX Runtime + TensorRT | 30–60% latency | 3–5 ngày |
| INT8 quantization | 40–50% VRAM, 30% latency | 2–3 ngày |
| Dynamic batching | 2–4× throughput | 2–3 ngày |
| Spot instances cho VLM pool | 60–70% chi phí GPU | 1–2 ngày |
| Cache theo template hash | 20–50% traffic (tùy corpus) | 1 tuần |
| Bỏ qua OCR cho PDF số | 30–60% traffic khỏi vision hoàn toàn | 3–5 ngày |

> **Hai dòng cuối thường mang lại tiết kiệm lớn nhất và hay bị bỏ qua.** Với PDF sinh từ
> máy, `pdfplumber`/PyMuPDF cho tọa độ ký tự chính xác tuyệt đối, không cần OCR, không cần
> lo skew. Kiểm tra text layer trước khi vào pipeline vision là bước đầu tiên nên có.

---

# PHẦN V — Giấy phép & phương án trả phí

## V.1 Ma trận giấy phép mã nguồn mở

### Model & code

| Thành phần | Giấy phép | Thương mại | Độ tin cậy |
|---|---|---|---|
| PaddleOCR / PP-Structure / SLANeXt | **Apache 2.0** | ✅ Được | Xác nhận |
| PaddleOCR-VL (0.9B, mọi phiên bản) | **Apache 2.0** | ✅ Được | Xác nhận |
| Table Transformer (TATR) — Microsoft | **MIT** | ✅ Được | Xác nhận |
| Docling + TableFormer — IBM | **MIT** | ✅ Được | Xác nhận |
| SEMv2 / SEMv3 | ⚠️ Chưa xác định | Cần kiểm tra | **Rủi ro** |
| TSRFormer | Không có code chính thức | Tự implement | — |
| SepFormer | ⚠️ Cần kiểm tra | Cần kiểm tra | — |
| UniTable / UniTabNet | ⚠️ Cần kiểm tra | Cần kiểm tra | — |
| OR-Tools (CP-SAT) | **Apache 2.0** | ✅ Được | Xác nhận |
| LightGBM / XGBoost | **MIT / Apache 2.0** | ✅ Được | Xác nhận |
| vLLM / SGLang | **Apache 2.0** | ✅ Được | Xác nhận |

### Dataset

| Dataset | Giấy phép | Thương mại |
|---|---|---|
| PubTabNet | CDLA-Permissive | ✅ (đọc kỹ điều khoản) |
| FinTabNet | CDLA-Permissive | ✅ (đọc kỹ điều khoản) |
| PubTables-1M | CDLA-Permissive 2.0 | ✅ (đọc kỹ điều khoản) |
| PubTables-v2 | CDLA-Permissive 2.0; code/model MIT | ✅ |
| SciTSR | MIT | ✅ |
| iFLYTAB | ⚠️ Cần kiểm tra repo | Cần kiểm tra |
| WTW | ⚠️ Cần kiểm tra | Cần kiểm tra |

### Ba cảnh báo pháp lý quan trọng

**1. Repo không ghi license = all rights reserved.** Nhiều repo học thuật (đặc biệt từ các
nhóm nghiên cứu Trung Quốc) không có file LICENSE. Về mặt pháp lý, mặc định là **không được
phép sử dụng, sao chép hay phân phối**. Nếu SEMv2/SEMv3 rơi vào trường hợp này, hai lựa
chọn: (a) liên hệ tác giả xin giấy phép thương mại bằng văn bản, (b) **đọc paper và tự
implement lại** — kiến trúc trong paper là ý tưởng, không được bảo hộ bản quyền như code.

**2. Giấy phép của trọng số pretrained khác giấy phép của code.** Một repo MIT vẫn có thể
phát hành checkpoint dưới giấy phép hạn chế hơn (non-commercial, research-only). **Kiểm tra
riêng cho từng checkpoint**, đặc biệt trên Hugging Face — đọc field `license` trong model card.

**3. Điều khoản về "Results" trong CDLA.** CDLA-Permissive cho phép dùng dữ liệu thương mại,
nhưng có quy định về việc phân phối lại data và về "Results" (kết quả tính toán từ data).
Model train trên PubTabNet nói chung được coi là Results và không bị ràng buộc như data gốc,
nhưng **đây là vấn đề pháp lý cần luật sư xác nhận**, không phải kỹ thuật. Tôi không phải
luật sư và phần này chỉ mang tính thông tin.

## V.2 Phương án trả phí — để cân nhắc tận dụng

Liệt kê ở đây để bạn đánh giá, không phải khuyến nghị.

### API document AI

| Dịch vụ | Thế mạnh cho bảng không viền | Ghi chú giá (cần kiểm tra lại) |
|---|---|---|
| **Azure AI Document Intelligence** (Layout) | Rất tốt với bảng không viền và merged cell; trả `rowSpan`/`columnSpan` trực tiếp | ~$10/1000 trang cho Layout; rẻ hơn ở tier cao |
| **Google Document AI** (Form/Layout Parser) | Tốt; tích hợp tốt với GCP | ~$10–30/1000 trang tùy processor |
| **AWS Textract** (AnalyzeDocument TABLES) | Ổn; có `MERGED_CELL` block | ~$15/1000 trang cho Tables |
| **Mathpix** | Rất mạnh với bảng khoa học và công thức | Theo gói |
| **Reducto / Extend / LlamaParse** | Chuyên document parsing, chất lượng cao | Theo gói, thường đắt hơn hyperscaler |

> Giá thay đổi thường xuyên và có tier theo volume — **phải kiểm tra trang pricing chính
> thức tại thời điểm quyết định**. Các con số trên chỉ để so sánh bậc độ lớn.

### Ba cách tận dụng dịch vụ trả phí mà không phụ thuộc

**Cách 1 — Dùng làm nguồn sinh nhãn (khuyến nghị mạnh nhất).**

Giống ý tưởng distill ở III.6 nhưng dùng API thay VLM. Gọi Azure Document Intelligence trên
50k–100k bảng thật, lọc qua verification, dùng làm training data cho model của bạn.

- Chi phí một lần: 100k trang × $10/1000 ≈ **$1,000**.
- Đổi lại: một tập training data domain-specific chất lượng cao mà bạn sở hữu output.
- So sánh: thuê người gán nhãn 100k bảng ≈ $30,000–100,000.

> **Bắt buộc kiểm tra Terms of Service** trước khi làm việc này. Một số nhà cung cấp cấm
> rõ ràng việc dùng output để train model cạnh tranh. Đây là ràng buộc hợp đồng, không phải
> ràng buộc kỹ thuật, và vi phạm có hậu quả thật.

**Cách 2 — Fallback cấp cuối thay cho human review.**

Ba tầng: pipeline nội bộ → VLM nội bộ → API trả phí → human. Nếu API xử lý được 50% số ca
lẽ ra phải review thủ công, và review thủ công tốn $0.50/bảng còn API tốn $0.01/bảng, tiết
kiệm rất lớn ở đuôi phân phối.

**Cách 3 — Benchmark đối chứng.**

Chạy API trên golden set để biết **trần chất lượng khả thi** cho corpus của bạn. Nếu Azure
cũng chỉ đạt 88% trên tập bảng không viền của bạn, thì mục tiêu 95% cho model tự train là
không thực tế và bạn nên điều chỉnh kỳ vọng hoặc đầu tư vào verification thay vì model.

Chi phí: 500 bảng ≈ $5. Đây là $5 đáng chi nhất trong toàn bộ dự án.

---

# PHẦN VI — Đánh giá & metric

## VI.1 Thiết lập golden set

**Quy mô tối thiểu:** 500 bảng, lý tưởng 1,000–2,000.

**Phân tầng bắt buộc** (theo mô hình iFLYTAB):

| Nhóm | Tỉ lệ đề xuất | Ghi chú |
|---|---|---|
| Wired-Digital | 15% | Ca dễ, làm baseline |
| Wired-Camera | 15% | Kiểm tra robustness hình học |
| **Wireless-Digital** | 25% | **Trọng tâm** |
| **Wireless-Camera** | 20% | **Khó nhất** |
| **Hybrid (lai)** | 25% | **Ca thực tế phổ biến nhất, thường thiếu trong benchmark công khai** |

Trong mỗi nhóm, đảm bảo có: bảng nhiều span, bảng có cell nhiều dòng, bảng có cell rỗng
cụm, bảng dài (>50 hàng), header nhiều tầng.

## VI.2 Metric — không dùng một con số duy nhất

### Metric cấu trúc

| Metric | Đo gì | Khi nào dùng |
|---|---|---|
| **TEDS-Struct** | Tương đồng cây, bỏ qua nội dung | Metric chính cho TSR |
| **TEDS** | Cả cấu trúc và nội dung | Đo end-to-end cùng OCR |
| **GriTS** | So khớp dạng ma trận, ít nhạy hơn với lỗi nhỏ | Bổ sung, ít bị penalty oan |
| **Cell adjacency F1** | Quan hệ kề giữa các cell | Chuẩn cũ, dễ so với literature |

### Metric chẩn đoán — quan trọng hơn cho việc cải tiến

TEDS tổng che giấu việc bạn có nhiều bài toán con khác nhau. Hãy đo riêng:

```python
diagnostics = {
    "row_count_error":       abs(pred_rows - gt_rows),
    "col_count_error":       abs(pred_cols - gt_cols),
    "cell_oversplit_rate":   n_gt_cells_split_into_multiple / n_gt_cells,   # ← lỗi của bạn
    "cell_undersplit_rate":  n_pred_cells_covering_multiple_gt / n_pred_cells,
    "span_error_rate":       n_wrong_span_cells / n_gt_span_cells,
    "empty_cell_recall":     n_correct_empty / n_gt_empty,
    "header_correct":        header_structure_exact_match,
}
```

`cell_oversplit_rate` là chỉ số phải theo dõi sát nhất — nó chính là failure mode bạn mô tả.

### Metric vận hành

| Metric | Ý nghĩa |
|---|---|
| **Auto-accept rate** | % bảng vượt ngưỡng confidence |
| **Precision của auto-accept** | Trong số auto-accept, bao nhiêu % thực sự đúng — **quan trọng nhất** |
| **Review rate** | % phải review thủ công |
| **Calibration (ECE)** | Confidence score có phản ánh đúng xác suất đúng không |

> **Precision của auto-accept là KPI production quan trọng nhất.** Một hệ thống
> auto-accept 60% với precision 99.5% tốt hơn nhiều so với auto-accept 90% với precision 92%.

## VI.3 Reliability diagram — công cụ chẩn đoán chính

Vẽ biểu đồ: trục x = confidence score, trục y = tỉ lệ đúng thực tế. Đường lý tưởng là
đường chéo.

- **Nằm dưới đường chéo** → model tự tin thái quá → hạ ngưỡng auto-accept.
- **Bậc thang phẳng ở vùng cao** → confidence không phân biệt được ở vùng quan trọng nhất
  → cần thêm feature vào `consistency_score`.

Đây là công cụ hữu ích hơn nhiều so với việc nhìn TEDS trung bình.

---

# PHẦN VII — Lộ trình triển khai

## Giai đoạn 0 — Nền tảng đo lường (Tuần 1–2)

**Đây là giai đoạn không được bỏ qua.** Không có nó, mọi cải tiến sau đều là đoán mò.

- [ ] Dựng golden set 500 bảng, phân tầng theo VI.1.
- [ ] Cài đặt TEDS-Struct, GriTS, và bộ metric chẩn đoán ở VI.2.
- [ ] Chạy pipeline hiện tại → baseline **theo từng tầng**, không phải một số tổng.
- [ ] Chạy Azure Document Intelligence trên golden set (~$5) → biết trần khả thi.
- [ ] Xác định `cell_oversplit_rate` chiếm bao nhiêu % tổng thiệt hại.

**Đầu ra:** một bảng số liệu cho biết chính xác lỗi nào đang chiếm phần lớn thiệt hại.

## Giai đoạn 1 — Thắng nhanh, không cần train (Tuần 3–5)

Chạy song song, độc lập nhau:

- [ ] **I.1** Nâng lên Table Recognition V2, thêm 3 vá cho bảng lai.
- [ ] **I.4** Multi-hypothesis + reranking với `consistency_score`.
- [ ] **III.2 lớp 2** Verification bằng đối chiếu OCR box.
- [ ] Kiểm tra text layer PDF → bypass vision cho PDF số.
- [ ] Cài Docling làm baseline đối chứng (3–5 ngày, MIT, rủi ro thấp).

**Kỳ vọng:** +3–8 điểm TEDS-Struct trên nhóm wireless, không cần GPU-hour nào cho training.

## Giai đoạn 2 — Phá trần thông tin (Tuần 6–9)

- [ ] **I.3** Merge head đa mô thức với feature text. ← *ưu tiên cao nhất*
- [ ] **I.2** Chuyển HTML → OTSL, thêm constrained decoding.
- [ ] Sửa phân phối synthetic data theo checklist I.6 (đặc biệt line-spacing ≈ row-spacing).
- [ ] Hard negative mining bằng `consistency_score`.

**Kỳ vọng:** giảm mạnh `cell_oversplit_rate` — đây là giai đoạn tấn công trực diện failure
mode chính.

## Giai đoạn 3 — Đổi kiến trúc nếu cần (Tuần 10–16)

Chỉ làm nếu Giai đoạn 2 chưa đạt mục tiêu.

- [ ] Dựng baseline split-and-merge (SEMv2 code nếu license cho phép, hoặc tự implement
      SEMv3/SepFormer từ paper).
- [ ] Thử Table Transformer fine-tune trên data nội bộ (MIT, an toàn).
- [ ] So sánh 3 kiến trúc trên golden set phân tầng.
- [ ] **I.5** CP-SAT solver thay rules cố định.

## Giai đoạn 4 — Fallback & vận hành (Tuần 12–18, song song)

- [ ] Deploy PaddleOCR-VL-1.6 (INT8, vLLM) làm nhánh fallback.
- [ ] Constrained decoding OTSL — VLM **chỉ sinh cấu trúc**.
- [ ] Router 3 nhánh + reliability diagram + hiệu chỉnh ngưỡng.
- [ ] Human review loop (Label Studio) — mọi bảng review xong quay lại làm training data.
- [ ] **III.6** Distillation: dùng VLM sinh nhãn quy mô lớn cho model nhỏ.

## Bảng ưu tiên theo ROI

| Hạng mục | Công sức | Tác động kỳ vọng | ROI |
|---|---|---|---|
| Golden set phân tầng + metric chẩn đoán | 1–2 tuần | Gián tiếp nhưng bắt buộc | ⭐⭐⭐⭐⭐ |
| **Merge head đa mô thức (I.3)** | 1–1.5 tuần | Cao — đúng failure mode | ⭐⭐⭐⭐⭐ |
| Sửa phân phối synthetic (I.6) | 1 tuần | Cao | ⭐⭐⭐⭐⭐ |
| Bypass vision cho PDF số | 3–5 ngày | Cao (nếu corpus có PDF số) | ⭐⭐⭐⭐⭐ |
| Multi-hypothesis + rerank (I.4) | 1 tuần | Trung bình–cao | ⭐⭐⭐⭐ |
| OTSL + constrained decoding (I.2) | 1–2 tuần | Trung bình–cao | ⭐⭐⭐⭐ |
| Verification + router (III) | 2 tuần | Cao cho production | ⭐⭐⭐⭐ |
| Table Recognition V2 (I.1) | 3–5 ngày | Trung bình | ⭐⭐⭐⭐ |
| Distillation từ VLM (III.6) | 2–3 tuần | Rất cao nếu có volume | ⭐⭐⭐⭐ |
| Docling baseline (II.3) | 3–5 ngày | Gián tiếp (đối chứng) | ⭐⭐⭐⭐ |
| CP-SAT solver (I.5) | 1.5–2 tuần | Trung bình | ⭐⭐⭐ |
| Split-and-Merge (II.1) | 4–6 tuần | Cao | ⭐⭐⭐ |
| Kiến trúc lai tự xây (II.5) | 2–3 tháng | Cao nhất | ⭐⭐ |

---

# Phụ lục

## A. Bỏ deskew — ba cách thay thế

### A.1 Skew là biến ẩn, tối ưu cùng với cột

```python
import numpy as np

def best_theta(text_boxes, theta_range=np.arange(-3, 3.01, 0.1)):
    """Chọn góc làm cực đại độ tách cột. Chính xác hơn Hough/projection toàn trang
    vì tối ưu trực tiếp cái ta cần."""
    best, best_score = 0.0, -np.inf
    for th in theta_range:
        rot = rotate_boxes(text_boxes, th)
        hist = x_projection_histogram(rot, bins=400)
        # Độ sắc của các khe trắng: tổng gradient bình phương
        score = np.sum(np.diff(hist) ** 2) / (np.sum(hist) + 1e-6)
        if score > best_score:
            best, best_score = th, score
    return best
```

Bảng không viền có căn lề cột rất mạnh nên hàm mục tiêu này **cực kỳ nhọn**.

### A.2 Hiệu chỉnh trên tọa độ, không trên pixel

Chạy detection trên ảnh gốc, áp affine/shear lên bbox trước khi assignment. Không mất chất
lượng ảnh, không phải chạy lại model.

### A.3 Nếu là keystone, xoay không cứu được

Ảnh chụp tay thường có keystone: cột **hội tụ** chứ không song song. Cần homography từ hai
vanishing point (một từ baseline các dòng chữ, một từ trục căn lề trái các cột). Hoặc dùng
kiến trúc regress separator dạng spline/curve — khi đó không cần deskew.

## B. Neo cột bằng RANSAC — chống domino

```python
def ransac_column_axes(text_boxes, n_iter=200, tol_px=6):
    """Ranh giới cột phải suy từ đồng thuận của TẤT CẢ hàng,
    không tích lũy tuần tự từ trên xuống."""
    anchors = [b.x0 for b in text_boxes]                 # mép trái
    anchors += [decimal_point_x(b) for b in text_boxes if is_numeric(b.text)]

    axes, remaining = [], sorted(anchors)
    while len(remaining) > 3:
        best_c, best_inliers = None, []
        for _ in range(n_iter):
            c = np.random.choice(remaining)
            inliers = [a for a in remaining if abs(a - c) < tol_px]
            if len(inliers) > len(best_inliers):
                best_c, best_inliers = np.median(inliers), inliers
        if len(best_inliers) < 3:
            break
        axes.append(best_c)
        remaining = [a for a in remaining if abs(a - best_c) >= tol_px]
    return sorted(axes)
```

> **Mẹo:** trong bảng số liệu, **dấu thập phân căn thẳng hơn cả mép trái box**. Dùng vị trí
> dấu thập phân làm neo cột cho các cột số cho kết quả ổn định hơn đáng kể.

## C. Chia khối độc lập — chặn lan lỗi

```python
def split_into_blocks(table_img, text_boxes):
    """Cắt tại các separator ngang tin cậy cao, giải từng khối riêng.
    Lỗi ở khối 3 không thể làm hỏng khối 1, 2, 4."""
    cuts = []
    cuts += detect_full_width_rules(table_img)        # đường kẻ ngang đầy đủ
    cuts += detect_full_width_whitespace(text_boxes)  # dải trắng xuyên suốt
    cuts += detect_section_headers(text_boxes)        # dòng tiêu đề nhóm
    return partition(text_boxes, sorted(set(cuts)))
```

Sau khi giải từng khối, ghép lại bằng **trục cột chung** (RANSAC trên toàn bảng ở B).

## D. Render-back verification

```python
def render_back_score(pred_html, original_img, cell_regions):
    """Cycle consistency — bắt lỗi lệch hàng/cột mà heuristic khác bỏ sót."""
    rendered = render_html_to_image(pred_html, size=original_img.shape[:2])
    mask_pred = ink_mask(rendered)
    mask_orig = ink_mask(original_img)
    iou = np.logical_and(mask_pred, mask_orig).sum() / \
          (np.logical_or(mask_pred, mask_orig).sum() + 1e-6)
    chamfer = chamfer_distance(mask_pred, mask_orig)
    return 0.7 * iou + 0.3 * (1 / (1 + chamfer))
```

Đây là tín hiệu confidence **độc lập hoàn toàn** với model sinh ra kết quả — giá trị của nó
nằm ở tính độc lập đó.

## E. Checklist trước khi lên production

- [ ] Golden set phân tầng, đo TEDS-Struct **riêng cho từng nhóm**
- [ ] Metric chẩn đoán (`cell_oversplit_rate`, `span_error_rate`, ...) được log
- [ ] Reliability diagram được vẽ, ngưỡng auto-accept hiệu chỉnh theo dữ liệu
- [ ] Precision của auto-accept đo được và ≥ mục tiêu nghiệp vụ
- [ ] Verification độc lập với model chính
- [ ] Router 3 nhánh có timeout và circuit breaker cho nhánh VLM
- [ ] Mọi ảnh + output được lưu để tích lũy training data
- [ ] Human review loop khép kín (review → training data → retrain)
- [ ] Rollback plan: giữ pipeline cũ chạy shadow mode 1–2 tháng
- [ ] Giấy phép của **từng** model, **từng** checkpoint, **từng** dataset đã được rà soát
- [ ] Nếu dùng API trả phí sinh nhãn: ToS đã được đọc và xác nhận cho phép

## F. Tài liệu tham khảo

**Split-and-Merge**
- SEM: *Split, Embed and Merge: An accurate table structure recognizer* — arXiv 2107.05214
- SEMv2: *Table separation line detection based on instance segmentation* — arXiv 2303.04384; code: `github.com/ZZR8066/SEMv2`; giới thiệu dataset **iFLYTAB**
- SEMv3: *A Fast and Robust Approach to Table Separation Line Detection* — arXiv 2405.11862

**Separator regression / DETR**
- TSRFormer: *Table Structure Recognition with Transformers* — arXiv 2208.04921
- SepFormer: *Coarse-to-fine Separator Regression Network for TSR* — arXiv 2506.21920

**Object-detection based**
- PubTables-1M / Table Transformer — CVPR 2022; code MIT: `github.com/microsoft/table-transformer`
- PubTables-v2 — arXiv 2512.10888 (CDLA-Permissive 2.0, code/model MIT)
- *Aligning benchmark datasets for table structure recognition* — arXiv 2303.00716

**Tokenization**
- OTSL: *Optimized Table Tokenization for Table Structure Recognition* — arXiv 2305.03393 (ICDAR 2023)
- Docling Technical Report — arXiv 2408.09869 (MIT)

**Đa mô thức / VLM**
- UniTable — arXiv 2403.04822
- UniTabNet — arXiv 2409.13148
- TDATR — arXiv 2603.22819 (phân tích lỗi trên iFLYTAB-full)
- PaddleOCR-VL — arXiv 2510.14528; PaddleOCR-VL-1.5 — arXiv 2601.21957; PaddleOCR-VL-1.6 — arXiv 2606.03264
- MinerU2.5 — arXiv 2509.22186 (dùng OTSL)

**Benchmark**
- OmniDocBench — `github.com/opendatalab/OmniDocBench`
- Dr. DocBench — arXiv 2606.01393 (phân tích riêng bảng không viền)

**PaddlePaddle**
- PP-StructureV3: `paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/`
- Table Recognition V2 pipeline docs
- Dataset docs: `paddlepaddle.github.io/PaddleOCR/main/en/datasets/table_datasets.html`

---

## Ghi chú cuối

Các con số về tài nguyên, latency và chi phí trong tài liệu này là **ước tính** dựa trên
kích thước kiến trúc, benchmark công bố và giá cloud tại thời điểm viết. Chúng dùng để so
sánh **bậc độ lớn** giữa các phương án, không phải để lập ngân sách chính xác. Phải đo lại
trên ảnh thực tế và hạ tầng của bạn.

Phần giấy phép mang tính thông tin kỹ thuật, không phải tư vấn pháp lý. Trước khi đưa bất
kỳ model hay dataset nào vào sản phẩm thương mại, hãy để bộ phận pháp lý rà soát.

Luận điểm trung tâm của tài liệu: **failure mode của bạn có nguyên nhân là thiếu thông tin
và sai primitive, không phải thiếu dữ liệu train.** Ba can thiệp đánh trúng nguyên nhân đó
là (1) đưa tín hiệu text vào quyết định merge, (2) chuyển từ cell detection sang separator,
và (3) xây tầng verification độc lập để biến "sai không phát hiện được" thành "sai được
định tuyến". Hai can thiệp đầu nâng trần chất lượng; can thiệp thứ ba mới là thứ đưa hệ
thống lên production.
