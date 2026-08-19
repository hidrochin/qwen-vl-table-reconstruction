# Lộ trình cải thiện KIE: LayoutXLM + SPADE/BROS decoder

**v2** — cập nhật sau khi đọc mã nguồn eval (`bros_spade_module.py`) và nhận câu trả lời Q1–Q4.

> Pipeline: **LayoutXLM backbone → 3 head (ITC / STC / EL)**, loss = tổng 3 CE không trọng số, sliding window **không overlap** cho document > 512 token (15% dataset), ~10% dữ liệu chữ viết tay.

---

## 0. Đính chính so với v1

Hai điều trong bản v1 sai hoặc không áp dụng được. Ghi lại để tránh đi nhầm hướng.

### ❌ Đính chính 1: phép nhân `0.96 × 0.95 × 0.95 ≈ 86.6%` là đếm trùng

Đọc `eval_ee_spade_example`:

```python
gt_first_words  = parse_initial_words(gt_itc_label, ...)      # dùng ITC
gt_class_words  = parse_subsequent_words(gt_stc_label, ..., gt_first_words, ...)  # rồi đi tiếp qua STC
...
gt_parse = set(gt_class_words[class_idx])   # set các TUPLE token index
pr_parse = set(pr_class_words[class_idx])
n_correct_classes += len(gt_parse & pr_parse)
```

Metric này **đã là entity-level exact match, gộp sẵn ITC + STC**. Một entity chỉ tính đúng khi:
- first token đúng **và** class đúng (ITC), **và**
- toàn bộ chuỗi token phía sau đúng cả về **thành phần lẫn thứ tự** (STC)

Nên không được nhân 96% với 95% — chúng không phải hai giai đoạn độc lập. Ước lượng đúng hơn:

```
end-to-end pair F1  ≈  EE_F1 × P(EL đúng | entity đúng)  ≈  0.95 × 0.95  ≈  0.90
```

(EL 95% được đo với entity oracle, nên vẫn cần nhân.) Kết luận vẫn giữ: **~90%, không phải 95%** — chỉ là khoảng cách nhỏ hơn tôi nói lần trước.

### ❌ Đính chính 2: "single-head constraint cho EL" — đã có sẵn, và Hungarian ở đây **có hại**

Bạn đã dùng softmax + CE theo chiều value → key. Argmax trên chiều đó **tự động** enforce "mỗi value ≤ 1 key". Không có gap nào để lấp.

Hơn nữa vì nhãn cho phép **1 key → nhiều value**, áp Hungarian vào EL sẽ ép quan hệ thành injective và **phá vỡ** đúng những trường hợp một key có nhiều value. **Không làm.** Mục §3.2 của v1 đã bị xoá.

> ✅ Nhưng với **STC** thì kết luận ngược lại và mạnh hơn v1 — xem §1.1.

---

## 1. Ba phát hiện mới từ mã nguồn

### 1.1 🔴 `parse_subsequent_words` âm thầm vứt bỏ token khi có xung đột

Đây là cơ chế vật lý của zigzag, tìm thấy trực tiếp trong code:

```python
next_token_idx_dict = {}
for token_idx in valid_token_indices[0]:
    next_token_idx_dict[stc_label_np[token_idx]] = token_idx   # ⚠️ ghi đè
```

Phân tích:

- STC softmax **theo chiều predecessor**: với mỗi token `j`, model chọn token `i` nào là tiền nhiệm của `j`. Ở dạng nhãn: `stc_label[j] = i`.
- Dòng trên **đảo ngược** map đó thành `i → j` để đi xuôi chuỗi.
- Argmax đảm bảo **mỗi token có đúng 1 predecessor** ✓
- Nhưng **không gì đảm bảo mỗi token có ≤ 1 successor** ✗

Khi hai token `j₁` và `j₂` cùng chọn predecessor `i`, dict giữ lại **j cuối cùng theo thứ tự index** và **im lặng vứt bỏ j₁**. Không warning, không log, không đếm.

Với chữ in, hình học đều nên xung đột hiếm. Với chữ tay, hình học nhiễu → xung đột tăng vọt → chuỗi bị cắt cụt hoặc rẽ sai. **Đây chính là "nhận diện thiếu/sai các cụm" mà bạn mô tả.**

→ Giải pháp ở §3.1. Cách đo lượng thiệt hại ở §2.1 (chỉ 5 dòng code).

### 1.2 🟠 Điều kiện dừng chuỗi chỉ chặn init token **cùng class**

```python
for init_token_indices in init_words:          # init_words[class_idx] — CHỈ 1 class
    for init_token_idx in init_token_indices:
        cur_token_indices = [init_token_idx]
        for _ in range(max_connections):        # = 50
            if cur_token_indices[-1] in next_token_idx_dict:
                if next_token_idx_dict[cur_token_indices[-1]] not in init_token_indices:
                    cur_token_indices.append(...)
                else:
                    break
```

`init_token_indices` là danh sách first-token **của riêng class đang xét**. Nên chuỗi của một entity class `QUESTION` **có thể chạy xuyên qua** first-token của một entity class `ANSWER` mà không bị chặn. Chuỗi tiếp tục nuốt token của entity khác cho tới khi hết map hoặc chạm `max_connections = 50`.

Sửa: đổi điều kiện dừng thành "không thuộc first-token của **bất kỳ** class nào", cộng phát hiện chu trình.

```python
all_init = set(i for cls in init_words_all_classes for i in cls)
...
nxt = next_token_idx_dict[cur_token_indices[-1]]
if nxt in all_init or nxt in cur_token_indices:   # chặn cross-class + chu trình
    break
```

Chi phí: ~5 dòng, inference-only. Đáng làm ngay cùng §3.1.

### 1.3 🟠 `dummy_idx` có thể lệch với độ dài chuỗi sau khi nối window

```python
self.eval_kwargs = {"dummy_idx": self.cfg.train.max_seq_length}
...
valid_token_indices = np.where((valid_stc_label != dummy_idx) * (valid_stc_label != 0))
```

`dummy_idx` là **cột "không có predecessor"** trong ma trận STC `(N, N+1)`. Nếu bạn nối hidden state của 2 window thành `N = 1024` nhưng `cfg.train.max_seq_length` vẫn là `512`:

- Token có predecessor thật ở **index 512** bị filter nhầm thành "không có predecessor" → chuỗi đứt ngay tại biên window
- Cột dummy thật (index 1024) lọt qua filter → rác chèn vào `next_token_idx_dict`

**Cần verify:** in ra `stc_outputs.shape[-1]`, `cfg.train.max_seq_length`, và độ dài chuỗi sau khi nối. Ba số này phải thỏa `stc_outputs.shape[-1] == N + 1` và `dummy_idx == N`. Nếu lệch, đây là bug thật, không phải vấn đề mô hình — và nó đánh đúng vào 15% document dài.

---

## 2. Đo đạc trước — Sprint 0 (1–2 ngày, không train)

### 2.1 🔴 Đếm xung đột STC — chỉ 5 dòng, giá trị cao nhất trong toàn tài liệu

Con số này cho biết chính xác Hungarian decoding sẽ mua lại được bao nhiêu:

```python
from collections import Counter

def count_stc_collisions(pr_stc_label, attention_mask, dummy_idx):
    """Đếm số token bị parse_subsequent_words vứt âm thầm."""
    valid = (pr_stc_label != dummy_idx) & (pr_stc_label != 0) & attention_mask.bool()
    preds = pr_stc_label[valid].tolist()          # danh sách predecessor được chọn
    c = Counter(preds)
    n_dropped   = sum(v - 1 for v in c.values() if v > 1)
    n_conflicts = sum(1 for v in c.values() if v > 1)
    return n_dropped, n_conflicts, len(preds)
```

Chạy trên toàn tập validation, tách riêng printed / handwritten. Nếu `n_dropped / len(preds)` ở nhóm chữ tay cao hơn nhóm chữ in vài lần → giả thuyết được xác nhận và §3.1 là ưu tiên tuyệt đối.

### 2.2 🔴 F1 bất biến thứ tự — tách "sai thứ tự" khỏi "sai gom nhóm"

Metric hiện tại so sánh **tuple** (có thứ tự). Thêm một biến thể so sánh **frozenset** (không thứ tự):

```python
# trong eval_ee_spade_example, thêm song song:
gt_orderless = set(frozenset(t) for t in gt_class_words[class_idx])
pr_orderless = set(frozenset(t) for t in pr_class_words[class_idx])
n_correct_orderless += len(gt_orderless & pr_orderless)
```

Đọc kết quả:

| Quan sát | Chẩn đoán | Hành động |
|---|---|---|
| `F1_orderless ≫ F1_exact` (chênh > 3 điểm) | Model gom **đúng** token nhưng **sai thứ tự** — thuần vấn đề decoding | §3.1 + §3.3, không cần train lại |
| `F1_orderless ≈ F1_exact`, cả hai đều thấp | Sai **biên entity**: nuốt thừa hoặc thiếu token | §1.2, §4, §5 |

Đây là diagnostic sắc nhất bạn có thể chạy hôm nay. Nó quyết định Sprint 1 hay Sprint 2 mới là ưu tiên.

### 2.3 Tách precision và recall

Code đã tính sẵn cả hai nhưng bạn chỉ báo F1. Bảng đọc:

| Quan sát | Ý nghĩa |
|---|---|
| `P ≫ R` | Model **bỏ sót** entity — chuỗi đứt sớm (đúng triệu chứng §1.1) |
| `R ≫ P` | Model **dự đoán thừa** — chuỗi nuốt quá dài (đúng triệu chứng §1.2) |

### 2.4 Lỗi theo độ dài entity

Vì metric là exact-match trên cả chuỗi, **một zigzag duy nhất giết cả entity**. Entity 8 token có xác suất sai cao hơn nhiều so với entity 2 token, kể cả khi accuracy per-token như nhau.

```python
error_rate  vs.  len(entity)      # tách printed / handwritten
```

Nếu value viết tay thường dài hơn (địa chỉ, họ tên đầy đủ), đây giải thích phần lớn gap giữa hai domain — **và nó không phải vấn đề "model không hiểu chữ tay"**, mà là vấn đề khuếch đại của exact-match trên chuỗi dài.

### 2.5 Lỗi quanh biên window

```python
W = 512
near_boundary = (token_idx % W < 32) | (token_idx % W > W - 32)
error_rate[near_boundary]  vs.  error_rate[~near_boundary]
```

Xem §4 để biết vì sao con số này quan trọng hơn bạn nghĩ.

### 2.6 Các việc còn lại

| # | Việc | Ghi chú |
|---|---|---|
| 2.6a | Đo **pair-level F1 end-to-end** (EE + EL nối tiếp, dùng entity dự đoán chứ không phải oracle) | Metric north-star thật |
| 2.6b | Tách metric printed vs handwritten | Q5 chưa trả lời |
| 2.6c | Audit tay 50 sample sai → % label noise | Q6 chưa trả lời |
| 2.6d | Train 3 seed, tính std | Q7 chưa trả lời — quyết định ngưỡng ý nghĩa |
| 2.6e | Verify `dummy_idx` (§1.3) | Có thể là bug thẳng |

> **Q5, Q6, Q7 vẫn cần bạn thống kê.** Đặc biệt Q7: nếu std giữa seed là ±0.8% thì mọi "cải thiện" dưới 1.5% đều là nhiễu, và bạn sẽ đuổi theo ma. Đây là việc rẻ nhất trong bảng.

---

## 3. Sprint 1 — Sửa decoding (inference-only, không train lại)

### 3.1 🔴 Global assignment cho STC — ưu tiên số 1

Trực tiếp thay thế cơ chế ghi đè ở §1.1. Bài toán "gán successor" là bài toán ghép cặp hai phía; giải bằng Hungarian với null-slot riêng cho mỗi token (để token cuối entity được phép không có successor):

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6

def decode_stc_global(stc_logits, boxes, line_id, dummy_idx, attention_mask):
    """
    stc_logits: (N, N+1) — cột dummy_idx là 'no predecessor'
    Trả về: dict {predecessor_idx: successor_idx}  (thay cho next_token_idx_dict)
    LƯU Ý: bài toán gốc là j chọn predecessor i. Ta chuyển vị để gán successor.
    """
    N = stc_logits.shape[0]
    logp = log_softmax(stc_logits, axis=-1)          # (N, N+1)

    # score[i, j] = log P(predecessor của j là i)  → chuyển vị
    score = logp[:, :N].T                             # score[i, j]

    cost = np.full((N, 2 * N), BIG, dtype=np.float64)
    cost[:, :N] = -score
    # null slot riêng cho từng i: chi phí = -log P(không ai chọn i làm predecessor)
    cost[np.arange(N), N + np.arange(N)] = -np.log1p(-np.exp(score).sum(axis=1).clip(max=1 - 1e-9))

    np.fill_diagonal(cost[:, :N], BIG)                # cấm self-loop
    invalid = ~attention_mask.astype(bool)
    cost[invalid, :] = BIG
    cost[:, :N][:, invalid] = BIG

    feasible = geometric_feasible(boxes, line_id)     # §3.2
    cost[:, :N][~feasible] = BIG

    rows, cols = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(rows, cols) if j < N and cost[i, j] < BIG}
```

Sau đó **phá chu trình**: duyệt các chuỗi, nếu gặp cycle thì cắt tại cạnh có score thấp nhất.

Ghép với §1.2 (chặn cross-class + cycle) là bạn có một decoder thoả **toàn bộ** ràng buộc cấu trúc: ≤1 predecessor, ≤1 successor, không chu trình, không xuyên entity.

**Tiền lệ:** SERA ([arXiv:2110.09915](https://arxiv.org/abs/2110.09915)) báo cáo rằng decode có hay không có ràng buộc cấu trúc tạo ra "a large performance gap" — và trong trường hợp của bạn, ràng buộc đó hiện đang được xử lý bằng cách... ghi đè dict.

### 3.2 Mask hình học khi decode

```python
def geometric_feasible(boxes, line_id, max_line_skip=1, max_gap_ratio=3.0):
    N = len(boxes)
    M = np.zeros((N, N), dtype=bool)
    h = np.median([b[3] - b[1] for b in boxes])
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dl = line_id[j] - line_id[i]
            if dl == 0:                                   # cùng dòng
                gap = boxes[j][0] - boxes[i][2]
                M[i, j] = 0 <= gap <= max_gap_ratio * h   # bên phải, đủ gần
            elif 0 < dl <= max_line_skip:                 # xuống dòng kế
                M[i, j] = True
    return M
```

⚠️ `line_id` **phải** đến từ line clustering thích ứng theo median height (§5.1), không phải threshold tuyệt đối. Dùng threshold cứng ở đây là tái tạo lại đúng vấn đề chữ tay đang muốn sửa.

Tune `max_gap_ratio` và `max_line_skip` trên validation — bắt đầu lỏng (5.0, 2) rồi siết dần, theo dõi recall.

### 3.3 Beam search thay greedy

Entity thường 2–8 token nên beam width 3–5 gần như miễn phí. Cho phép sửa lỗi cục bộ mà argmax/Hungarian một lượt bỏ sót, đặc biệt khi cộng prior hình học (phạt khoảng cách lớn, phạt nhảy ngược dòng).

### 3.4 Bảng tổng kết Sprint 1

| # | Việc | Chi phí | Mục tiêu | Train lại? |
|---|---|---|---|---|
| 3.1 | Hungarian decoding cho STC | ~40 dòng | 🔴 Zigzag, cụm thiếu | Không |
| 1.2 | Chặn cross-class + phát hiện chu trình | ~5 dòng | 🟠 Chuỗi nuốt quá dài | Không |
| 1.3 | Verify & sửa `dummy_idx` | ~1 dòng | 🟠 Document dài | Không |
| 3.2 | Mask hình học | ~30 dòng | 🔴 Zigzag | Không |
| 3.3 | Beam search | ~50 dòng | 🟡 Tinh chỉnh | Không |

> Toàn bộ Sprint 1 **không cần train lại một epoch nào**. Đây là phần có tỉ lệ lợi ích/công sức cao nhất của toàn tài liệu. Làm hết trước khi động vào bất kỳ thứ gì khác.

---

## 4. Sliding window — đánh giá lại lập luận của bạn

Bạn nói: *"key nhỏ như name, date of birth thường có tương quan với nhau nên hầu như không bao giờ vượt quá 512 token; Name ở đầu sẽ không bao giờ nối đến Patrick ở cuối trang."*

Lập luận này **đúng về mặt không gian** nhưng có ba lỗ hổng:

**(a) Window cắt theo thứ tự token, không theo vị trí không gian.** Thứ tự token đến từ OCR/reading order — thứ mà bạn *đã biết là không đáng tin với chữ tay*. Một key và value nằm sát nhau trên trang có thể cách xa nhau về token index nếu reading order sai. Chính miền dữ liệu khó nhất của bạn cũng là miền mà giả định "gần trên trang ⇒ gần trong token index" gãy mạnh nhất.

**(b) Zero overlap = thiệt hại tại biên là tối đa.** Token 511 và 512 liền kề trong reading order nhưng **không bao giờ được co-attend**. Đây chính xác là nơi cặp key–value dễ bị chẻ đôi nhất, và bạn đang để nó trần. Chi phí sửa cực thấp: overlap 128 token.

**(c) Chuỗi STC cũng bị chẻ**, không chỉ EL. Một entity trải từ token 508 đến 514 bị cắt làm đôi giữa hai window, và mỗi nửa được tính từ biểu diễn không thấy nửa kia.

Với 15% document > 512 và mỗi document có ≥1 biên, tác động không lớn nhưng **không bằng không** — và §2.5 cho bạn con số chính xác.

### 4.1 Sửa (chi phí thấp)

```python
def layout_aware_chunks(order, boxes, max_len=512, min_len=384, overlap=128):
    """Cắt tại khoảng trống dọc lớn nhất, có overlap."""
    chunks, start = [], 0
    while start < len(order):
        hard_end = min(start + max_len, len(order))
        if hard_end == len(order):
            chunks.append(order[start:hard_end]); break
        cand = range(start + min_len, hard_end)
        cut  = max(cand, key=lambda k: vertical_gap(boxes, order[k - 1], order[k]))
        chunks.append(order[start:cut])
        start = cut - overlap          # ⬅️ lùi lại để tạo overlap
    return chunks
```

Khi có overlap, token xuất hiện ở nhiều window → merge hidden state bằng cách **lấy từ window mà token nằm xa biên nhất**, thay vì average:

```python
def pick_hidden(token_idx, window_hiddens, windows):
    best, best_margin = None, -1
    for h, (ws, we) in zip(window_hiddens, windows):
        if ws <= token_idx < we:
            margin = min(token_idx - ws, we - token_idx)
            if margin > best_margin:
                best, best_margin = h[token_idx - ws], margin
    return best
```

### 4.2 🟢 Tận dụng chính quan sát của bạn: **banded head**

Nếu key và value thực sự luôn ở gần nhau, hãy **mã hoá điều đó thành ràng buộc** thay vì chỉ hy vọng model tự học:

```python
# chỉ tính score cho các cặp thoả điều kiện lân cận
band = (np.abs(i[:, None] - j[None, :]) < W_tok) | spatially_near(boxes, thresh)
stc_logits[~band] = -1e4
el_logits[~band]  = -1e4
```

Lợi ích kép:
- **Cân bằng lớp:** tỉ lệ dương tăng từ ~1/N lên ~1/W, giảm mạnh sức ép lên CE (§6.1)
- **Tốc độ:** N² → N·W
- **Regularization:** loại bỏ toàn bộ một họ lỗi tầm xa vô lý

Đây là cách rẻ nhất để biến prior domain của bạn thành lợi thế thay vì thành rủi ro ngầm. Áp dụng cho **cả** train và inference.

---

## 5. Sprint 2 — Chữ viết tay (cần train lại)

### 5.1 Chuẩn hoá bbox theo dòng — đề xuất mạnh nhất cho vấn đề 2

Chữ in cho box cao bằng nhau; chữ tay cho box cao thấp so le. Nên `y0/y1` đưa vào LayoutXLM đang mã hoá **độ cao nét chữ** chứ không mã hoá **dòng**. Model học "cùng dòng ⇔ y giống nhau" trên chữ in rồi gãy trên chữ tay.

```python
def line_normalize_boxes(boxes, tol=0.6):
    h_med = np.median([b[3] - b[1] for b in boxes])
    yc    = np.array([(b[1] + b[3]) / 2 for b in boxes])
    line_id = cluster_1d(yc, eps=tol * h_med)      # thích ứng, KHÔNG threshold cứng

    out = np.array(boxes, dtype=float).copy()
    for lid in np.unique(line_id):
        m = line_id == lid
        out[m, 1] = np.median(out[m, 1])
        out[m, 3] = np.median(out[m, 3])
    return out, line_id
```

`line_id` trả về ở đây dùng lại được cho §3.2 — làm một lần, dùng hai chỗ.

Tiền lệ: PEneo dùng đúng nguyên tắc này khi tách dòng — so sánh khoảng cách dọc giữa hai từ liền kề với **chiều cao trung bình của từ trong entity**, thay vì ngưỡng cứng.

Biến thể đáng thử: giữ cả hai tín hiệu — box đã chuẩn hoá vào layout embedding, còn `h_gốc / h_median` đưa vào head như feature phụ.

### 5.2 Augmentation bbox — áp lên **cả** dữ liệu chữ in

Đây là cách hiệu quả nhất để 90% dữ liệu chữ in phục vụ cho 10% chữ tay.

```python
def augment_boxes_handwriting(boxes, line_id, p=0.6):
    if random.random() > p:
        return boxes
    b = np.array(boxes, dtype=float).copy()
    h_med = np.median(b[:, 3] - b[:, 1])

    for lid in np.unique(line_id):
        m = line_id == lid
        b[m, 1] += random.gauss(0, 0.20 * h_med)       # baseline drift
        b[m, 3] += random.gauss(0, 0.20 * h_med)
        theta = math.radians(random.uniform(-2.0, 2.0)) # slant
        xc = b[m, 0].mean()
        b[np.ix_(m, [1, 3])] += ((b[m, 0:1] - xc) * math.tan(theta))

    # per-box height jitter — ĐÂY là thứ chữ in không có
    scale = np.random.uniform(0.70, 1.45, size=len(b))
    yc    = (b[:, 1] + b[:, 3]) / 2
    half  = (b[:, 3] - b[:, 1]) / 2 * scale
    b[:, 1], b[:, 3] = yc - half, yc + half

    b[:, [0, 2]] += np.random.normal(0, 0.05 * h_med, (len(b), 2))
    return np.clip(b, 0, 1000)
```

Tiền lệ: LOCR thêm Gaussian noise vào toạ độ bbox để mô phỏng localization thiếu chính xác; TILT scale khoảng cách ngang/dọc giữa token theo hệ số ngẫu nhiên ([survey 2026](https://arxiv.org/abs/2601.12318)).

### 5.3 Cân bằng domain

- Oversample document chữ tay lên **30–40% mỗi batch** (`WeightedRandomSampler`)
- Fine-tune hai giai đoạn: train toàn bộ → fine-tune LR thấp trên tập chữ tay
- **Đo riêng metric hai domain** — nếu printed 97% / handwritten 78% thì con số 95% đang che giấu vấn đề

### 5.4 Shuffle thứ tự token

SPADE vốn được thiết kế để **không phụ thuộc thứ tự**, nhưng model của bạn có thể đang lén học shortcut dựa trên thứ tự tuyến tính. Shuffle chặn shortcut đó.

LayTextLLM gọi kỹ thuật này là Shuffled-OCR SFT, xáo trộn **20%** số mẫu ([arXiv:2407.01976](https://arxiv.org/abs/2407.01976)).

```python
# 20% sample: hoán vị token (giữ nguyên bbox và nhãn tương ứng)
# 10% sample: chỉ swap dòng liền kề
```

⚠️ Nhớ hoán vị **cả `stc_labels` và `el_labels`** theo đúng permutation, nếu không bạn đang train trên nhãn rác.

### 5.5 Augmentation mức ảnh

Dùng **Augraphy** ([arXiv:2208.14558](https://arxiv.org/abs/2208.14558) · [GitHub](https://github.com/sparkfish/augraphy)).

Một paper giải đúng bài toán domain gap digital→handwritten trên Form-NLU đã sàng lọc còn **sáu** augmentation: `InkBleed`, `Letterpress`, `LowInkRandomLines`, `LowInkPeriodicLines`, `JPEG`, `DirtyScreen` ([arXiv:2502.06132](https://arxiv.org/abs/2502.06132)). Danh sách khởi đầu tốt, khỏi grid search 24 cái.

### 5.6 Sinh dữ liệu chữ tay tổng hợp

Nếu Q5 = "form in sẵn, value viết tay" thì đây là hướng ROI cao nhất: render value bằng font handwriting (hoặc model HTG) vào template in sẵn, giữ nguyên nhãn. Tham khảo: [Advancing Offline HTR (2025)](https://arxiv.org/abs/2507.06275).

---

## 6. Loss & training

### 6.1 🟠 `loss = loss_itc + loss_stc + loss_el` — tổng không trọng số

Ba loss có **cardinality output rất khác nhau**:

| Head | Số lớp | Tỉ lệ dương |
|---|---|---|
| ITC | C+1 (nhỏ, ~5–20) | cao |
| STC | N+1 (~513) | ~1/N |
| EL | N+1 (~513) | ~1/N |

CE trên 513 lớp có scale khác hẳn CE trên 10 lớp. Tổng thô nghĩa là bạn đang **ngầm gán trọng số theo cardinality**, không theo tầm quan trọng.

```python
loss = l_itc + λ_stc * l_stc + λ_el * l_el
```

Ablation `λ ∈ {0.5, 1, 2, 5}`. Rẻ, và thường ăn 0.5–1.5 điểm. Kết hợp với §4.2 (banded head) thì mất cân bằng giảm mạnh và λ ổn định hơn nhiều.

### 6.2 Mất cân bằng lớp

- **Focal loss** (γ=2) cho STC/EL ([arXiv:1708.02002](https://arxiv.org/abs/1708.02002))
- **Hard negative mining**: chỉ lấy top-k negative có loss cao nhất (k ≈ 20× số positive)
- **Banded candidate set** (§4.2) — giải pháp gọn nhất, làm trước hai cái trên

### 6.3 Consistency regularization

Ba head đang train độc lập hoàn toàn. Thêm ràng buộc mềm phản ánh đúng cấu trúc mà decoder cần:

```python
# token được ITC coi là first-token thì không nên là successor của token khác
L_consist = (p_itc_is_first * p_stc_has_predecessor).sum() / N

# ⬅️ khớp trực tiếp với §1.1: phạt khi nhiều token cùng chọn 1 predecessor
succ_count = p_stc.sum(dim=0)                     # kỳ vọng số successor của mỗi token
L_uniq     = F.relu(succ_count - 1.0).mean()

loss = l_itc + λ_stc*l_stc + λ_el*l_el + λ_c*L_consist + λ_u*L_uniq
```

`L_uniq` đặc biệt đáng thử: nó dạy model **trong lúc train** đúng ràng buộc mà Hungarian enforce **lúc inference**. Hai thứ cộng hưởng — Hungarian sẽ ít phải "sửa" hơn.

Tiền lệ: RE2 ([arXiv:2305.14590](https://arxiv.org/abs/2305.14590)) dùng "a constraint objective to regularize the model towards consistency with the inherent constraints of the relation extraction task".

### 6.4 Chi tiết dễ ăn điểm

- **Layer-wise LR**: backbone 2e-6 ~ 5e-6, head 1e-4 (PEneo dùng đúng cấu hình này)
- **Train lâu hơn bạn nghĩ**: PEneo fine-tune **650 epoch** trên RFUND. Nếu bạn đang train 50–100 epoch, có thể chưa hội tụ.
- EMA weights, label smoothing 0.05–0.1 cho ITC
- Ensemble 3–5 seed, average **score matrix** trước khi decode (không phải average prediction)

---

## 7. Sprint 3 — Kiến trúc

### 7.1 Geometric bias trong pair score

Head bilinear của BROS chỉ thấy hình học **gián tiếp** qua hidden state. Cộng bias tính trực tiếp:

```python
class GeoBiasedPairScore(nn.Module):
    def __init__(self, d, d_geo=64):
        super().__init__()
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.geo = nn.Sequential(nn.Linear(11, d_geo), nn.GELU(), nn.Linear(d_geo, 1))
        self.scale = d ** -0.5

    def forward(self, h, boxes):
        content = (self.q(h) @ self.k(h).transpose(-1, -2)) * self.scale
        g = geo_features(boxes)          # dx, dy, dx/w_i, dy/h_i, h_j/h_i, w_j/w_i,
        return content + self.geo(g).squeeze(-1)   # IoU_x, IoU_y, dist, cos, sin
```

Đây là **spatial compatibility attention bias** của KVPFormer — thứ giúp họ đạt kết quả mạnh trên RE **mà không cần pre-training**, nên port được sang LayoutXLM. GeoLayoutLM cũng cho thấy hình học tường minh mới là yếu tố quyết định ở RE: LayoutLMv3 "depends on the semantic information excessively and ignores the layout more or less".

### 7.2 EL đang vứt bỏ nội dung của value

Với `Date of birth : Feb 19th 2025`, head EL chỉ thấy `Date` và `Feb`. Toàn bộ `of birth` và `19th 2025` bị bỏ.

```python
# train: pool theo span ground-truth; infer: pool theo span do ITC+STC decode
e_i = torch.cat([h[first_i], h[span_i].mean(0), h[last_i]], dim=-1)
```

Rủi ro: train/inference mismatch + error propagation từ STC. Giảm thiểu bằng **scheduled sampling** — tăng dần tỉ lệ dùng span dự đoán (0% → 50% qua các epoch).

> ⚠️ **Lưu ý riêng cho bạn:** nếu hiện tại loss EL đang tính trên first-token **ground-truth** nhưng inference lại dùng first-token **dự đoán**, thì đã có sẵn một mismatch — model chưa bao giờ học cách xử lý entity sai. Đây là ứng viên rất mạnh cho hiện tượng "có trong validation mà inference vẫn sai". §2.6a (đo end-to-end với entity dự đoán) sẽ lộ ra ngay.

### 7.3 GOSE — head thay thế phù hợp nhất

[arXiv:2305.13850](https://arxiv.org/abs/2305.13850) · [GitHub](https://github.com/chenxn2020/GOSE) · EMNLP 2023 Findings

Repo chính thức dùng trực tiếp **LayoutXLM và LiLT** làm backbone. Cơ chế: dự đoán quan hệ sơ bộ → khai thác *global structure knowledge* từ vòng trước → nhúng ngược vào biểu diễn entity → lặp K lần. Có thêm **spatial prefix** guide attention.

Bài báo nêu chính xác vấn đề của head độc lập: thiếu cấu trúc toàn cục khiến model *"struggle to learn long-range relations and easily predict conflicted results"*. "Conflicted results" chính là xung đột ở §1.1 — nhưng giải ở tầng **học** thay vì tầng **decode**.

### 7.4 Các lựa chọn khác

| Method | Link | Ghi chú |
|---|---|---|
| **KVPFormer** | [arXiv:2304.07957](https://arxiv.org/abs/2304.07957) | Yêu cầu biết trước entity span và **không xử lý được OCR không thứ tự** → chỉ nên lấy spatial compatibility bias (§7.1) |
| **GeoLayoutLM** | [arXiv:2304.10759](https://arxiv.org/abs/2304.10759) · [code](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/GeoLayoutLM) | RFE head chỉ 3.5% tham số. Nhưng ablation: CRP không pre-train = 82.2% F1, có pre-train mới +2.7% — một phần lợi ích nằm ở pre-training bạn không có |
| **PEneo** | [arXiv:2401.03472](https://arxiv.org/abs/2401.03472) · [code](https://github.com/ZeningLin/PEneo) | Line grouping bằng **hai ma trận** head/tail thay vì chuỗi tuần tự → về cấu trúc **không thể sinh zigzag**. Cần re-annotate mức dòng. ⚠️ License: non-commercial research only |
| **RE2** | [arXiv:2305.14590](https://arxiv.org/abs/2305.14590) | Edge-aware GAT + constraint objective |
| **TPP** | [arXiv:2310.11016](https://arxiv.org/abs/2310.11016) | Token path prediction trên đồ thị đầy đủ. Tốn memory |
| **UNER** | [arXiv:2408.01038](https://arxiv.org/abs/2408.01038) | Xử lý entity gián đoạn/lồng nhau, đúng reading order |

### 7.5 Bơm reading order vào backbone

- **ROAP** ([arXiv:2601.05470](https://arxiv.org/abs/2601.05470), preprint 2026) — AXG-Tree sinh reading sequence bền với skew và khoảng cách bất thường, inject qua RO-RPB. Tự mô tả "lightweight and architecture-agnostic... without altering their pre-trained backbones". Preprint rất mới, tự verify trước khi đầu tư.
- **RORE** ([arXiv:2409.19672](https://arxiv.org/abs/2409.19672)) — ma trận nhị phân n×n + relation-aware attention, dùng được **pseudo-label** từ model ROP có sẵn.

---

## 8. Lộ trình đã cập nhật

### Sprint 0 — Đo đạc (1–2 ngày, **không train**)

| # | Việc | Ưu tiên |
|---|---|---|
| 2.1 | Đếm xung đột STC | 🔴 |
| 2.2 | F1 bất biến thứ tự | 🔴 |
| 1.3 | Verify `dummy_idx` vs độ dài chuỗi ghép | 🔴 |
| 2.3 | Tách precision / recall | 🟠 |
| 2.4 | Lỗi theo độ dài entity | 🟠 |
| 2.5 | Lỗi quanh biên window | 🟠 |
| 2.6a | Pair-level F1 end-to-end (entity dự đoán, không oracle) | 🟠 |
| 2.6b–d | Tách domain · audit label noise · 3 seed | 🟡 |

### Sprint 1 — Decoding (**không train lại**)

| # | Việc | Chi phí |
|---|---|---|
| 3.1 | Hungarian decoding cho STC | ~40 dòng |
| 1.2 | Chặn cross-class + chu trình | ~5 dòng |
| 3.2 | Mask hình học | ~30 dòng |
| 4.1 | Overlap 128 + layout-aware chunking | ~40 dòng |
| 3.3 | Beam search | ~50 dòng |

### Sprint 2 — Train lại, giữ kiến trúc

| # | Việc |
|---|---|
| 4.2 | Banded head (tận dụng prior locality của bạn) |
| 5.1 | Line-normalized bbox |
| 5.2 | Bbox augmentation, áp lên cả chữ in |
| 5.3 | Oversample chữ tay 30–40% |
| 6.1 | λ weighting cho 3 loss |
| 6.3 | `L_uniq` — dạy ràng buộc uniqueness lúc train |
| 5.4 | Shuffle 20% thứ tự token |
| 5.5 | Augraphy 6 augmentation |

### Sprint 3 — Kiến trúc

| # | Việc |
|---|---|
| 7.1 | Geometric bias trong pair score |
| 7.2 | Entity-level pooled repr + scheduled sampling |
| 7.3 | GOSE head |

### Sprint 4 — Nếu vẫn chưa đủ

Synthetic handwritten data (§5.6) · ROAP/RORE (§7.5) · PEneo decoder (§7.4) · domain-adaptive pre-training LayoutXLM

---

## 9. Giao thức ablation

1. Cố định split, seed set (≥3), số epoch, LR schedule giữa các thí nghiệm
2. Báo cáo **mean ± std** trên 3 seed, không phải best run
3. Mỗi lần báo cáo **5 con số**: EE-F1(exact) · EE-F1(orderless) · EL-F1 · **pair-F1 end-to-end** · số xung đột STC
4. Tách riêng **printed / handwritten**
5. Giữ một **held-out test set** không bao giờ dùng để chọn hyperparameter
6. Mỗi thay đổi một biến; nếu bundle, làm ablation ngược

---

## 10. Nếu chỉ làm được ba việc

1. **Đếm xung đột STC + F1 bất biến thứ tự** (§2.1, §2.2) — nửa ngày, và quyết định toàn bộ phần còn lại có đáng làm không.
2. **Hungarian decoding + chặn cross-class + mask hình học** (§3.1, §1.2, §3.2) — inference-only, tấn công trực diện cơ chế zigzag tìm thấy trong code.
3. **Line-normalized bbox + bbox augmentation áp lên cả chữ in** (§5.1, §5.2) — trị gốc rễ vấn đề chữ tay, biến 90% dữ liệu chữ in thành tài nguyên cho 10% chữ tay.

Cả ba giữ nguyên **LayoutXLM backbone** và **không cần annotation mới**.

---

## 11. Danh mục tài liệu

### Nền tảng pipeline hiện tại
- **BROS** — [arXiv:2108.04539](https://arxiv.org/abs/2108.04539) · [code](https://github.com/clovaai/bros) · [HF docs](https://huggingface.co/docs/transformers/model_doc/bros)
- **SPADE** (Hwang et al., 2021) — [arXiv:2005.00642](https://arxiv.org/abs/2005.00642)
- **LayoutXLM / XFUND** — [arXiv:2104.08836](https://arxiv.org/abs/2104.08836)

### Head / decoder thay thế
- **GOSE** (EMNLP 2023 Findings) — [arXiv:2305.13850](https://arxiv.org/abs/2305.13850) · [code](https://github.com/chenxn2020/GOSE) — *hỗ trợ LayoutXLM sẵn*
- **KVPFormer** (AAAI 2023) — [arXiv:2304.07957](https://arxiv.org/abs/2304.07957)
- **GeoLayoutLM** (CVPR 2023) — [arXiv:2304.10759](https://arxiv.org/abs/2304.10759) · [code](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/GeoLayoutLM)
- **PEneo** (ACM MM 2024) — [arXiv:2401.03472](https://arxiv.org/abs/2401.03472) · [code](https://github.com/ZeningLin/PEneo)
- **RE2** (NAACL 2024) — [arXiv:2305.14590](https://arxiv.org/abs/2305.14590)
- **SERA** (EMNLP 2021) — [arXiv:2110.09915](https://arxiv.org/abs/2110.09915) · [ACL](https://aclanthology.org/2021.emnlp-main.218/)
- **TPP** (EMNLP 2023) — [arXiv:2310.11016](https://arxiv.org/abs/2310.11016)
- **UNER** — [arXiv:2408.01038](https://arxiv.org/abs/2408.01038)
- **ESP** (CVPR 2023) — [arXiv:2303.13095](https://arxiv.org/abs/2303.13095)
- **HIP** — [arXiv:2411.01139](https://arxiv.org/abs/2411.01139)
- **Biaffine parser** (Dozat & Manning) — [arXiv:1611.01734](https://arxiv.org/abs/1611.01734)
- **TPLinker** (nguồn handshaking của PEneo) — [arXiv:2010.13415](https://arxiv.org/abs/2010.13415)

### Reading order
- **ROAP** (preprint 2026) — [arXiv:2601.05470](https://arxiv.org/abs/2601.05470)
- **RORE** — [arXiv:2409.19672](https://arxiv.org/abs/2409.19672)

### Augmentation / dữ liệu
- **Augraphy** — [arXiv:2208.14558](https://arxiv.org/abs/2208.14558) · [GitHub](https://github.com/sparkfish/augraphy)
- **Enhancing Document Key Information Localization Through Data Augmentation** (digital→handwritten, Form-NLU) — [arXiv:2502.06132](https://arxiv.org/abs/2502.06132)
- **LayTextLLM** (Shuffled-OCR SFT) — [arXiv:2407.01976](https://arxiv.org/abs/2407.01976)
- **Beyond Human Annotation** (survey, 2026) — [arXiv:2601.12318](https://arxiv.org/abs/2601.12318)
- **Advancing Offline HTR** (survey, 2025) — [arXiv:2507.06275](https://arxiv.org/abs/2507.06275)

### Bối cảnh
- **LayoutXLM vs. GNN** (hạn chế 512-token trên XFUND RE) — [arXiv:2206.10304](https://arxiv.org/abs/2206.10304)
- **Focal Loss** — [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- **LiLT** — [arXiv:2202.13669](https://arxiv.org/abs/2202.13669)
- **Document AI Recommendations** — [GitHub](https://github.com/SCUT-DLVCLab/Document-AI-Recommendations)
