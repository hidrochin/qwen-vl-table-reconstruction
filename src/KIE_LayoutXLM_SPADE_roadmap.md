# Lộ trình cải thiện KIE: LayoutXLM + SPADE/BROS decoder

**v3** — viết lại sau khi Sprint 0 và Sprint 1 được kiểm chứng thực nghiệm và **bị bác bỏ**, cùng với hai phát hiện mới có giá trị chẩn đoán cao.

> Pipeline: **LayoutXLM backbone → 3 head (ITC / STC / EL)**, loss = tổng 3 CE không trọng số, sliding window không overlap, ~5% dữ liệu chữ viết tay. OCR đã qua deskew/dewarp.

---

## 0. Trạng thái các giả thuyết cũ

| Giả thuyết (v1/v2) | Kết quả | Trạng thái |
|---|---|---|
| `parse_subsequent_words` vứt token khi xung đột (§1.1 v2) | Đã đo, không có xung đột đáng kể | ❌ Bác bỏ |
| Điều kiện dừng chuỗi chỉ chặn init token cùng class (§1.2 v2) | Code đã xử lý chặt chẽ | ❌ Bác bỏ |
| `dummy_idx` lệch sau khi nối window (§1.3 v2) | Verify, không lệch | ❌ Bác bỏ |
| Hungarian decoding cho STC | Đã thử, không cải thiện | ❌ Bác bỏ |
| Mask hình học khi decode | Đã thử, không cải thiện | ❌ Bác bỏ |
| Sliding window là nguồn lỗi lớn | Thiệt hại tại biên chỉ ~2% | 🟡 Hạ ưu tiên |
| Bbox augmentation cho chữ tay (§5.2 v2) | Đã train lại, không giải quyết vấn đề | ❌ Không đủ |
| **Band STC theo offset reading-order** (đề xuất giữa chừng) | — | ❌ **Tôi rút lại — xem §2.4** |

Việc cả năm giả thuyết về decoding đều sai **là một kết quả có giá trị**: nó loại trừ toàn bộ tầng decode ra khỏi phạm vi nghi vấn. Vấn đề nằm ở tầng **biểu diễn và huấn luyện**, không phải tầng suy luận.

---

## 1. 🔴 Chẩn đoán mới: model đang học shortcut theo thứ tự OCR

### 1.1 Bằng chứng

Ba quan sát của bạn, khi ghép lại, chỉ tương thích với đúng một giải thích:

1. **Chuỗi zigzag bám sát thứ tự OCR.** Với `ocr.txt` dạng `ngày / tháng / năm / sinh / tháng / sáu / năm / 08 / hai / 06 / nghìn / ...`, output của model gần như là chính chuỗi này đọc tuần tự.
2. **Softmax gần như phẳng** tại các link sai — không phải "phân vân giữa hai ứng viên hợp lý" mà là "không có tri thức nào về câu hỏi này".
3. **Hai mẫu chữ tay gần giống nhau, một đúng một sai.** Không giải thích được bằng nội dung value, nhưng giải thích được hoàn hảo bằng **OCR có đan xen dòng hay không**.

Kết luận: STC head không học "token nào là successor về mặt ngữ nghĩa và hình học". Nó học **`successor = token kế tiếp trong file OCR`**.

### 1.2 Vì sao đây là nghiệm tối ưu của bài toán bạn đang đặt ra

Với chữ in, OCR trả về thứ tự sạch. Heuristic "offset +1" đúng gần 100%. Model đạt 95% F1 **mà chưa bao giờ phải học điều gì thực sự** về layout. Gradient không có lý do nào để đẩy model đi xa hơn — shortcut đã đủ để tối thiểu hoá loss.

Khi OCR đan xen hai dòng (chữ tay, dấu tiếng Việt làm bbox chồng lấn dọc, line grouping của OCR engine gộp nhầm), heuristic gãy. Và model **không có gì để rơi về**, vì cơ chế thay thế chưa từng được học. Softmax phẳng chính là chữ ký của khoảng trống đó.

Điều này cũng giải thích vì sao Sprint 1 vô hiệu: **decoder không thể sửa được thứ nó không nhận được**. Khi score matrix gần đều, mọi thuật toán gán tối ưu — Hungarian, beam search, mask hình học — đều chỉ đang tối ưu hoá nhiễu.

### 1.3 Vì sao augmentation bbox ở Sprint 2 không giúp

Bbox jitter thay đổi toạ độ nhưng **không thay đổi thứ tự token**. Shortcut nằm ở thứ tự, nên augmentation không bao giờ chạm tới nó. Bạn đã tăng độ bền cho một tín hiệu mà model không dùng.

### 1.4 ⚠️ Điểm cần làm rõ trước khi đi tiếp: "GT đúng thứ tự" nghĩa là gì?

Đây là câu hỏi quan trọng nhất trong tài liệu này, vì hai câu trả lời dẫn đến hai lộ trình khác nhau hoàn toàn.

Bạn nói GT được đảm bảo "đi đúng thứ tự". Có hai cách hiểu:

**(a) GT đơn điệu tăng theo index OCR.** Khi đó chuỗi GT của entity ngày sinh chính là `tháng → sáu → năm → 08 → hai → 06 → nghìn → ...` — tức là **GT cũng zigzag**, và model bám theo thứ tự OCR đang... trả lời đúng. Điều này mâu thuẫn với việc bạn quan sát thấy lỗi. Nếu đây là trường hợp thực tế thì vấn đề không phải model mà là **chính nhãn đang mã hoá một thứ tự vô nghĩa**, và không mô hình nào học được từ đó.

**(b) GT theo thứ tự đọc ngữ nghĩa** (`08 → 06 → 2011 → ngày → mùng → tám → tháng → sáu → ...`). Khi đó GT **không đơn điệu** theo index OCR, model buộc phải học nhảy ngược — đúng thứ nó chưa bao giờ được dạy vì 95% dữ liệu chữ in không có tình huống này. Đây là giả thuyết tôi cho là đúng.

**Cách kiểm tra (5 phút, chạy trên server công ty):**

```python
def chain_monotonic(chain):
    return all(chain[i] < chain[i+1] for i in range(len(chain)-1))

rate = np.mean([chain_monotonic(c) for c in gt_chains])
print(rate)   # tách printed / handwritten
```

- Chữ in ≈ 100%, chữ tay ≈ 100% → **trường hợp (a)**, nhãn đang bị nhiễm thứ tự OCR sai. Phải sửa line grouping ở khâu OCR trước, mọi thứ khác là vô nghĩa.
- Chữ in ≈ 100%, chữ tay thấp hơn rõ rệt → **trường hợp (b)**, xác nhận chẩn đoán §1.1 và toàn bộ tài liệu này áp dụng được.

Tương ứng, đo phân bố offset:

```python
offsets = [c[k+1]-c[k] for c in gt_chains for k in range(len(c)-1)]
print(sum(o == 1 for o in offsets) / len(offsets))   # tỉ lệ link "+1"
```

Nếu ≥ 0.93 thì shortcut là nghiệm tối ưu — con số này định lượng chính xác mức độ hấp dẫn của đường tắt mà bạn cần phá.

---

## 2. Sprint 0′ — Chẩn đoán mới (nửa ngày, không train)

Toàn bộ chạy được nội bộ, không cần chia sẻ dữ liệu ra ngoài.

### 2.1 🔴 Permutation test — thí nghiệm quyết định

Lấy một mẫu **chữ in** mà model đang nối đúng 100%. Giữ nguyên bbox, giữ nguyên nhãn, chỉ **hoán vị thứ tự token** để mô phỏng đúng kiểu đan xen của OCR:

```python
def interleave_two_lines(order, boxes, line_id, lid_a, lid_b):
    """Gộp token của 2 dòng liền kề rồi sắp lại theo xmin — tái tạo lỗi OCR."""
    idx = [i for i in order if line_id[i] in (lid_a, lid_b)]
    idx_sorted = sorted(idx, key=lambda i: boxes[i][0])
    out, it = [], iter(idx_sorted)
    for i in order:
        out.append(next(it) if line_id[i] in (lid_a, lid_b) else i)
    return out
```

Chạy inference lại với thứ tự mới (hoán vị **cả** `input_ids`, `bbox`, và các nhãn tương ứng để chấm điểm).

| Kết quả | Diễn giải |
|---|---|
| Model vẫn nối đúng | Shortcut không tồn tại → chẩn đoán §1 sai, dừng lại và xem §7 |
| Model gãy y hệt giấy khai sinh | ✅ **Xác nhận** — và bạn vừa có cách sinh vô hạn test case từ 95% dữ liệu chữ in |

### 2.2 🔴 Ablation `position_ids`

```python
# đóng băng model, đặt position_ids = hằng số (hoặc hoán vị ngẫu nhiên)
outputs = model(input_ids=ids, bbox=bbox, position_ids=torch.full_like(ids, 2), ...)
```

Đo lại F1 trên tập chữ in. F1 sụp đổ → model dựa chủ yếu vào 1D positional embedding chứ không phải layout. Đây là phép đo **trực tiếp** mức độ phụ thuộc shortcut, và nó mâu thuẫn với tinh thần thiết kế của SPADE/BROS (vốn được cho là bất biến thứ tự).

> Lưu ý kỹ thuật: LayoutXLM kế thừa quy ước RoBERTa, `position_ids` bắt đầu từ `pad_token_id + 1 = 2`, không phải 0. Xem §3.1.

### 2.3 🟠 Phân tích **link sai đầu tiên**

Bạn nói không tìm thấy điểm chung giữa các trường hợp zigzag. Đây gần như chắc chắn là artifact của cách đo: khi link thứ *k* sai, chuỗi đã lạc sang vùng token khác nên mọi link từ *k+1* trở đi là **rác thứ cấp**. Bạn đang so sánh các chuỗi rác với nhau.

```python
def first_break(gt_chain, next_map):
    for k in range(len(gt_chain) - 1):
        i, j_true = gt_chain[k], gt_chain[k+1]
        j_pred = next_map.get(i, None)
        if j_pred != j_true:
            return dict(k=k, i=i, j_true=j_true, j_pred=j_pred,
                        offset_true=j_true - i, offset_pred=(j_pred - i) if j_pred else None,
                        cross_line=line_id[i] != line_id[j_true])
    return None
```

Thống kê trên toàn bộ lỗi: tỉ lệ `cross_line`, histogram `offset_true`, histogram `offset_pred`. Dự đoán: **`offset_pred` tập trung áp đảo ở +1**, còn `offset_true` phân tán. Đó chính là chữ ký shortcut, và cũng chính là "hành vi nhất quán" bạn cần để hậu xử lý.

### 2.4 ❌ Đính chính: đề xuất band theo offset là sai

Ở lượt trao đổi trước tôi đề nghị chặn `stc_logits` ngoài khoảng `0 < j - i ≤ K`. Với dữ liệu của bạn điều đó **có hại trực tiếp**: chính những link cần sửa lại là những link có offset lớn hoặc âm. Band sẽ khoá cứng luôn cái shortcut bạn đang muốn phá. Bỏ hẳn.

(Mục §4.2 "banded head" của v2 cũng bị xoá vì cùng lý do.)

### 2.5 🟠 Tỉ lệ zigzag theo từng trường

Trường `Ngày tháng năm sinh` có một đặc điểm không trường nào khác có: **value chứa cùng thông tin hai lần**, một lần bằng số một lần bằng chữ (`08`↔`tám`, `06`↔`sáu`, `2011`↔`hai nghìn không trăm mười một`). Head bilinear tính score từ hidden state; hai token mang nội dung ngữ nghĩa gần trùng nhau sẽ có key vector gần nhau → score gần nhau → argmax lung lay.

Nếu tỉ lệ zigzag của trường này cao vượt trội so với các trường viết tay dài khác (`Nơi thường trú`, `Nơi đăng ký` — cũng dài, cũng xuống dòng), thì đây là **nguyên nhân thứ hai chồng lên shortcut**, và cách chữa là geometric bias (§5.1) chứ không phải augmentation.

### 2.6 🟠 Còn giữ lại từ v2

| # | Việc | Ghi chú |
|---|---|---|
| a | F1 bất biến thứ tự (frozenset thay vì tuple) | Tách "sai gom nhóm" khỏi "sai thứ tự" — vẫn rất đáng đo |
| b | Pair-level F1 end-to-end (entity dự đoán, không oracle) | Metric north-star |
| c | Tách metric printed / handwritten | Bắt buộc từ giờ trở đi |
| d | Train 3 seed, tính std | Quyết định ngưỡng ý nghĩa. Rẻ nhất trong bảng |
| e | EL: loss tính trên first-token GT nhưng infer dùng first-token dự đoán? | Mismatch train/infer, xem §5.2 |

---

## 3. Giải thích: 1D positional embedding trong LayoutXLM

Bạn hỏi đúng chỗ. Config bạn gửi có **ba nhóm** cơ chế vị trí khác nhau, và tôi đã nói không đủ rõ ở lượt trước.

### 3.1 Nhóm 1 — Absolute 1D position embedding ← **đây là thứ tôi muốn nói**

```json
"max_position_embeddings": 514
```

Đây là bảng embedding `embeddings.position_embeddings` — một `nn.Embedding(514, 768)`, kế thừa từ XLM-R. Nó mã hoá **token này là token thứ mấy trong chuỗi**, tức là **chính xác thứ tự OCR**.

- Có pretrained weight đầy đủ ✅
- `514 = 512 + 2` vì quy ước RoBERTa: `position_ids` bắt đầu từ `padding_idx + 1 = 2`
- Đây là kênh mà shortcut §1 đi qua

Trong `forward`, nếu bạn không truyền `position_ids`, HF sẽ tự sinh `arange` từ `input_ids`. Bạn **có thể** truyền vào:

```python
# vô hiệu hoá hoàn toàn
position_ids = torch.full_like(input_ids, 2)

# hoặc hoán vị (mềm hơn, khuyến nghị)
perm = torch.randperm(seq_len, device=ids.device) + 2
position_ids = perm.unsqueeze(0).expand_as(input_ids)
```

⚠️ Vô hiệu hoá **hoàn toàn** rất rủi ro: 1D-PE đan xen chặt với backbone văn bản XLM-R đã pretrain. Bỏ nó đi khiến biểu diễn ngôn ngữ suy giảm mạnh, và bạn sẽ mất nhiều hơn được. **Dùng biến thể hoán vị theo tỉ lệ sample** (§4.2) thay vì tắt cứng.

### 3.2 Nhóm 2 — 2D layout embedding (đang hoạt động)

```json
"max_2d_position_embeddings": 1024,
"coordinate_size": 128,
"shape_size": 128
```

Mã hoá `x0, y0, x1, y1, w, h` đã chuẩn hoá về thang 0–1000. Có pretrained weight, đang chạy bình thường. **Không đụng vào.** Ngược lại, đây là kênh bạn muốn model dựa vào nhiều hơn.

### 3.3 Nhóm 3 — Relative attention bias ← **thứ bạn hỏi, và bạn đoán đúng**

```json
"has_relative_attention_bias": false,
"has_spatial_attention_bias": false,
"rel_pos_bins": 32,      "max_rel_pos": 128,
"rel_2d_pos_bins": 64,   "max_rel_2d_pos": 256
```

Đây là cơ chế của LayoutLMv2: cộng thêm bias vào attention logits dựa trên **khoảng cách tương đối** giữa hai token — `rel_pos` theo index chuỗi, `rel_2d_pos` theo toạ độ x/y.

Bạn nhận xét chính xác: **hai cờ này `false` trong checkpoint LayoutXLM, nên không có pretrained weight cho `rel_pos_bias` và `rel_pos_x_bias`/`rel_pos_y_bias`.** Bốn tham số `rel_pos_bins`, `max_rel_pos`, `rel_2d_pos_bins`, `max_rel_2d_pos` đang nằm im, không có tác dụng gì.

**Đây không phải thứ tôi định nói ở lượt trước** — tôi nói về nhóm 1. Nhưng nhóm 3 hoá ra là một **cơ hội độc lập đáng giá**, xem §5.3.

### 3.4 Bảng tổng kết

| Cơ chế | Config | Pretrained? | Vai trò trong vấn đề của bạn | Hành động |
|---|---|---|---|---|
| Absolute 1D PE | `max_position_embeddings: 514` | ✅ Có | 🔴 **Kênh của shortcut** | Hoán vị theo tỉ lệ (§4.2) |
| 2D layout embedding | `max_2d_position_embeddings: 1024` | ✅ Có | Kênh bạn muốn tăng cường | Giữ nguyên |
| Relative 1D bias | `has_relative_attention_bias: false` | ❌ Không | Đang tắt | Để tắt |
| Relative 2D spatial bias | `has_spatial_attention_bias: false` | ❌ Không | Đang tắt — **lãng phí** | Cân nhắc bật, train from scratch (§5.3) |

---

## 4. Sprint A — Phá shortcut (ưu tiên tuyệt đối)

Ba việc dưới đây cùng một mục tiêu: làm cho "offset +1" **không còn là tín hiệu dự đoán được**, buộc model phải học hình học và ngữ nghĩa.

### 4.1 🔴 Augmentation đan xen dòng — mô phỏng đúng lỗi OCR

Không shuffle ngẫu nhiên. Shuffle ngẫu nhiên tạo ra phân bố mà production không bao giờ gặp, và model sẽ học cách bỏ qua thứ tự **quá mức**, mất luôn tín hiệu hữu ích khi OCR đúng. Mô phỏng đúng cơ chế gây lỗi:

```python
def augment_ocr_interleave(order, boxes, line_id, labels, p=0.35, max_pairs=2):
    """Mô phỏng lỗi gộp dòng của OCR engine: chọn 1-2 cặp dòng liền kề,
    gộp token rồi sắp lại theo xmin. Áp lên dữ liệu CHỮ IN."""
    if random.random() > p:
        return order, labels
    lids = sorted(set(line_id))
    new_order = list(order)
    for _ in range(random.randint(1, max_pairs)):
        if len(lids) < 2:
            break
        a = random.randrange(len(lids) - 1)
        la, lb = lids[a], lids[a + 1]
        pos = [k for k, i in enumerate(new_order) if line_id[i] in (la, lb)]
        toks = sorted((new_order[k] for k in pos), key=lambda i: boxes[i][0])
        for k, t in zip(pos, toks):
            new_order[k] = t
    return new_order, permute_labels(labels, order, new_order)
```

⚠️ Bắt buộc hoán vị **cả `itc_labels`, `stc_labels`, `el_labels`** theo đúng permutation. `stc_labels` và `el_labels` là ma trận index → phải remap **cả hai chiều**. Sai chỗ này là bạn đang train trên nhãn rác:

```python
def permute_labels(stc_label, old_to_new):
    """stc_label[j] = i (predecessor). Cần remap cả key lẫn value."""
    N = len(stc_label)
    out = np.full(N, DUMMY, dtype=np.int64)
    for j_old, i_old in enumerate(stc_label):
        if i_old == DUMMY:
            continue
        out[old_to_new[j_old]] = old_to_new[i_old]
    return out
```

Đây là cách biến **95% dữ liệu chữ in thành dữ liệu huấn luyện cho chính xác chế độ lỗi của 5% chữ tay** — mạnh hơn nhiều so với bbox augmentation vì nó chạm đúng vào kênh mà shortcut đi qua.

Bắt đầu với `p = 0.3`, tăng dần nếu F1 chữ tay cải thiện mà F1 chữ in không giảm.

### 4.2 🔴 Hoán vị `position_ids`

Bổ sung cho §4.1, ở tầng khác:

```python
# 30-50% sample: hoán vị position_ids, GIỮ NGUYÊN thứ tự input_ids và bbox
if random.random() < 0.4:
    position_ids = torch.randperm(L, device=dev)[None] + 2
else:
    position_ids = None   # để HF tự sinh arange
```

Khác biệt so với §4.1: §4.1 xáo trộn thứ tự thật (model thấy chuỗi khác), §4.2 giữ nguyên chuỗi nhưng làm nhiễu tín hiệu vị trí. Hai cái tấn công shortcut từ hai phía. Nên thử riêng lẻ trước để biết cái nào ăn điểm.

Biến thể nhẹ hơn nếu §4.2 làm sập chất lượng ngôn ngữ: **scale 1D-PE bằng một hệ số học được** khởi tạo ở 1.0, để model tự quyết định giảm phụ thuộc.

### 4.3 🟠 Shuffled-OCR SFT (LayTextLLM)

Kỹ thuật gốc: xáo trộn 20% số mẫu. Đây là phiên bản ít nhắm đích hơn §4.1 nhưng đã có tiền lệ công bố ([arXiv:2407.01976](https://arxiv.org/abs/2407.01976)). Dùng như đối chứng để chứng minh §4.1 (có mục tiêu) tốt hơn §4.3 (ngẫu nhiên) — nếu bằng nhau thì bạn tiết kiệm được công sức triển khai.

### 4.4 ⚠️ Bù lại tín hiệu đã lấy đi

Khi bạn lấy đi shortcut mà không cho gì thay thế, model sẽ đơn giản là **kém đi**. §4 phải đi cùng §5 trong cùng một vòng train, không tách rời.

---

## 5. Sprint B — Tăng tín hiệu hình học

### 5.1 🔴 Geometric bias trong pair score, dùng khung toạ độ cục bộ

Head bilinear của BROS chỉ thấy hình học **gián tiếp** qua hidden state. Cộng bias tính trực tiếp từ toạ độ:

```python
class GeoBiasedPairScore(nn.Module):
    def __init__(self, d, n_geo=14, d_geo=64):
        super().__init__()
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.geo = nn.Sequential(nn.Linear(n_geo, d_geo), nn.GELU(), nn.Linear(d_geo, 1))
        self.scale = d ** -0.5

    def forward(self, h, geo_feat):          # geo_feat: (B, N, N, n_geo)
        content = (self.q(h) @ self.k(h).transpose(-1, -2)) * self.scale
        return content + self.geo(geo_feat).squeeze(-1)
```

**Về feature hình học — bạn đã deskew/dewarp nên lo ngại ban đầu giảm đáng kể, nhưng chưa biến mất.** Deskew ở mức trang không sửa được **baseline drift và slant trong từng dòng chữ tay** — người viết tay vẫn đi lên đi xuống trong cùng một dòng. Nên vẫn dùng khung cục bộ, nhưng bây giờ nó là biện pháp phòng ngừa chứ không phải yêu cầu bắt buộc:

```python
def local_frame_features(boxes, i, j, neigh_k=7):
    """Vị trí tương đối của j so với i, trong hệ toạ độ xoay theo hướng dòng cục bộ tại i."""
    theta = fit_local_baseline(boxes, i, k=neigh_k)   # regression y_center ~ x qua k box lân cận
    h_loc = local_xheight(boxes, i, k=neigh_k)        # percentile 25-75, tránh dấu tiếng Việt
    dx, dy = center(boxes[j]) - center(boxes[i])
    d_par  = ( dx * cos(theta) + dy * sin(theta)) / h_loc   # dọc theo dòng
    d_perp = (-dx * sin(theta) + dy * cos(theta)) / h_loc   # vuông góc với dòng
    return [d_par, d_perp,
            iou_x(boxes[i], boxes[j]), iou_y(boxes[i], boxes[j]),
            width(boxes[j]) / width(boxes[i]), h_loc_ratio(boxes, i, j),
            gap_x(boxes[i], boxes[j]) / h_loc,
            float(d_perp > 0.5), float(d_perp < -0.5),      # xuống dòng / lên dòng
            *margin_features(boxes, i), *margin_features(boxes, j)]
```

Điểm mấu chốt: đây vẫn là **hình học học được**, không phải thuật toán cứng. Bạn không áp đặt quy tắc "phải nối sang phải" — bạn chỉ đưa cho model tín hiệu ở dạng bất biến với skew và để nó tự quyết định.

Tiền lệ: đây là **spatial compatibility attention bias** của KVPFormer ([arXiv:2304.07957](https://arxiv.org/abs/2304.07957)), thứ giúp họ đạt kết quả mạnh trên RE **mà không cần pre-training** — nên port sang LayoutXLM được. GeoLayoutLM ([arXiv:2304.10759](https://arxiv.org/abs/2304.10759)) cũng chỉ ra rằng hình học tường minh mới là yếu tố quyết định ở RE, còn LayoutLMv3 phụ thuộc quá mức vào ngữ nghĩa.

### 5.2 🟠 Feature lề — phân biệt "value xuống dòng" với "value dừng giữa dòng"

Đây là ràng buộc mà form của bạn cần model học được, và nó rẻ:

| Feature | Ý nghĩa |
|---|---|
| `is_line_final`, `is_line_initial` | Token cuối/đầu dòng |
| `dist_to_right_margin / char_width` | Value xuống dòng thường kết thúc **gần lề phải** |
| `left_indent / char_width` | Dòng tiếp theo của value thường thụt lề khác dòng mới |
| `n_kv_pairs_on_line` | Dòng `Dân tộc: Kinh · Quốc tịch: Việt Nam · Năm sinh: 1980` có 3 cặp |

Nhóm cuối đáng chú ý riêng: form giấy khai sinh có nhiều dòng chứa 2–3 cặp key–value. Model phải học "dừng value trước khi key kế tiếp bắt đầu". **Đo riêng error rate trên dòng đa-cặp so với dòng đơn-cặp** — tôi ngờ đây là nhóm lỗi lớn thứ hai sau shortcut.

### 5.3 🟢 Bật `has_spatial_attention_bias` — tận dụng cơ chế đang bỏ không

Như §3.3, LayoutXLM tắt cờ này và không có pretrained weight. Nhưng **module này rất nhỏ**: bảng bias `rel_2d_pos_bins × num_heads` cho x và y, chia sẻ giữa các layer. Train from scratch hoàn toàn khả thi khi bạn đang fine-tune trên dữ liệu domain-specific.

```python
cfg.has_spatial_attention_bias = True
model = LayoutLMv2Model.from_pretrained(path, config=cfg)   # bias init ngẫu nhiên
# đặt LR riêng cho các tham số bias, cao hơn backbone
```

Lợi ích: đưa hình học vào **mọi layer attention**, không chỉ ở pair head cuối cùng. Đây đúng là thứ bạn cần khi đang cố giảm phụ thuộc vào 1D-PE.

⚠️ Rủi ro: bias khởi tạo ngẫu nhiên trong 12 layer có thể phá vỡ biểu diễn pretrain giai đoạn đầu. Giảm thiểu bằng: khởi tạo bias ≈ 0, warmup dài hơn, và **giữ `has_relative_attention_bias = False`** (bias 1D — bạn đang muốn giảm phụ thuộc thứ tự, không phải tăng).

Đây là một ablation độc lập, đo được rõ ràng. Đáng thử vì chi phí thấp và không ai ngăn bạn tắt lại.

---

## 6. Sprint C — Thay đổi cấu trúc head

Phần này bạn chưa thử. Giữ nguyên từ v2 và bổ sung cách kết hợp với chẩn đoán mới.

### 6.1 🔴 BIO head phụ trợ + ensemble lúc decode

Nếu §1.4 cho thấy phần lớn offset GT là `+1`, chuỗi STC gần tương đương segmentation liên tục. Một head BIO song song **không thể sinh zigzag về mặt cấu trúc**:

```python
score(i → j) = log p_stc(i → j) + β · log p_bio(j = continuation)
```

Cơ chế: khi STC lưỡng lự (softmax phẳng — đúng triệu chứng bạn quan sát), BIO kéo về nghiệm liên tục. Khi entity thật sự gián đoạn, STC đủ tự tin để thắng.

Chi phí: một linear head 3 lớp. Nhãn suy trực tiếp từ chuỗi GT hiện có, **không cần annotate lại gì cả**. Đây có lẽ là tỉ lệ lợi ích/công sức cao nhất trong Sprint C.

### 6.2 🟠 Head đối xứng — gom nhóm thay vì chuỗi có hướng

Bạn đã cân nhắc và từ chối vì lo thuật toán sắp xếp theo x/y nhầm khi ảnh nghiêng. Hai điểm đáng xem lại:

**(a) OCR của bạn đã deskew/dewarp.** Lo ngại ban đầu giảm đáng kể. Phần còn lại là baseline drift trong dòng chữ tay — xử lý được bằng khung cục bộ (§5.1) chứ không cần threshold cứng.

**(b) Thứ tự trong `ocr.txt` đã là kết quả của một thuật toán sắp xếp hình học rồi** — thuật toán của OCR engine, mà bạn không kiểm soát và nó đang sai. Lựa chọn thực tế không phải "thuật toán hay model", mà là **"thuật toán bạn kiểm soát hay thuật toán bạn không kiểm soát"**.

Nếu vẫn muốn giữ quyết định thứ tự cho model, có phương án lai:

```python
# head đối xứng quyết định GOM NHÓM (bất biến thứ tự, dễ học)
S = 0.5 * (P_pair + P_pair.T)
groups = connected_components(S > tau)

# thứ tự trong nhóm: STC head cũ, nhưng chỉ trên tập ứng viên nhỏ (≤10 token)
# → không gian tìm kiếm giảm từ N² xuống k², softmax sắc hơn nhiều
```

Tách bài toán làm hai: "hai token này cùng entity không" (dễ, bất biến thứ tự) và "thứ tự trong nhóm 8 token" (dễ hơn nhiều so với chọn 1 trong 512). Model vẫn quyết định thứ tự, nhưng ở quy mô mà nó có thể học được.

Test rẻ: chạy nhánh gom nhóm với checkpoint hiện tại, đo bằng metric frozenset (§2.6a). Nếu F1 gom nhóm cao hơn F1 exact nhiều → hướng này đúng.

### 6.3 🟠 GOSE — head thay thế phù hợp nhất

[arXiv:2305.13850](https://arxiv.org/abs/2305.13850) · [GitHub](https://github.com/chenxn2020/GOSE) · EMNLP 2023 Findings

Repo chính thức dùng trực tiếp **LayoutXLM và LiLT** làm backbone — chi phí tích hợp thấp nhất trong các phương án.

Cơ chế: dự đoán quan hệ sơ bộ → khai thác *global structure knowledge* từ vòng trước → nhúng ngược vào biểu diễn entity → lặp K lần. Có thêm **spatial prefix** guide attention.

Vì sao phù hợp với chẩn đoán mới: bài báo nêu chính xác vấn đề của head độc lập — thiếu cấu trúc toàn cục khiến model *"struggle to learn long-range relations and easily predict conflicted results"*. Softmax phẳng của bạn là biểu hiện của việc mỗi cặp được quyết định độc lập, không có thông tin về các quyết định khác. GOSE giải ở **tầng học** thay vì tầng decode — mà tầng decode thì bạn đã chứng minh là vô ích.

### 6.4 🟠 PEneo — về cấu trúc không thể sinh zigzag

[arXiv:2401.03472](https://arxiv.org/abs/2401.03472) · [code](https://github.com/ZeningLin/PEneo)

Line grouping bằng **hai ma trận** head/tail (handshaking, kế thừa từ TPLinker) thay vì chuỗi tuần tự. Không có khái niệm "successor" nên **không thể sinh zigzag**.

Bản gốc cần re-annotate mức dòng, nhưng với nhãn hiện tại của bạn (chuỗi token có thứ tự) bạn **suy ra được nhãn cạnh head/tail** mà không cần annotate tay — chỉ là chuyển đổi format.

⚠️ License: non-commercial research only. Với production ở công ty, dùng làm tham khảo ý tưởng chứ đừng copy code.

### 6.5 🟡 EL đang vứt bỏ nội dung của value

Với `Ngày tháng năm sinh : 08/6/2011 ...`, head EL chỉ thấy first-token của key và first-token của value. Toàn bộ nội dung còn lại bị bỏ.

```python
e_i = torch.cat([h[first_i], h[span_i].mean(0), h[last_i]], dim=-1)
```

Rủi ro: train/inference mismatch. Giảm thiểu bằng **scheduled sampling** — tăng dần tỉ lệ dùng span dự đoán (0% → 50% qua các epoch).

> ⚠️ Kiểm tra ngay (§2.6e): nếu loss EL đang tính trên first-token **ground-truth** nhưng inference dùng first-token **dự đoán**, thì đã có sẵn mismatch — model chưa bao giờ học cách xử lý entity sai. Đây là ứng viên mạnh cho hiện tượng "validation tốt mà inference sai".

### 6.6 Các lựa chọn khác

| Method | Link | Ghi chú với chẩn đoán mới |
|---|---|---|
| **UNER** | [arXiv:2408.01038](https://arxiv.org/abs/2408.01038) | Xử lý entity **gián đoạn** — đúng bài toán §7.1 dưới đây |
| **RE2** | [arXiv:2305.14590](https://arxiv.org/abs/2305.14590) | Edge-aware GAT + constraint objective |
| **TPP** | [arXiv:2310.11016](https://arxiv.org/abs/2310.11016) | Token path prediction. Tốn memory |
| **GeoLayoutLM** | [arXiv:2304.10759](https://arxiv.org/abs/2304.10759) | RFE head chỉ 3.5% tham số, nhưng một phần lợi ích nằm ở pre-training bạn không có |
| **RORE** | [arXiv:2409.19672](https://arxiv.org/abs/2409.19672) | Model reading-order riêng, chịu skew, dùng pseudo-label. Xem §7.3 |

---

## 7. Vấn đề nhãn — cần quyết định, không phải kỹ thuật

### 7.1 🔴 `Ghi bằng chữ` là key in sẵn nhưng đang bị gán vào value

Đây có thể là vấn đề nghiêm trọng và độc lập với mọi thứ ở trên.

Trên form giấy khai sinh, `Ghi bằng chữ:` là **chữ in, có dấu hai chấm, nằm giữa dòng** — về mọi mặt hình thức nó là một key, y hệt `Nơi sinh:` hay `Dân tộc:`. Quy ước nhãn hiện tại của bạn buộc model phải:

- Nhận `Ghi bằng chữ` là **một phần của value**, trong khi mọi token có đặc điểm hình thức giống hệt ở khắp form đều là **key**
- Học một entity **gián đoạn về mặt ngữ nghĩa**: value = `08/6/2011` + `ngày mùng tám tháng sáu...`, với một cụm chữ in chen giữa

Đây là hai tín hiệu mâu thuẫn trực tiếp với nhau. Model không có cách nào phân biệt `Ghi bằng chữ` (phải nuốt) với `Quốc tịch` (phải dừng) ngoài việc học thuộc lòng chuỗi ký tự cụ thể đó.

**Đề xuất: tách thành hai cặp key–value độc lập, rồi gộp ở post-process.**

```
key: "Ngày, tháng, năm sinh"  →  value: "08/6/2011"
key: "Ghi bằng chữ"           →  value: "ngày mùng tám tháng sáu năm hai nghìn không trăm mười một"

# post-process: với template giấy khai sinh, merge hai cặp này thành một field
```

Lợi ích: nhãn trở nên nhất quán với hình thức trực quan, entity trở thành liên tục, và việc gộp là một quy tắc template đơn giản mà bạn kiểm soát hoàn toàn.

**Kiểm tra ngay:** đo error rate của các entity **liên tục** so với các entity **có key in chen giữa**. Nếu chênh lệch lớn, đây là nguyên nhân độc lập cần sửa bằng nhãn chứ không bằng model.

### 7.2 🟠 Audit nhất quán trên toàn tập

Kiểm tra 20–30 giấy khai sinh: `Ghi bằng chữ` có **luôn** được đánh cùng một cách không? Các trường nhiều cặp trên một dòng (`Dân tộc · Quốc tịch · Năm sinh`) có nhất quán không? Label noise ở mức vài phần trăm trong 5% dữ liệu chữ tay là đủ để triệt tiêu mọi cải thiện bạn đo được.

### 7.3 🟡 Reading order như một module riêng

Nếu §1.4 rơi vào trường hợp (a) — GT bị nhiễm thứ tự OCR sai — thì đây không còn là lựa chọn mà là bắt buộc.

Hai hướng:

- **Sửa line grouping trước khi tạo chuỗi token.** Ràng buộc mấu chốt: hai token cùng dòng **không được chồng lấn theo x**. Ràng buộc này chữ tay không phá vỡ, khác với y-threshold. Kèm theo: ước lượng chiều cao **thân chữ** (percentile 25–75 của y-center) thay vì chiều cao bbox, vì dấu tiếng Việt làm bbox cao gấp rưỡi và đó chính là thứ khiến OCR engine gộp nhầm hai dòng.
- **RORE** ([arXiv:2409.19672](https://arxiv.org/abs/2409.19672)) — ma trận nhị phân n×n + relation-aware attention, dùng pseudo-label từ model ROP có sẵn. Vẫn là "để model quyết định", nhưng tách bạch khỏi KIE nên dễ debug và dễ đo hơn nhiều.

⚠️ **Dù chọn hướng nào, vẫn phải làm §4.** Nếu không, model chỉ chuyển từ phụ thuộc vào thứ tự OCR sang phụ thuộc vào thứ tự của module mới — shortcut còn nguyên, và bạn gặp lại đúng vấn đề này khi module mới sai.

---

## 8. Sprint D — Dữ liệu

### 8.1 🔴 Synthetic chữ tay với **điểm ngắt dòng biến thiên**

Với 5% chữ tay và lỗi tập trung ở vài template cố định, đây là hướng ROI cao nhất.

Cơ chế cần mô phỏng: **độ rộng nét chữ quyết định chỗ ngắt dòng.** Người viết chữ to thì `ngày mùng tám` ngắt sau `mùng`; viết chữ nhỏ thì ngắt sau `tám` hoặc `tháng`. Layout giống hệt nhau nhưng cấu hình hình học tại đúng link khó nhất thì mỗi mẫu một khác — và điều này giải thích trọn vẹn hiện tượng "hai mẫu gần giống nhau, một đúng một sai" của bạn.

```
render value bằng font handwriting (hoặc model HTG) vào template giấy khai sinh in sẵn
  · biến thiên MẠNH độ rộng nét chữ  →  điểm ngắt dòng rơi vào mọi vị trí có thể
  · biến thiên baseline drift, slant, chiều cao từng chữ
  · giữ nguyên nhãn (bạn kiểm soát nội dung)
  · chạy qua CHÍNH OCR engine production để lấy thứ tự token thật, kể cả khi nó sai
```

Điểm cuối quan trọng nhất: **đừng sinh thứ tự token nhân tạo**. Cho ảnh synthetic đi qua đúng OCR engine bạn dùng, để dữ liệu train chứa đúng phân bố lỗi thứ tự mà production gặp.

Vài nghìn mẫu như vậy dạy model đúng cái bất biến nó đang thiếu: *"chỗ ngắt dòng ở đâu không quan trọng"*.

Tham khảo: [Advancing Offline HTR (2025)](https://arxiv.org/abs/2507.06275).

### 8.2 🟠 Augraphy — 6 augmentation đã sàng lọc

`InkBleed` · `Letterpress` · `LowInkRandomLines` · `LowInkPeriodicLines` · `JPEG` · `DirtyScreen`

Danh sách từ một paper giải đúng bài toán domain gap digital→handwritten trên Form-NLU ([arXiv:2502.06132](https://arxiv.org/abs/2502.06132)). Khỏi grid search 24 cái.

Lưu ý: cái này ảnh hưởng đến **chất lượng OCR**, không ảnh hưởng đến shortcut. Nó nằm ở tầng khác và bổ trợ chứ không thay thế §4.

### 8.3 🟠 Cân bằng domain

- Oversample document chữ tay lên **30–40% mỗi batch** (`WeightedRandomSampler`). Với 5%, tỉ lệ này quan trọng hơn nhiều so với khi có 10%.
- Fine-tune hai giai đoạn: train toàn bộ → fine-tune LR thấp trên tập chữ tay
- **Luôn báo cáo metric tách hai domain.** Con số 95% tổng thể đang che giấu vấn đề.

### 8.4 🟡 Line-normalized bbox (giữ từ v2, hạ ưu tiên)

```python
def line_normalize_boxes(boxes, tol=0.6):
    h_med   = np.median([b[3] - b[1] for b in boxes])
    yc      = np.array([(b[1] + b[3]) / 2 for b in boxes])
    line_id = cluster_1d(yc, eps=tol * h_med)
    out = np.array(boxes, dtype=float).copy()
    for lid in np.unique(line_id):
        m = line_id == lid
        out[m, 1], out[m, 3] = np.median(out[m, 1]), np.median(out[m, 3])
    return out, line_id
```

Vẫn hợp lý về nguyên tắc — `y0/y1` của chữ tay đang mã hoá **độ cao nét chữ** chứ không mã hoá **dòng**. Nhưng nó không chạm vào shortcut, nên xếp sau §4 và §5. Biến thể đáng thử: giữ cả hai tín hiệu — box chuẩn hoá vào layout embedding, `h_gốc / h_median` đưa vào head như feature phụ.

---

## 9. Loss & training

### 9.1 🟠 λ weighting cho 3 loss

| Head | Số lớp | Tỉ lệ dương |
|---|---|---|
| ITC | C+1 (~5–20) | cao |
| STC | N+1 (~513) | ~1/N |
| EL | N+1 (~513) | ~1/N |

CE trên 513 lớp có scale khác hẳn CE trên 10 lớp. Tổng thô nghĩa là bạn đang **ngầm gán trọng số theo cardinality**, không theo tầm quan trọng.

```python
loss = l_itc + λ_stc * l_stc + λ_el * l_el
```

Ablation `λ ∈ {0.5, 1, 2, 5}`. Rẻ, thường ăn 0.5–1.5 điểm.

> Lưu ý: §4.2 của v2 (banded head) từng được đề xuất như cách giảm mất cân bằng — **đã bị xoá** vì lý do §2.4. Thay thế bằng focal loss hoặc hard negative mining.

### 9.2 🟠 Focal loss + hard negative mining

- **Focal loss** (γ=2) cho STC/EL ([arXiv:1708.02002](https://arxiv.org/abs/1708.02002))
- **Hard negative mining**: chỉ lấy top-k negative có loss cao nhất (k ≈ 20× số positive)

Với softmax phẳng như bạn quan sát, focal loss đặc biệt đáng thử: nó tăng gradient ở đúng những ví dụ model đang lưỡng lự.

### 9.3 🟡 Consistency regularization

```python
# token được ITC coi là first-token thì không nên là successor của token khác
L_consist = (p_itc_is_first * p_stc_has_predecessor).sum() / N
loss = ... + λ_c * L_consist
```

> `L_uniq` (phạt nhiều successor) của v2 **đã bị xoá** — bạn đã đo và xác nhận không có xung đột successor. Không có gì để phạt.

### 9.4 Chi tiết dễ ăn điểm

- **Layer-wise LR**: backbone 2e-6 ~ 5e-6, head 1e-4 (PEneo dùng đúng cấu hình này). Với §5.3 (spatial bias train from scratch), đặt nhóm LR thứ ba cao hơn.
- **Train lâu hơn bạn nghĩ**: PEneo fine-tune **650 epoch** trên RFUND. Nếu bạn đang train 50–100 epoch, có thể chưa hội tụ — và điều này đặc biệt đúng sau khi bạn phá shortcut, vì bài toán trở nên khó hơn.
- EMA weights, label smoothing 0.05–0.1 cho ITC
- Ensemble 3–5 seed, average **score matrix** trước khi decode

---

## 10. Production — vá ngay, không cần train lại

### 10.1 🔴 Abstention theo margin

Softmax phẳng là một **tín hiệu abstention rất đáng tin** trong trường hợp của bạn, vì phẳng tương ứng khá sạch với "model không biết" (khác với trường hợp model tự tin sai, vốn không phát hiện được).

```python
margin = p_top1 - p_top2
doc_min_margin = min(margin[i] for i in all_predicted_links)
if doc_min_margin < tau:
    route_to_review(doc)      # hoặc rơi về post-process template
```

Calibrate `tau` trên validation: vẽ đường cong precision vs. coverage, chọn điểm phù hợp SLA. Đây là thứ chạy được **trong tuần này**, trong khi mọi thay đổi training cần vài vòng train.

### 10.2 🟠 Post-process theo template

Giấy khai sinh là form cố định. Bạn biết trước có bao nhiêu field, field nào ở đâu, field nào là ngày/năm/tên người. Với các document bị abstain:

- Ghép lại value từ các mảnh token bằng quy tắc template, **không cần đúng thứ tự nối của model**
- Cross-check: `08/6/2011` phải khớp với `ngày mùng tám tháng sáu năm hai nghìn không trăm mười một`. Hai biểu diễn của cùng một thông tin là **tài nguyên kiểm tra chéo miễn phí**, không phải chỉ là nguồn nhiễu (§2.5)
- Validate `Năm sinh` của cha/mẹ < `Năm sinh` của con, quốc tịch thuộc tập hữu hạn, v.v.

Không sang, nhưng với form pháp lý cố định đây là lựa chọn đúng và nó ăn nốt phần đuôi rẻ hơn bất kỳ thay đổi kiến trúc nào.

---

## 11. Lộ trình

### Sprint 0′ — Chẩn đoán (nửa ngày, không train)

| # | Việc | Ưu tiên | Quyết định điều gì |
|---|---|---|---|
| 1.4 | Kiểm tra tính đơn điệu của GT chain + phân bố offset | 🔴 | Toàn bộ tài liệu này có áp dụng được không |
| 2.1 | Permutation test trên mẫu chữ in đang đúng | 🔴 | Shortcut có thật không |
| 2.2 | Ablation `position_ids` | 🔴 | Mức độ phụ thuộc 1D-PE |
| 2.3 | Phân tích link sai đầu tiên | 🟠 | "Hành vi nhất quán" để hậu xử lý |
| 7.1 | Error rate: entity liên tục vs. có key in chen giữa | 🔴 | Vấn đề nhãn có độc lập không |
| 2.5 | Tỉ lệ zigzag theo từng trường | 🟠 | Có aliasing nội dung lặp không |
| 2.6a–d | F1 orderless · pair-F1 e2e · tách domain · 3 seed | 🟠 | Ngưỡng ý nghĩa |

### Sprint A — Phá shortcut (train lại, ~1 tuần)

| # | Việc | Ghi chú |
|---|---|---|
| 4.1 | Augmentation đan xen dòng, áp lên chữ in | 🔴 Cốt lõi |
| 4.2 | Hoán vị `position_ids` 30–50% sample | 🔴 Thử riêng lẻ trước để tách đóng góp |
| 5.1 | Geometric bias trong khung cục bộ | 🔴 Bắt buộc đi kèm §4 |
| 5.2 | Feature lề / line-final / n_kv_pairs | 🟠 Rẻ |
| 9.1 | λ weighting | 🟠 Rẻ |
| 8.3 | Oversample chữ tay 30–40% | 🟠 |

> §4 và §5 phải nằm trong **cùng một vòng train**. Lấy đi shortcut mà không bù tín hiệu thì model chỉ đơn giản kém đi.

### Sprint B — Song song, độc lập

| # | Việc |
|---|---|
| 10.1 | Abstention theo margin — **làm ngay tuần này** |
| 10.2 | Post-process template + cross-check số/chữ |
| 7.1 | Sửa quy ước nhãn `Ghi bằng chữ` |
| 5.3 | Bật `has_spatial_attention_bias`, train from scratch |
| 9.2 | Focal loss |

### Sprint C — Cấu trúc head

| # | Việc |
|---|---|
| 6.1 | BIO head phụ trợ + ensemble decode — **tỉ lệ lợi ích/công sức cao nhất** |
| 6.2 | Head đối xứng gom nhóm + STC cục bộ trong nhóm |
| 6.3 | GOSE (hỗ trợ LayoutXLM sẵn) |
| 6.5 | Entity-level pooled repr + scheduled sampling |

### Sprint D — Dữ liệu

| # | Việc |
|---|---|
| 8.1 | Synthetic chữ tay với điểm ngắt dòng biến thiên, chạy qua OCR engine thật |
| 8.2 | Augraphy 6 augmentation |
| 8.4 | Line-normalized bbox |

### Sprint E — Nếu vẫn chưa đủ

Reading-order module (§7.3) · PEneo decoder (§6.4) · UNER cho entity gián đoạn · domain-adaptive pre-training LayoutXLM · overlap 128 cho sliding window (đã biết chỉ ~2%)

---

## 12. Nếu chỉ làm được ba việc

1. **Sprint 0′ items 1.4, 2.1, 7.1** — nửa ngày, và quyết định toàn bộ phần còn lại có đúng hướng không. Đặc biệt §1.4: nếu GT đang đơn điệu theo index OCR thì mọi thứ khác trong tài liệu này là vô nghĩa và bạn phải sửa OCR line grouping trước.

2. **§4.1 + §5.1: augmentation đan xen dòng + geometric bias khung cục bộ** — tấn công trực diện shortcut, đồng thời cấp cho model tín hiệu thay thế. Biến 95% dữ liệu chữ in thành tài nguyên cho 5% chữ tay theo cách mà bbox augmentation không làm được.

3. **§10.1 + §10.2: abstention theo margin + post-process template** — vá production ngay tuần này, độc lập với mọi thay đổi model, và tận dụng đúng thứ bạn đã quan sát được (softmax phẳng).

Cả ba giữ nguyên **LayoutXLM backbone** và **không cần annotation mới** (ngoại trừ việc tách nhãn `Ghi bằng chữ`, vốn là chuyển đổi format chứ không phải gán nhãn lại).

---

## 13. Giao thức ablation

1. Cố định split, seed set (≥3), số epoch, LR schedule giữa các thí nghiệm
2. Báo cáo **mean ± std** trên 3 seed, không phải best run
3. Mỗi lần báo cáo **5 con số**: EE-F1(exact) · EE-F1(orderless) · EL-F1 · **pair-F1 end-to-end** · **F1 trên permutation test (§2.1)**
4. Tách riêng **printed / handwritten**, và tách riêng **entity liên tục / gián đoạn**
5. Giữ một **held-out test set** không bao giờ dùng để chọn hyperparameter
6. Mỗi thay đổi một biến; nếu bundle, làm ablation ngược

> Con số thứ 5 là mới và quan trọng nhất: **F1 trên tập chữ in đã bị hoán vị đan xen dòng** là proxy trực tiếp cho "shortcut đã bị phá chưa", và bạn đo được nó trên lượng dữ liệu lớn hơn nhiều so với 5% chữ tay thật.

---

## 14. Danh mục tài liệu

### Nền tảng pipeline hiện tại
- **BROS** — [arXiv:2108.04539](https://arxiv.org/abs/2108.04539) · [code](https://github.com/clovaai/bros)
- **SPADE** (Hwang et al., 2021) — [arXiv:2005.00642](https://arxiv.org/abs/2005.00642)
- **LayoutXLM / XFUND** — [arXiv:2104.08836](https://arxiv.org/abs/2104.08836)
- **LayoutLMv2** (kiến trúc gốc của config bạn gửi, §3) — [arXiv:2012.14740](https://arxiv.org/abs/2012.14740)

### Head / decoder thay thế
- **GOSE** (EMNLP 2023 Findings) — [arXiv:2305.13850](https://arxiv.org/abs/2305.13850) · [code](https://github.com/chenxn2020/GOSE) — *hỗ trợ LayoutXLM sẵn*
- **KVPFormer** (AAAI 2023, nguồn của spatial compatibility bias) — [arXiv:2304.07957](https://arxiv.org/abs/2304.07957)
- **GeoLayoutLM** (CVPR 2023) — [arXiv:2304.10759](https://arxiv.org/abs/2304.10759) · [code](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/GeoLayoutLM)
- **PEneo** (ACM MM 2024) — [arXiv:2401.03472](https://arxiv.org/abs/2401.03472) · [code](https://github.com/ZeningLin/PEneo) · ⚠️ non-commercial
- **RE2** (NAACL 2024) — [arXiv:2305.14590](https://arxiv.org/abs/2305.14590)
- **TPP** (EMNLP 2023) — [arXiv:2310.11016](https://arxiv.org/abs/2310.11016)
- **UNER** (entity gián đoạn) — [arXiv:2408.01038](https://arxiv.org/abs/2408.01038)
- **TPLinker** (nguồn handshaking của PEneo) — [arXiv:2010.13415](https://arxiv.org/abs/2010.13415)
- **Biaffine parser** — [arXiv:1611.01734](https://arxiv.org/abs/1611.01734)

### Reading order
- **RORE** — [arXiv:2409.19672](https://arxiv.org/abs/2409.19672)
- **ROAP** (preprint 2026, tự verify trước khi đầu tư) — [arXiv:2601.05470](https://arxiv.org/abs/2601.05470)

### Augmentation / dữ liệu
- **LayTextLLM** (Shuffled-OCR SFT) — [arXiv:2407.01976](https://arxiv.org/abs/2407.01976)
- **Augraphy** — [arXiv:2208.14558](https://arxiv.org/abs/2208.14558) · [GitHub](https://github.com/sparkfish/augraphy)
- **Enhancing Document Key Information Localization Through Data Augmentation** (digital→handwritten, Form-NLU) — [arXiv:2502.06132](https://arxiv.org/abs/2502.06132)
- **Advancing Offline HTR** (survey, 2025) — [arXiv:2507.06275](https://arxiv.org/abs/2507.06275)

### Bối cảnh
- **Focal Loss** — [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- **SERA** (EMNLP 2021) — [arXiv:2110.09915](https://arxiv.org/abs/2110.09915)
- **LiLT** — [arXiv:2202.13669](https://arxiv.org/abs/2202.13669)
- **Document AI Recommendations** — [GitHub](https://github.com/SCUT-DLVCLab/Document-AI-Recommendations)
