# v4 — Sau khi xác nhận shortcut: hai hướng và cách kết hợp

**Bối cảnh mới:** shortcut đã được xác nhận. Thuật toán sắp xếp trước model đôi khi trả về thứ tự sai; khi đó model bám vào shortcut và sinh zigzag. **Sửa tay thứ tự OCR → model trả kết quả hoàn hảo.**

Kết luận quan trọng nhất từ dữ kiện này: **model của bạn không bị hỏng.** Biểu diễn LayoutXLM + 3 head đã đủ tốt để giải bài toán khi đầu vào đúng. Bạn không cần đổi kiến trúc, không cần GOSE, không cần PEneo. Bạn có một bài toán **reading order**, và một bài toán **độ bền trước lỗi reading order**. Đó là hai hướng bạn nêu, và chúng bổ trợ chứ không thay thế nhau.

---

## 0. Định vị lại bài toán trước khi chọn thuật toán

Thuật toán sắp xếp có ba tầng, và chúng hỏng theo những cách rất khác nhau:

| Tầng | Việc | Mức độ khó với form của bạn |
|---|---|---|
| **T1. Gom từ thành dòng** | word nào thuộc cùng một dòng vật lý | 🔴 **Đây là chỗ hỏng** |
| T2. Sắp thứ tự các dòng | dòng nào trước dòng nào | 🟢 Dễ — form là layout Manhattan, sắp theo y là đủ |
| T3. Sắp thứ tự trong dòng | từ nào trước từ nào trong cùng dòng | 🟢 Dễ — sắp theo x |

Chuỗi lỗi `tháng sáu năm 08 hai 06 nghìn` là bằng chứng T1 hỏng: hai dòng vật lý bị gộp thành một, rồi T3 sắp lại theo x và trộn chúng vào nhau. T2 và T3 hoàn toàn vô tội.

**Hệ quả thực tiễn: bạn không cần XY-cut, không cần LayoutReader, không cần thuật toán "cực kỳ tuyệt vời" cho toàn trang.** XY-cut đạt 100% trên layout Manhattan — form của bạn thuộc loại đó. Vấn đề của bạn nằm ở tầng dưới nó: gom từ thành dòng khi bbox chữ tay chồng lấn dọc.

Kiểm tra 30 phút để xác nhận: lấy các mẫu bị lỗi, so sánh line grouping của thuật toán hiện tại với line grouping đúng (gán tay). Nếu 100% lỗi nằm ở T1, bạn tiết kiệm được rất nhiều công sức đi nhầm hướng.

---

## Phần A — Hướng 1: thuật toán sắp xếp chỉ từ bbox

### A.1 🔴 Vì sao chữ tay tiếng Việt phá vỡ line grouping

Đây là nguyên nhân gốc và nó rất cụ thể:

```
Dòng 1:  "ngày mùng tám"     bbox_height = 30px  (có chữ "g" thòng xuống + dấu huyền)
Dòng 2:  "tháng sáu"          bbox_height = 34px  (có "ố", "á" nhô lên + "g" thòng xuống)
```

Tiếng Việt có **hai tầng dấu**: dấu mũ/móc (ê, ô, ơ, ư, ă, â) cộng dấu thanh (sắc huyền hỏi ngã nặng) chồng lên trên, cộng dấu nặng chấm bên dưới. Một từ như `ưỡn` hay `ệ` có bbox cao gần **gấp đôi** thân chữ. Với chữ in, chiều cao dòng cố định nên vẫn không chồng nhau. Với chữ tay, người viết không giữ khoảng cách dòng đều → bbox dòng 1 và dòng 2 **chồng lấn dọc** → mọi thuật toán clustering theo y-center hoặc y-overlap đều gộp chúng.

Đây là lý do chỉ chữ tay tiếng Việt bị, và chỉ ở một số mẫu (phụ thuộc người viết).

### A.2 🔴 Ràng buộc quyết định: **hai từ cùng dòng không được chồng lấn theo x**

Đây là ràng buộc mạnh nhất bạn có, và chữ tay **không phá vỡ** nó. Con người không viết đè hai từ lên nhau theo chiều ngang trong cùng một dòng.

```python
def x_overlap_ratio(bi, bj):
    ov = min(bi[2], bj[2]) - max(bi[0], bj[0])
    return ov / max(1e-6, min(bi[2]-bi[0], bj[2]-bj[0]))

# nếu x_overlap_ratio > 0.3  →  CHẮC CHẮN khác dòng, bất kể y thế nào
```

Ràng buộc này một mình đã đủ để tách hai dòng bị gộp trong ví dụ giấy khai sinh: `08`(x≈255) và `hai`(x≈255) chồng nhau hoàn toàn theo x, nên không thể cùng dòng. Thuật toán hiện tại của bạn nhiều khả năng đang thiếu đúng ràng buộc này.

### A.3 🔴 Ước lượng thân chữ thay vì bbox height

Thay vì dùng `y1 - y0`, ước lượng vùng **thân chữ** (x-height band) — nơi phần lớn nét nằm, loại bỏ dấu và phần thòng:

```python
def body_band(boxes_in_line):
    """Ước lượng dải thân chữ của một dòng bằng percentile, chịu được dấu tiếng Việt."""
    tops    = np.array([b[1] for b in boxes_in_line])
    bottoms = np.array([b[3] for b in boxes_in_line])
    # baseline ổn định hơn đỉnh: dùng percentile thấp của bottom
    base = np.percentile(bottoms, 40)
    # x-height: dùng percentile cao của top thay vì min
    top  = np.percentile(tops, 60)
    return top, base
```

Nguyên tắc: **baseline (đáy thân chữ) là tín hiệu ổn định nhất của một dòng.** Đỉnh bbox bị dấu làm nhiễu mạnh, đáy bbox bị phần thòng (g, y, p, ạ, ợ) làm nhiễu nhẹ hơn. Ước lượng theo percentile chịu được cả hai.

Với box đơn lẻ chưa biết thuộc dòng nào, dùng thống kê toàn trang: `h_median` của tất cả box, rồi coi box nào cao hơn `1.4 × h_median` là "có dấu" và co lại về `h_median` khi tính y-overlap.

### A.4 🟠 Docstrum: ước lượng skew **cục bộ**, bất biến với ảnh nghiêng/cong

Đây là câu trả lời kinh điển cho yêu cầu "chịu được ảnh nghiêng nhẹ hoặc cong nhẹ", và nó có từ 1993 (O'Gorman, *The Document Spectrum*). Ưu điểm được chính tác giả nêu: độc lập với góc nghiêng, độc lập với khoảng cách chữ, và xử lý được các vùng có hướng khác nhau trong cùng một ảnh.

Cơ chế:

1. Với mỗi box, tìm `k` láng giềng gần nhất (k = 4–6) theo khoảng cách tâm
2. Tính histogram **góc** của các cặp láng giềng → đỉnh histogram là hướng dòng
3. Cặp "cùng dòng" = cặp có góc gần hướng dòng (±15°)
4. Đóng bao truyền ứng (transitive closure) các cặp cùng dòng → dòng

Điểm mấu chốt cho ảnh cong: **tính hướng dòng cục bộ cho từng box** từ chính k láng giềng của nó, thay vì một góc skew toàn cục. Ảnh cong thì hướng dòng biến thiên chậm theo vị trí, và ước lượng cục bộ bám theo được.

```python
def local_orientation(boxes, i, k=5):
    """Hướng dòng cục bộ tại box i, từ k láng giềng gần nhất."""
    c = centers(boxes)
    d = np.linalg.norm(c - c[i], axis=1)
    nb = np.argsort(d)[1:k+1]
    angles = np.arctan2(c[nb, 1] - c[i, 1], c[nb, 0] - c[i, 0])
    angles = np.where(angles > np.pi/2, angles - np.pi, angles)
    angles = np.where(angles < -np.pi/2, angles + np.pi, angles)
    # lọc bỏ các cặp gần vuông góc (láng giềng dòng trên/dưới)
    horiz = angles[np.abs(angles) < np.radians(40)]
    return np.median(horiz) if len(horiz) else 0.0
```

⚠️ Lưu ý thực tế: Docstrum được thiết kế cho **connected components**, còn bạn có **word boxes** từ OCR. Điều này thực ra thuận lợi hơn — ít nhiễu hơn, ít box hơn. Nhưng `k` cần nhỏ hơn (4–6 thay vì 4–5 cho CC) và cần lọc góc cẩn thận vì word box thưa hơn CC.

⚠️ Docstrum thuần **không đủ** cho chữ tay: literature nhất quán chỉ ra Docstrum và các phương pháp k-NN CC grouping thất bại trên tài liệu viết tay vì dòng nghiêng không đều, cong, và khoảng cách dòng không rõ hơn khoảng cách chữ. Nên phải kết hợp với A.2 (ràng buộc x-overlap) và A.5 (tối ưu toàn cục).

### A.5 🔴 Công thức đúng: **degree-constrained path cover với max-regret**

Đây là phần đáng giá nhất trong tài liệu này. Thay vì clustering rồi sắp xếp, hãy đặt cả T1 và T3 thành **một bài toán duy nhất**: tìm tập cạnh "successor" tối đa hoá tổng điểm, với ràng buộc mỗi node ≤1 successor và ≤1 predecessor, không chu trình.

Đây chính là formulation trong một bài 2026 về reading order cho layout phức tạp (arXiv:2607.01018): mỗi dòng OCR là một node trong đồ thị có hướng, và thứ tự đọc được khôi phục dưới dạng degree-constrained directed path cover. Áp dụng cho bạn ở mức **word** thay vì mức line — và điều đó hợp lý vì bài toán của bạn nằm ở tầng gom từ.

**Phát hiện quan trọng nhất của bài đó: cách chọn cạnh quan trọng hơn cách chấm điểm cạnh.**

Greedy (chọn cạnh điểm cao nhất trước) mắc lỗi *edge theft*: một cạnh sai sớm chiếm mất in-degree của một node, khiến predecessor đúng không còn chỗ, gây lỗi dây chuyền. Trên benchmark của họ, greedy chỉ đạt 56.0% trong khi max-regret đạt 93.0% — **chênh 37 điểm với cùng bộ score**. Số lỗi cross-stream giảm từ 27.8 xuống 4.4, same-stream từ 85.6 xuống 9.2.

Max-regret: ưu tiên quyết định có **chi phí cơ hội cao nhất**.

```python
def max_regret_path_cover(cand_edges, score, N):
    """
    cand_edges: dict {u: [v1, v2, ...]}  ứng viên successor của u
    score: score[(u,v)] -> float
    Trả về: {u: v} — path cover không chu trình
    """
    E, out_used, in_used = {}, set(), set()
    succ = {}                      # để kiểm tra chu trình bằng truy vết

    def creates_cycle(u, v):
        x = v
        for _ in range(N):
            if x == u: return True
            if x not in succ: return False
            x = succ[x]
        return True

    def feasible(u, v):
        return (u not in out_used and v not in in_used
                and u != v and not creates_cycle(u, v))

    while True:
        best_u, best_regret, best_list = None, -1, None
        for u in cand_edges:
            if u in out_used: continue
            C = sorted((v for v in cand_edges[u] if feasible(u, v)),
                       key=lambda v: -score[(u, v)])
            if not C: continue
            r = (score[(u, C[0])] - score[(u, C[1])]) if len(C) > 1 else 0.0
            if r > best_regret or (r == best_regret and best_u is not None
                                   and score[(u, C[0])] > score[(best_u, best_list[0])]):
                best_u, best_regret, best_list = u, r, C
        if best_u is None:
            break
        v = best_list[0]
        E[best_u] = v; succ[best_u] = v
        out_used.add(best_u); in_used.add(v)
    return E
```

Với `M` cạnh ứng viên và candidate list đã sort sẵn, độ phức tạp là `O(M log d)` — hoàn toàn chạy được real-time trên vài trăm word box.

**Vì sao formulation này đúng cho bạn:** nó cho ra **nhiều chuỗi rời rạc** (multiple disjoint paths), tức là nhiều dòng, chứ không ép toàn trang thành một chuỗi duy nhất. Line grouping và within-line ordering được giải đồng thời: mỗi path chính là một dòng, thứ tự trong path chính là thứ tự đọc. Không còn khâu clustering riêng để hỏng.

### A.6 🔴 Chấm điểm cạnh: hình học **cộng** ngôn ngữ

Ràng buộc cứng loại bỏ cạnh không hợp lệ; score xếp hạng phần còn lại.

**Ràng buộc cứng (đặt score = −∞):**

```python
def hard_infeasible(bi, bj, theta_i, h_body):
    if x_overlap_ratio(bi, bj) > 0.3:        return True   # §A.2
    if bj[0] < bi[2] - 0.1 * (bi[2]-bi[0]):  return True   # j không ở bên phải i
    d_par, d_perp = rotate_to_local_frame(bi, bj, theta_i)  # §A.4
    if abs(d_perp) > 0.6 * h_body:           return True   # lệch dòng quá nhiều
    if d_par > 6.0 * h_body:                 return True   # cách quá xa
    return False
```

**Score hình học:**

```python
s_geo = -w1 * (d_par / h_body) - w2 * abs(d_perp / h_body) - w3 * height_ratio_penalty(bi, bj)
```

**Score ngôn ngữ — đây là bổ sung mạnh nhất và bạn đang bỏ không.**

Bài 2026 nói trên chấm điểm cạnh bằng ensemble hai tín hiệu training-free: log-likelihood có điều kiện của một causal LM, và next-sentence-prediction của BERT. Ablation của họ cho thấy causal LM là tín hiệu chủ đạo, NSP đóng góp thêm ổn định, còn cosine similarity của sentence embedding **không giúp gì** và làm giảm kết quả khi tăng trọng số — nên bỏ hẳn.

Với tiếng Việt, đây là tín hiệu cực kỳ mạnh cho đúng ca của bạn:

```
P("tháng" | "ngày mùng tám")           →  rất cao
P("08"    | "ngày mùng tám")           →  rất thấp
P("hai"   | "08")                       →  thấp
```

Model không cần hình học để biết `tám → tháng` là đúng. Một causal LM tiếng Việt nhỏ (hoặc chính XLM-R với masked-LM scoring) trả lời câu này gần như chắc chắn đúng.

```python
def s_lm(text_u, text_v, lm, tok, kappa=0.5):
    ids_u, ids_v = tok(text_u).input_ids, tok(text_v).input_ids
    logits = lm(torch.tensor([ids_u + ids_v])).logits[0]
    lp = 0.0
    for k in range(len(ids_u), len(ids_u) + len(ids_v)):
        lp += log_softmax(logits[k-1], -1)[ids_v[k - len(ids_u)]]
    s = lp / len(ids_v)
    return s - kappa * uncond_logprob(ids_v, lm) / len(ids_v)   # chuẩn hoá tần suất
```

Chi phí: cache score theo cặp, chỉ tính cho các cạnh vượt qua ràng buộc cứng (thường vài trăm cạnh mỗi trang). Trên GPU là dưới một giây.

**Điểm cuối:**

```
S(u, v) = w_geo * s_geo + w_clm * s_clm + w_nsp * s_nsp
```

Tune trọng số **một lần** trên validation rồi cố định. Bài 2026 làm đúng vậy: sweep trên tập synthetic rồi transfer nguyên trọng số sang mọi dataset khác, không re-tune per document.

### A.7 🟢 Neo theo template — rẻ nhất, mạnh nhất cho form cố định

Giấy khai sinh là form in sẵn, bất biến. Các key in (`Họ và tên`, `Ngày, tháng, năm sinh`, `Ghi bằng chữ`, `Nơi sinh`, ...) luôn ở cùng vị trí tương đối và luôn là chữ in.

Thuật toán:

1. Nhận diện các key in bằng khớp chuỗi (chúng đọc rất chuẩn vì là chữ in)
2. Ước lượng biến đổi affine/homography từ vị trí key phát hiện được → vị trí key trong template chuẩn. **Đây cũng là bước deskew/dewarp chính xác hơn deskew toàn trang**, vì nó dùng chính điểm neo trong tài liệu.
3. Với mỗi word chữ tay, ánh xạ về toạ độ template → xác định nó thuộc **vùng field nào**
4. Trong mỗi field, sắp xếp bằng A.5 nhưng không gian tìm kiếm chỉ còn 5–15 word

Lợi ích: bài toán từ "sắp xếp 200 word trên trang nghiêng" thành "sắp xếp 8 word trong một ô đã biết trước". Sai số gần như không thể xảy ra. Ràng buộc: chỉ áp dụng được cho các loại mẫu bạn đã biết — nhưng bạn nói chỉ một số loại mẫu như giấy khai sinh mới trộn chữ tay và chữ in, nên độ phủ có thể rất cao.

Tôi khuyến nghị **làm A.7 trước A.5** nếu tập template của bạn hữu hạn. Nó rẻ hơn, dễ verify hơn, và chạy được ngay.

### A.8 Đo lường

Metric cho tầng reading order, độc lập với KIE:

```python
edge_accuracy = (# word có successor dự đoán khớp GT) / (# word không phải cuối chuỗi)
```

Đây là metric của bài 2026 và nó đúng cho bạn: nó đo trực tiếp thứ bạn cần, tách khỏi mọi nhiễu của model KIE. Tách riêng printed / handwritten, và tách riêng **lỗi trong dòng** (same-line skip) với **lỗi nhảy dòng** (cross-line link).

Bạn có sẵn ground truth: chuỗi GT của mỗi entity đã mã hoá thứ tự đúng. Không cần annotate thêm.

---

## Phần B — Hướng 2: ép model thực sự học

### B.0 Nguyên tắc

Model học shortcut vì shortcut **có tương quan cao với nhãn trong tập train**. Cách duy nhất để phá là làm cho tương quan đó biến mất. Ba mức độ can thiệp, từ nhẹ đến nặng:

| Mức | Cơ chế | Rủi ro |
|---|---|---|
| B.1 | Augmentation: làm nhiễu thứ tự | Thấp |
| B.2 | Consistency loss: phạt khi output đổi theo thứ tự | Trung bình |
| B.3 | Adversarial: ép hidden state không mã hoá thứ tự | Cao |

### B.1 🔴 Augmentation ở đúng mức granularity

Đã nêu ở v3 §4.1, nhưng có một chi tiết kỹ thuật quan trọng chưa nói:

⚠️ **Hoán vị ở mức WORD, không ở mức TOKEN.** XLM-R tách `phường` thành nhiều subword. Nếu bạn hoán vị ở mức token, các subword của cùng một từ bị tách rời và biểu diễn ngôn ngữ sụp đổ hoàn toàn — bạn sẽ kết luận nhầm rằng "phá shortcut làm model kém đi".

```python
def permute_words(word_groups, perm):
    """word_groups: list các list token index thuộc cùng một word.
    Hoán vị THỨ TỰ CÁC WORD, giữ nguyên thứ tự subword bên trong."""
    new_order = []
    for w in perm:
        new_order.extend(word_groups[w])      # subword giữ nguyên thứ tự
    return new_order
```

Phân bố hoán vị nên mô phỏng đúng lỗi thuật toán của bạn (gộp hai dòng liền kề rồi sắp theo x), không phải hoán vị ngẫu nhiên. Xem v3 §4.1.

### B.2 🔴 Consistency loss giữa hai hoán vị — đề xuất mạnh nhất của hướng 2

Đây là cách trực tiếp nhất để **định nghĩa** tính bất biến thứ tự thành một mục tiêu tối ưu, thay vì hy vọng model tự học được.

Cho cùng một document đi qua model **hai lần** với hai thứ tự token khác nhau, rồi phạt khoảng cách giữa hai phân bố dự đoán sau khi ánh xạ về cùng một không gian index chuẩn:

```python
def order_consistency_loss(model, batch, perm):
    out_a = model(**batch)                              # thứ tự gốc
    out_b = model(**apply_perm(batch, perm))            # thứ tự hoán vị

    # ánh xạ output của b về không gian index của a
    p_a = softmax(out_a.stc_logits, -1)                 # (N, N+1)
    p_b = softmax(unpermute(out_b.stc_logits, perm), -1)

    return 0.5 * (kl(p_a, p_b.detach()) + kl(p_b, p_a.detach()))

loss = l_itc + λ_stc*l_stc + λ_el*l_el + λ_cons * order_consistency_loss(...)
```

Vì sao mạnh hơn augmentation đơn thuần: augmentation chỉ dạy model rằng "thứ tự đôi khi khác", còn consistency loss dạy model rằng **output không được phép thay đổi khi thứ tự thay đổi**. Đó là ràng buộc đúng, và nó áp dụng cho mọi sample chứ không chỉ sample được augment.

Chi phí: gấp đôi forward pass. Có thể giảm bằng cách chỉ áp cho 30% batch.

⚠️ Cẩn thận với `unpermute` trên ma trận `(N, N+1)`: phải hoán vị **cả hai chiều** và xử lý cột dummy riêng. Sai chỗ này là loss trở thành nhiễu thuần tuý. Viết unit test: với `perm = identity`, loss phải bằng 0 chính xác.

### B.3 🟠 Adversarial debiasing — ép hidden state quên thứ tự

Nếu B.1 và B.2 chưa đủ, đây là can thiệp trực tiếp nhất: gắn một head phụ cố gắng **dự đoán chỉ số OCR của token từ hidden state**, và nối nó qua gradient reversal layer.

```python
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None

# head phụ: từ h_i đoán rank chuẩn hoá của token i trong chuỗi OCR
pos_pred = pos_head(GradReverse.apply(h, lambd))       # (N, 1)
l_adv    = F.mse_loss(pos_pred.squeeze(-1), norm_rank)

loss = ... + l_adv     # gradient reversal → backbone bị ép XOÁ thông tin thứ tự
```

Ý nghĩa: head phụ càng khó đoán được thứ tự OCR từ hidden state, thì hidden state càng ít mã hoá shortcut. `lambd` tăng dần theo schedule (0 → 0.1) để không phá training giai đoạn đầu.

⚠️ Rủi ro thật: gradient reversal nổi tiếng khó tune và có thể làm sụp training. Chỉ dùng nếu B.1 + B.2 đã cho kết quả nhưng chưa đủ. Và luôn giữ một baseline không adversarial để so sánh.

### B.4 🟠 Curriculum, không phải bật/tắt

Đừng bật augmentation 100% từ epoch 0. Model chưa có biểu diễn hình học tốt sẽ chỉ học được nhiễu.

```
epoch 0-10   : p_permute = 0.0      # học biểu diễn cơ bản với thứ tự sạch
epoch 10-30  : p_permute 0.0 → 0.4  # tăng tuyến tính
epoch 30+    : p_permute = 0.4, bật λ_cons
```

### B.5 🟢 Ensemble theo hoán vị lúc inference

Rẻ, không cần train lại, và cho bạn một tín hiệu chẩn đoán miễn phí:

```python
# chạy K=5 hoán vị khác nhau, ánh xạ về index gốc, trung bình score matrix
S = mean([unpermute(model(perm_k(x)).stc_logits, perm_k) for k in range(K)])
pred = decode(S)
```

Hai lợi ích:
- **Giảm variance**: nếu một hoán vị rơi vào trường hợp shortcut sai, các hoán vị khác kéo lại
- **Đo bất biến trực tiếp**: độ phân tán giữa K dự đoán chính là thước đo "model phụ thuộc thứ tự bao nhiêu". Đây là metric bạn nên báo cáo cho mọi thí nghiệm ở hướng 2.

Thử ngay với checkpoint hiện tại — nếu nó đã cải thiện, bạn có bằng chứng mạnh rằng hướng 2 đáng đầu tư.

### B.6 🟠 Multi-task: cho model tự dự đoán reading order

Thay vì lấy đi thứ tự, hãy **dạy model thứ tự đúng** như một task phụ:

```python
# head phụ: p(j là successor của i trong reading order đúng)  — ma trận (N, N)
l_ro = cross_entropy(ro_logits, gt_reading_order_successor)
loss = ... + λ_ro * l_ro
```

Nhãn có sẵn từ chuỗi GT của bạn. Lợi ích: hidden state buộc phải mã hoá "thứ tự đúng theo hình học và ngữ nghĩa" chứ không phải "thứ tự trong file OCR" — hai thứ khác nhau, và ép model phân biệt chúng chính là điều bạn muốn.

Đây cũng là cầu nối giữa hai hướng: head này có thể **thay thế** thuật toán ở Phần A, hoặc dùng để cross-check nó (khi hai bên bất đồng → abstain).

Tiền lệ: TPP ([arXiv:2310.11016](https://arxiv.org/abs/2310.11016)) xác nhận rằng một model dự đoán reading order dùng để sửa chuỗi token đầu vào cho các model layout-aware là có hiệu quả. RORE ([arXiv:2409.19672](https://arxiv.org/abs/2409.19672)) mô hình hoá reading order như quan hệ thứ tự bằng ma trận nhị phân n×n, dùng được pseudo-label.

---

## Phần C — So sánh và chiến lược

### C.1 Hai hướng giải hai bài toán khác nhau

| | Hướng 1 (thuật toán) | Hướng 2 (model) |
|---|---|---|
| Giải quyết | Đầu vào sai | Model không chịu được đầu vào sai |
| Verify được không? | ✅ Có — `edge_accuracy` đo trực tiếp | ❌ Khó — chỉ đo gián tiếp qua F1 |
| Cần train lại? | Không | Có, nhiều vòng |
| Rủi ro | Thấp, deterministic, debug được | Có thể làm model kém đi |
| Trần trên | Bị giới hạn bởi chất lượng OCR box | Không rõ |
| Thời gian tới production | Ngày–tuần | Tuần–tháng |

**Hướng 1 nên làm trước.** Không phải vì nó tốt hơn, mà vì:

1. Bạn **đã chứng minh** nó hoạt động — sửa tay thứ tự thì model hoàn hảo. Đây là bằng chứng thực nghiệm mạnh nhất bạn có trong toàn bộ dự án.
2. Nó verify được độc lập, không cần train lại.
3. Nó **sinh ra dữ liệu cho hướng 2**: một khi có thứ tự đúng, bạn có nhãn reading order để train head ở B.6, và có cặp (thứ tự sai, thứ tự đúng) để làm augmentation ở B.1.

**Nhưng đừng dừng ở hướng 1.** Một pipeline chỉ dựa vào thuật toán sắp xếp sẽ hỏng lặng lẽ khi gặp mẫu lạ trong production. Hướng 2 là bảo hiểm, và ít nhất B.5 (ensemble hoán vị) nên có mặt trong mọi phiên bản production.

### C.2 Kiến trúc pipeline tôi đề nghị

```
ảnh
 └─ deskew/dewarp (đã có)
     └─ OCR → word boxes + text
         └─ [A.7] neo template nếu nhận diện được form
             └─ [A.5] max-regret path cover
                  score = hình học (A.6) + LM tiếng Việt (A.6)
                  ràng buộc cứng = x-overlap (A.2) + thân chữ (A.3) + skew cục bộ (A.4)
                 └─ thứ tự token
                     └─ LayoutXLM + 3 head
                         [B.5] ensemble K hoán vị → score matrix trung bình
                        └─ decode
                            └─ [v3 §10.1] abstention theo margin
                                └─ [v3 §10.2] post-process template + cross-check số↔chữ
```

Ba tầng phòng vệ độc lập: thuật toán đúng → model bền → hậu xử lý bắt lỗi còn sót. Không tầng nào phải hoàn hảo.

### C.3 Lộ trình

**Tuần 1 — verify và vá**

| # | Việc | Kết quả mong đợi |
|---|---|---|
| §0 | Xác nhận lỗi nằm ở T1 (gom dòng), không phải T2/T3 | Định hướng toàn bộ phần còn lại |
| A.8 | Cài `edge_accuracy` trên tập validation | Có baseline để đo mọi thay đổi |
| A.2 | Thêm ràng buộc x-overlap vào thuật toán hiện tại | Có thể sửa được phần lớn ngay |
| B.5 | Ensemble 5 hoán vị lúc inference | Đo mức phụ thuộc thứ tự, có thể cải thiện luôn |

A.2 đáng thử đầu tiên vì chi phí gần bằng 0 và nó nhắm đúng cơ chế trong ví dụ giấy khai sinh.

**Tuần 2–3 — thuật toán**

| # | Việc |
|---|---|
| A.3 | Ước lượng thân chữ theo percentile |
| A.4 | Skew cục bộ kiểu Docstrum |
| A.5 | Max-regret path cover thay cho clustering + sort |
| A.6 | Thêm score LM tiếng Việt |
| A.7 | Neo template cho các mẫu đã biết |

**Tuần 4+ — model**

| # | Việc |
|---|---|
| B.1 | Augmentation hoán vị ở mức word, mô phỏng lỗi gộp dòng |
| B.4 | Curriculum cho tỉ lệ hoán vị |
| B.2 | Consistency loss giữa hai hoán vị |
| B.6 | Head reading order phụ trợ |
| B.3 | Adversarial debiasing — chỉ nếu trên chưa đủ |

### C.4 Metric báo cáo cho mọi thí nghiệm từ giờ

1. `edge_accuracy` của tầng reading order (printed / handwritten riêng)
2. EE-F1 exact và orderless
3. Pair-F1 end-to-end với entity dự đoán
4. **Độ phân tán dự đoán giữa K hoán vị** (B.5) — thước đo phụ thuộc thứ tự
5. F1 trên tập chữ in đã bị hoán vị nhân tạo (v3 §2.1)

Số 4 và 5 là hai con số mới, và chúng là thứ duy nhất cho bạn biết hướng 2 có thực sự tiến triển hay không.

---

## Phần D — Tài liệu

### Reading order — thuật toán
- **Docstrum** (O'Gorman 1993) — nearest-neighbor clustering, bất biến skew, xử lý được vùng đa hướng. Nền tảng cho §A.4
- **Recursive XY-Cut** (Ha, Haralick & Phillips 1995) — nền tảng, đạt 100% trên layout Manhattan nhưng cần khoảng trắng phân tách sạch
- **Optimized XY-Cut** (Meunier, ICDAR 2005) — dynamic programming, dưới 1 giây mỗi trang
- **XY-Cut++** — [arXiv:2504.10258](https://arxiv.org/abs/2504.10258) · ngưỡng thích ứng theo median box length, hierarchical mask. Có bản cài đặt trong OpenDataLoader PDF
- **Reading Order Inference for Complex Document Layouts** — [arXiv:2607.01018](https://arxiv.org/abs/2607.01018) · **quan trọng nhất cho §A.5/A.6**: path cover + max-regret + scoring bằng LM, training-free

### Reading order — model
- **LayoutReader / ReadingBank** — [GitHub](https://github.com/doc-analysis/ReadingBank) · seq2seq, 500K trang. ⚠️ Giám sát ở mức **word** từ file DOCX; bài 2607.01018 cho thấy nó transfer rất kém sang input mức dòng/đoạn và không bất biến với phép lật trang
- **RORE** — [arXiv:2409.19672](https://arxiv.org/abs/2409.19672) · ma trận thứ tự n×n, pseudo-label
- **TPP** — [arXiv:2310.11016](https://arxiv.org/abs/2310.11016) · xác nhận việc sửa thứ tự token đầu vào bằng model reading order là có hiệu quả
- **DLAFormer** — [arXiv:2405.11757](https://arxiv.org/abs/2405.11757) · unified label space cho nhiều relation prediction task
- **UniHDSA** — [arXiv:2503.15893](https://arxiv.org/abs/2503.15893)

### Line segmentation cho chữ tay
- Quirós & Vidal — reading order decoding cho tài liệu viết tay, giả định partial order ở mức element
- Handwritten Chinese text line segmentation by clustering with distance metric learning — nêu rõ vì sao projection analysis và k-NN CC grouping thất bại trên chữ tay
- Robust line segmentation for handwritten documents (CEDAR/Buffalo) — piece-wise projection + bivariate Gaussian cho dòng chồng nhau

### Đã dẫn ở v3
LayoutLMv2 · BROS · SPADE · LayoutXLM · GOSE · KVPFormer · GeoLayoutLM · PEneo · RE2 · UNER · Augraphy · LayTextLLM
