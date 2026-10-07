"""Build the submission from measured artifacts; refuse incomplete experiments."""
from __future__ import annotations

import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labkit import evaluate as ev


def load(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|",
                      *("| " + " | ".join(cell(v) for v in r) + " |" for r in rows)])


def relation(a, b):
    return "thắng" if a > b else "thua" if a < b else "hoà"


def main():
    required = ["mask_proof.json", "template_check.json", "token_stats.json",
                "baselines_frozen.json", "verdict.json", "autopsy.json", "qualitative.json",
                "environment.json", "runs.csv"]
    missing = [n for n in required if not (ROOT / "results" / n).is_file()]
    if missing:
        raise SystemExit(f"Experiment incomplete; report not overwritten. Missing: {missing}")
    proof, template, stats, frozen, verdict, environment = (
        load(n) for n in required[:5] + ["environment.json"])
    runs = {r["run"]: r for r in csv.DictReader(
        (ROOT / "results" / "runs.csv").open(encoding="utf-8", newline=""))}
    autopsy = {r["run"]: r for r in load("autopsy.json")}
    keys = ["correct", "attn_only", "wrong_lr", "qlora"]
    if any(k not in runs or k not in autopsy for k in keys):
        raise SystemExit("All four trained and scored runs are required")
    if frozen.get("smoke_mode"):
        raise SystemExit("Full evaluation required; abbreviated results cannot be submitted")
    if not (proof["answer_is_supervised"] and proof["question_is_masked"]):
        raise SystemExit("Mask proof failed")
    steps = {int(runs[k]["actual_steps"]) for k in keys}
    if len(steps) != 1 or any(int(runs[k]["max_steps"]) not in steps for k in keys):
        raise SystemExit("Actual and planned step budgets must match")
    c, a, w, q = (runs[k] for k in keys)
    budget_delta = abs(int(a["trainable_params"]) / int(c["trainable_params"]) - 1)
    if budget_delta >= 0.05:
        raise SystemExit("Attention placement contrast has an unmatched budget")
    comparisons = verdict["comparison"]
    ft = comparisons[2]
    gate = verdict["verdict"]
    b = frozen["baseline_b"]
    baseline_table = table(["Run", "Target", "Regression", "Format", "Latency ms/mẫu"],
                          [[r["run"], f'{r["target"]:.4f}', f'{r["regression"]:.4f}',
                            f'{r["format"]:.4f}', f'{r["latency_ms"]:.1f}'] for r in comparisons])
    run_table = table(["Run", "Vị trí", "r", "Trainable", "LR", "Mean train loss",
                       "Target", "Giây train", "Peak VRAM GB", "Actual steps"],
                      [[k, runs[k]["placement"], runs[k]["r"], runs[k]["trainable_params"],
                        runs[k]["learning_rate"], runs[k]["final_loss"],
                        f'{autopsy[k]["target"]:.4f}', runs[k]["train_seconds"],
                        runs[k]["peak_vram_gb"], runs[k]["actual_steps"]] for k in keys])
    curves = []
    for k in keys:
        history = load(f"training_{k}.json")
        losses = [r for r in history if "loss" in r]
        if losses:
            curves.append([k, losses[0]["step"], losses[0]["loss"],
                           losses[-1]["step"], losses[-1]["loss"]])
    curve_table = table(["Run", "Step đầu log", "Loss đầu", "Step cuối log", "Loss cuối"], curves)
    examples = load("qualitative.json")
    counts = {o: sum(r.get("outcome") == o for r in examples) for o in ("win", "loss", "tie")}
    selected = []
    for outcome in ("loss", "win", "tie"):
        available = sorted((r for r in examples if r.get("outcome") == outcome),
                           key=lambda r: (r["delta"] if outcome != "win" else -r["delta"], r["i"]))
        selected.extend(available[:2])
    for r in examples:
        if len(selected) >= 5:
            break
        if r not in selected:
            selected.append(r)
    selected = selected[:5]
    qualitative = []
    for num, r in enumerate(selected, 1):
        qualitative.append(
            f'### Ví dụ {num} — eval index {r["i"]}, {r["outcome"]}\n\n'
            f'Ticket: {r["ticket"]}\n\n'
            f'Nhãn:\n```json\n{json.dumps(r["label"], ensure_ascii=False)}\n```\n\n'
            f'Baseline (b), điểm {r["baseline_b_score"]:.2f}:\n```text\n{r["baseline_b_pred"]}\n```\n\n'
            f'Fine-tune, điểm {r["ft_score"]:.2f}, delta {r["delta"]:+.2f}:\n```text\n{r["ft_pred"]}\n```\n')
    saved_vram = float(c["peak_vram_gb"]) - float(q["peak_vram_gb"])
    q_delta = autopsy["qlora"]["target"] - autopsy["correct"]["target"]
    rank_order = sorted(keys, key=lambda k: (-autopsy[k]["target"], k))
    loss_order = sorted(keys, key=lambda k: float(runs[k]["final_loss"]))
    status = "PASSED" if gate["passed"] else "FAILED"
    decision = ("Cổng lab cho phép tiếp tục đánh giá pilot, nhưng chưa đủ để triển khai production."
                if gate["passed"] else "Không deploy adapter correct theo cổng đã định trước.")
    reason = " ".join(gate["reasons"])
    qualitative_text = "\n".join(qualitative)
    report = f'''# Lab 21 — Kết quả thực nghiệm LoRA trên ticket CSKH

**Học viên:** Đàm Việt Hưng — **MSSV:** 2A202602600 — **Ngày:** 07/10/2026.

**Kết quả:** {status}. {decision}

## 1. Thiết kế và khả năng tái lập

Chọn `{frozen["model"]}` vì GPU thực tế là {environment["device"]["name"]},
VRAM {environment["device"]["vram_gb"]} GB. Model nhỏ cho phép đo đủ bốn cấu hình trên
cùng phần cứng. Recipe LAPTOP giữ batch 1, gradient accumulation 8, batch hiệu dụng 8,
hai epoch, seed 42, precision {environment["precision"]}. Không suy rộng kết quả này
sang model 4B hay phần cứng T4. Lựa chọn được ghi trước huấn luyện trong
[EXPERIMENT.md](EXPERIMENT.md); phiên bản gói được pin trong
[requirements-lock.txt](requirements-lock.txt) và `results/environment.json`.

Giữ corpus tổng hợp mặc định: 250 ticket tiếng Việt thành JSON bốn trường; split
225 train / 25 validation, 50 ticket target và 15 câu regression. Chọn bài toán này vì
có thể chấm từng trường mà không cần LLM judge. Không thay dataset, không dùng holdout
để chọn cấu hình. Train không có ticket trùng hoàn toàn trong nội bộ hoặc với target;
tuy nhiên hai tập dùng cùng các khuôn tổng hợp, nên không coi đó là bằng chứng khái quát
ra dữ liệu khách hàng thật. Phân bố intent trong 250 mẫu: doi_tra 61, san_pham_loi 49,
hoi_thong_tin 49, hoan_tien 45, van_chuyen 46.

Mốc đóng băng lúc **{frozen["frozen_at_utc"]}**, trước NB3/NB4. Dùng toàn bộ
{frozen["n_target"]} target và {frozen["n_regression"]} regression, không EVAL_LIMIT.
SHA prompt tối ưu là `{frozen["optimized_prompt_sha"]}`; không sửa prompt (b).
NB5 kiểm tra model, prompt, số mẫu và checksum eval với mốc đóng băng. Các checksum
gốc trong `data/checksums.json` khớp sau khi khôi phục LF của Windows checkout;
không sửa nội dung dataset hay checksum tham chiếu.

## 2. Bằng chứng mask và độ dài

`results/mask_proof.json`: supervised {proof["n_supervised"]}/{proof["n_total"]} token,
tỷ lệ **{proof["supervised_fraction"]:.4f}**; answer_is_supervised =
**{proof["answer_is_supervised"]}**, question_is_masked = **{proof["question_is_masked"]}**.
Phần được tính loss giải mã thành:

```text
{proof["supervised_preview"]}
```

Template check: **{template["verdict"]}**. Template giữ reasoning khi assistant
có trace; corpus này chỉ có JSON. Không diễn giải valid_trace_rate =
{verdict["valid_trace_rate"]} là reasoning collapse, vì generation tắt thinking và
không huấn luyện trace. Kiểm tra tokenizer cho thấy assistant mask tự sinh có 0/31
token, trong khi mask của lab có 9/31 token. Vì thế SFT nhận labels đã token hoá theo
mask NB1, packing tắt, không dùng assistant_only_loss từ template.

Độ dài đo trên {stats["n"]} mẫu: mean {stats["mean"]}, p50 {stats["p50"]},
p95 **{stats["p95"]}**, p99 {stats["p99"]}, max {stats["max"]}; lab gợi ý
max_length {stats["suggested_max_length"]}. Giữ trần 1024 của tier để mọi đối chứng
chia sẻ recipe phần cứng. Batch 1 và dữ liệu pre-tokenized không pad cố định lên trần
đó; không mẫu nào bị cắt. `results/prompt_alignment.json` xác nhận prompt generation
là prefix của prompt training: mask đúng và conditioning cũng khớp.

## 3. Ba baseline và bốn nhóm đánh giá

{baseline_table}

Baseline (b) {relation(b["target"], frozen["baseline_a"]["target"])} baseline (a)
trên target. Fine-tune dùng prompt ngắn như dữ liệu train; baseline (b) giữ schema
và ví dụ đầy đủ. Target là trung bình độ chính xác bốn trường; regression là keyword
recall của 15 câu phổ thông. Format là trung bình tỷ lệ khoá cần thiết có trong JSON
trích xuất được, chưa phải kiểm tra JSON thuần hay schema nghiêm ngặt. Latency là
thời gian sinh trung bình mỗi mẫu trong batch 4, greedy, 160 token target / 96 token
regression. Đây không phải latency yêu cầu đơn lẻ hay benchmark đủ để cam kết SLA.

## 4. Giải phẫu cấu hình và loss

{run_table}

`final_loss` trong CSV là `TrainOutput.training_loss`, tức loss trung bình toàn run,
không phải loss ở step cuối. Log step được lưu riêng:

{curve_table}

### 4.1. Vị trí so với rank

`attn_only` {relation(autopsy["attn_only"]["target"], autopsy["correct"]["target"])}
`correct` trên target: {autopsy["attn_only"]["target"]:.4f} so với
{autopsy["correct"]["target"]:.4f}. Rank q/v là {a["r"]}, còn text-linear là
{c["r"]}; số tham số lệch **{budget_delta * 100:.3f}%**, nằm dưới 5%.
Đây là đối chứng vị trí ở ngân sách gần bằng nhau, không phải phép quét rank độc lập.
Thứ tự target: {" > ".join(rank_order)} (các điểm bằng nhau vẫn là hoà).
Thứ tự loss tăng dần: {" < ".join(loss_order)}. Việc hai thứ tự
{"giống" if rank_order == loss_order else "khác"} nhau cho thấy cần đọc trực tiếp
điểm tác vụ; không dùng loss làm tiêu chuẩn thắng. Một seed, model nhỏ và tác vụ hẹp
không đủ để tuyên bố vị trí hay rank luôn tốt hơn trong mọi trường hợp.

### 4.2. Learning rate

`wrong_lr` chỉ đổi LR từ {c["learning_rate"]} xuống {w["learning_rate"]}; ngân sách
step, mask và vị trí giữ nguyên. Mean loss là {w["final_loss"]} so với
{c["final_loss"]}; target là {autopsy["wrong_lr"]["target"]:.4f} so với
{autopsy["correct"]["target"]:.4f}. Các điểm đầu/cuối log trong bảng trên là bằng
chứng về đường loss, thay vì giả định LR nhỏ luôn tạo đường phẳng. Nếu chỉ nhìn loss,
có thể nhầm tốc độ thích nghi chậm do LR với thiếu năng lực của LoRA; phép đổi đúng
một biến giúp phân biệt hai giải thích. Tuy nhiên không dùng một run để xác nhận cơ
chế nhân quả phổ quát, và cần nhiều seed nếu chênh lệch nhỏ.

### 4.3. QLoRA và chi phí

Peak VRAM PyTorch của qlora là {q["peak_vram_gb"]} GB, correct là
{c["peak_vram_gb"]} GB: chênh lệch tiết kiệm **{saved_vram:+.2f} GB**.
Target đổi **{q_delta:+.4f}**; train mất {q["train_seconds"]} so với
{c["train_seconds"]} giây. QLoRA được chấm trên đúng base 4-bit dùng khi train,
để không trộn sai lệch base/adapter vào hiệu ứng lượng tử hoá. Số đo này chỉ mô tả
đánh đổi quan sát trên 0.8B; không đủ để kiểm chứng khuyến nghị cho toàn họ Qwen3.5.
VRAM là max_memory_allocated của PyTorch, không bao gồm mọi overhead hiển thị ở
nvidia-smi, và không coi 4-bit chắc chắn nhanh hơn nếu runtime không hỗ trợ kernel tốt.

## 5. Phán quyết và diễn giải

**{status}**; target delta **{gate["target_delta"]:+.4f}**, regression delta
**{gate["regression_delta"]:+.4f}**. Lý do máy chấm: {reason}

Cổng đã định trước yêu cầu target cao hơn baseline prompt tối ưu, đồng thời regression
không giảm quá 0.02. Hai điều kiện được xét đồng thời, vì mô hình học xuất JSON tốt hơn
có thể vẫn mất khả năng trả lời những câu ngoài miền. Kết quả phải được đọc với baseline
(b), thay vì chỉ nhìn bước nhảy từ baseline ngây thơ (a); prompt dài tự nó đã cung cấp
schema, nhãn hợp lệ và ví dụ. Một thắng lợi so với (a) không chứng minh chi phí huấn luyện
là cần thiết. Mặt khác, điểm regression chỉ là keyword recall trên 15 câu, không phải
một đánh giá năng lực tổng quát đầy đủ. Những chênh lệch nhỏ có thể phụ thuộc dữ liệu
tổng hợp hoặc một seed. Vì thế phán quyết này là quyết định cho phép so sánh đã đăng ký,
không phải chứng nhận chất lượng production. Không nới tolerance, chỉnh nhãn hoặc thay
prompt sau khi biết kết quả. {decision}

## 6. Ví dụ định tính và ca thua

Trên toàn bộ target: **{counts["win"]} thắng, {counts["loss"]} thua,
{counts["tie"]} hoà** so với baseline (b). Chọn ca thua trước, rồi ca thắng và hoà;
không chọn chỉ theo điểm fine-tune tuyệt đối. Mọi đầu ra đầy đủ nằm trong
`results/baseline_predictions.json` và `results/qualitative.json`.
{"Có ít nhất hai ca thua thật trong các ví dụ dưới đây." if counts["loss"] >= 2 else "Không quan sát đủ hai ca thua; báo hạn chế này thay vì tạo ví dụ giả."}

{qualitative_text}

## 7. Kết luận và điều rút ra

{decision} Lý do là giá trị của fine-tuning phải được xác định bằng khả năng vượt
base đã prompt tử tế và bảo toàn năng lực ngoài tác vụ, không phải bằng việc đường loss
đi xuống. Trong thí nghiệm này, mọi run dùng chung dữ liệu, seed, mask và ngân sách step;
attention-only còn nâng rank để khớp số tham số. Điều đó loại bỏ một số giải thích thay
thế rõ ràng, nhưng chưa loại bỏ nhiễu của một seed hay hạn chế của tập kiểm tra nhỏ.
Kết quả phản ánh một mô hình 0.8B trên các khuôn ticket tổng hợp, nên không thể áp dụng
trực tiếp cho model 4B hoặc dữ liệu doanh nghiệp. Bằng chứng mask là điều kiện cần: nếu
tính loss cả prompt, các con số sau có thể đo một tác vụ khác. Căn chỉnh prompt cũng cần
được chứng minh, vì một mask đúng vẫn có thể đi cùng conditioning sai. Sau khi hai điều
đó đúng, LR và vị trí adapter mới là các biến có ý nghĩa để so sánh. Tiết kiệm VRAM của
QLoRA phải đi cùng điểm target và thời gian thực đo; tỷ lệ bit không tự bảo đảm tốc độ.
Nếu có thêm hai giờ, ưu tiên đánh giá ticket tự nhiên đã khử nhiễm, kiểm tra JSON schema
nghiêm ngặt và chạy thêm seed, rồi mới cân nhắc replay dữ liệu regression hay quét rank.
Không dùng tập eval hiện tại làm dữ liệu huấn luyện để chữa những lỗi vừa quan sát.

Ba điều cụ thể rút ra: (1) assistant mask của template có thể rỗng dù chat vẫn render
bình thường; kiểm tra labels thật quan trọng hơn tên cờ thư viện; (2) loss huấn luyện
và điểm target là hai đại lượng khác nhau, nên xếp hạng theo target; (3) line ending của
Windows có thể làm checksum byte thay đổi dù nội dung nhìn giống nhau, vì vậy phải giữ
đúng byte corpus thay vì ghi lại checksum mới cho vừa cổng.

**Hỗ trợ AI:** Codex đọc repository, sửa lưu bằng chứng và tính tái lập, vận hành setup,
chạy kiểm tra và tạo bản report này từ artifact. Những con số là kết quả máy đo;
nhận xét kỹ thuật không được trình bày như trải nghiệm cá nhân chưa xác nhận của học viên.
Học viên cần đọc lại các ví dụ và diễn giải trước khi nộp.

Không khai báo điểm thưởng NB6, dataset riêng, reasoning collapse, rank sweep hay
Hugging Face upload: các nội dung đó nằm ngoài core đã thực hiện.
'''
    dest = ROOT / "submission" / "REPORT.md"
    dest.write_text(report, encoding="utf-8")
    reflection = f'''# Reflection — Lab 21

Ghi chú từ kết quả đo thực tế, được soạn với hỗ trợ Codex. Học viên cần rà soát
và bổ sung trải nghiệm cá nhân; không coi nhận xét kỹ thuật là cảm xúc đã xác nhận.

**1. Kết quả đáng chú ý nhất?**

Baseline prompt tối ưu đạt target {b["target"]:.4f}; fine-tune đạt
{ft["target"]:.4f}, cổng {status}. Kết quả cần đọc cùng regression delta
{gate["regression_delta"]:+.4f}, không chỉ loss hoặc mức tăng so với naive prompt.

**2. Thời gian thực nghiệm tập trung ở đâu?**

Bốn run train mất tổng {sum(float(runs[k]["train_seconds"]) for k in keys):.1f} giây.
Run lâu nhất là {max(keys, key=lambda k: float(runs[k]["train_seconds"]))}.
Con số này chỉ gồm train; setup/tải model và generation không nằm trong train_seconds.
Log trong results/ ghi rõ phạm vi để không lẫn thời gian huấn luyện với toàn pipeline.

**3. Giả định nào cần xem lại?**

Loss thấp hơn không tự bảo đảm thắng baseline prompt tử tế. Thứ tự target và
thứ tự mean training loss được trình bày riêng trong REPORT.md. Attention-only
cũng phải nâng rank lên {a["r"]} để khớp ngân sách, thay vì so cùng rank 16.

**4. AI assistant được dùng vào việc gì?**

Codex đọc project, sửa lưu bằng chứng và tính tái lập, chạy kiểm tra, chuẩn bị
runner/report/ZIP. Các số đo lấy từ artifact; không sửa nhãn hay nới gate để
đổi phán quyết. Native assistant mask rỗng đã được phát hiện bằng tokenizer
thật; training dùng labels được kiểm chứng ở NB1.

**5. Bước đầu với khách hàng thật?**

Chốt schema, nhãn và chi phí sai từng trường; tạo eval ticket tự nhiên đã khử
nhiễm rồi đóng băng baseline prompt. Sau đó mới kiểm chứng loss mask và chọn
LoRA. Tập tổng hợp 50 target/15 regression hiện tại chưa đủ để cam kết production.
'''
    (ROOT / "submission" / "REFLECTION.md").write_text(reflection, encoding="utf-8")
    print(f"Wrote {dest}; {status}; {len(report.split())} words")


if __name__ == "__main__":
    main()
