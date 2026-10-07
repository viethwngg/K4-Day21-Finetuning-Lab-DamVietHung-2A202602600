# Lab 21 — Bằng chứng và trạng thái thí nghiệm

**Học viên:** Đàm Việt Hưng — **MSSV:** 2A202602600 — **Ngày:** 07/10/2026.

**Trạng thái: CHƯA HOÀN TẤT core NB1–NB5, chưa đủ điều kiện nộp.** NB1 đã chạy
trên tokenizer thật. Baseline, adapter và verdict chưa có vì tải CUDA PyTorch
và trọng số bị timeout/DNS. Chưa đo không có nghĩa là verdict FAILED.

## 1. Thiết kế đã chọn

Model `Qwen/Qwen3.5-0.8B` được chọn vì máy có GTX 1660 Ti 6 GB VRAM và khoảng
7.8 GB RAM; model 4B mặc định không phù hợp. Recipe LAPTOP giữ batch 1,
gradient accumulation 8, batch hiệu dụng 8, hai epoch và seed 42. Khi torch
CUDA cài xong sẽ dùng fp16 với gradient scaling. Hiện environment.json ghi
CPU/fp32 vì chưa có torch, không phải bằng chứng đã train.

Giữ corpus tổng hợp mặc định: 250 ticket CSKH tiếng Việt → JSON bốn trường,
split 225 train / 25 validation; eval có 50 target và 15 regression. Chọn bài
này vì nhãn cho phép chấm khách quan, không cần LLM judge. Không có ticket
trùng hoàn toàn trong train hoặc giữa train và target; vẫn có giới hạn do
các tập sử dụng cùng khuôn tổng hợp. Không dùng holdout để chọn cấu hình.

Kế hoạch trước train và quy tắc phán quyết được ghi trong
[EXPERIMENT.md](EXPERIMENT.md). Không sửa prompt (b), không dùng EVAL_LIMIT,
không nới cổng. Bốn run dự kiến cùng ngân sách 58 step; cần kiểm tra actual_steps.

## 2. Mask proof và template đã chạy

Nguồn: results/mask_proof.json.

| Kiểm tra | Kết quả |
|---|---|
| n_supervised / n_total | 37 / 94 |
| supervised_fraction | 0.3936 |
| answer_is_supervised | true |
| question_is_masked | true |
| MASK_MODE | assistant-only |

Phần thực sự được tính loss:

```text
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đối chứng everything tính loss 94/94 token, bao gồm câu hỏi. Không train với
chế độ đó. results/prompt_alignment.json xác nhận evaluation prompt là prefix
của training prompt trên tokenizer thật.

Template check kết luận reasoning preserved: khối think được giữ khi assistant
có trace. Corpus chỉ có JSON; chưa có căn cứ nói model giữ hay mất suy luận
sau fine-tune. check_mask_agreement.py phát hiện template không có generation
marker: assistant mask tự sinh là 0/31 token, mask của lab là 9/31 token và
chứa câu trả lời + EOS. Pipeline dùng labels pre-tokenized đã chứng minh ở
NB1, không dựa vào assistant_only_loss của template.

## 3. Độ dài và tính tái lập

Nguồn: results/token_stats.json.

| n | mean | p50 | p95 | p99 | max | suggested_max_length |
|---|---|---|---|---|---|---|
| 250 | 93.1 | 93 | 98 | 100 | 101 | 256 |

Giữ trần 1024 của tier cho cùng recipe phần cứng. Dữ liệu pre-tokenized không
pad cố định lên trần, batch 1 không có padding giữa mẫu; không mẫu nào bị cắt.
Windows checkout đã đổi JSONL sang CRLF, khiến checksum byte sai. Chỉ khôi
phục LF sau khi xác nhận hash LF khớp từng checksum gốc; không sửa nội dung,
nhãn hay checksum tham chiếu. .gitattributes giữ LF cho checkout sau.

NB2 được sửa để lưu nguyên predictions, thời điểm đóng băng và checksum eval.
NB5 kiểm tra model, prompt, số mẫu, checksum và lưu điểm thắng/thua/hoà từng
mẫu cùng hai đầu ra đầy đủ. NB3/NB4 lưu actual_steps và loss logs. Test suite:
**123 passed, 3 skipped**; các test skip cần torch đang thiếu. Smoke pass không
có nghĩa full submission pass.

## 4. Kết quả còn thiếu

| Giai đoạn | Trạng thái |
|---|---|
| NB1 | Hoàn thành mask proof, template check, token stats, split |
| NB2 | Chưa đo baselines_frozen và baseline_predictions |
| NB3 | Chưa train adapters/correct và tạo runs.csv |
| NB4 | Chưa train attn_only, wrong_lr, qlora |
| NB5 | Chưa đo verdict, autopsy và qualitative |

Chưa xếp hạng adapter, chưa tính chênh lệch VRAM hoặc loss curve, chưa có
năm ví dụ baseline/FT. Các số đo này không được thay bằng kỳ vọng lý thuyết.

## 5. Kết luận tại thời điểm hiện tại

Chưa deploy fine-tune vì chưa có adapter và phép so sánh thực nghiệm. Mask proof
chỉ xác nhận labels của mẫu được xây đúng; nó không chứng minh fine-tuning đem
lại giá trị. Trước khi kết luận, cần đo base với prompt ngây thơ và prompt tối
ưu trên toàn bộ eval, đóng băng các đầu ra rồi mới train. Ba đối chứng phải
chia sẻ step budget và dữ liệu, còn attention-only phải khớp số tham số. Nếu
bỏ các điều kiện đó, khác biệt điểm có thể do ngân sách hoặc phép đo thay đổi
thay vì vị trí adapter, LR hay lượng tử hoá. Cổng cuối phải đồng thời xét khả
năng vượt baseline prompt tử tế và mức giảm regression; đường loss đẹp không
thay thế được hai kiểm tra này. Ví dụ định tính phải là ca thắng và ca thua
thật so với baseline, không phải chỉ đầu ra fine-tune có điểm thấp. Dữ liệu
tổng hợp và một seed cũng giới hạn khả năng suy rộng ngay cả khi cổng PASS.
Bước tiếp theo là hoàn thành môi trường chạy hoặc chuyển sang Colab, thực
hiện NB2–NB5 đúng thứ tự và thay báo cáo tiến độ này bằng số đo thực tế.

**Hỗ trợ AI:** Codex đọc repository, sửa tính tái lập/lưu bằng chứng, chạy NB1
và tests, chuẩn bị runner, report builder và packager. Không tạo số GPU giả.
Phản tư kỹ thuật trong REFLECTION.md cần học viên rà soát trải nghiệm cá nhân.

## 6. Tiếp tục

Lệnh Windows nằm trong README. Sau pipeline, chạy record_environment.py,
build_submission_report.py và package_submission.py. Builder từ chối report
cuối nếu thiếu kết quả hoặc step không khớp; packager yêu cầu full verify pass.

Fallback: mở colab/Lab21_SUBMISSION_RUN.ipynb trong Colab, chọn T4 GPU, upload
lab21_source_2A202602600.zip ở thư mục cha rồi chạy tuần tự. Runner sử dụng
đúng source đã sửa và tạo ZIP bài nộp khi core hoàn tất.
