# Kế hoạch thí nghiệm trước huấn luyện

**Học viên:** Đàm Việt Hưng — **MSSV:** 2A202602600 — **Ngày:** 07/10/2026.

## Lựa chọn và phạm vi

Chọn `Qwen/Qwen3.5-0.8B` thay model 4B mặc định vì máy thực tế dùng NVIDIA
GeForce GTX 1660 Ti, VRAM 6 GB. Giữ recipe `LAPTOP`: batch 1, gradient accumulation
8, batch hiệu dụng 8, hai epoch, seed 42. Model nhỏ cho phép thực hiện đủ bốn run
trên cùng thiết bị; kết quả chỉ áp dụng cho model và corpus đã chọn. Precision do
thiết bị quyết định là fp16; GPU Turing không có bf16 native.

Giữ nguyên corpus tổng hợp tiếng Việt của lab: 250 ticket huấn luyện, 50 ticket
target, 15 câu regression. Split huấn luyện/validation là 225/25, seed 42.
Không thay corpus riêng, không sử dụng `holdout_secret.jsonl` để chọn cấu hình.
Kiểm tra đúng chuỗi ticket cho thấy không có duplicate trong train và không có
ticket trùng hoàn toàn giữa train và target. Điều này không chứng minh không có
rò rỉ ngữ nghĩa: các tập dùng cùng bộ khuôn tổng hợp, nên khả năng khái quát ra
ticket khách hàng thật vẫn cần kiểm tra riêng.

## Bằng chứng tiền kiểm đã đo

NB1 trên tokenizer thật ghi nhận `n_supervised=37`, `n_total=94`, tỷ lệ supervised
`0.3936`; cả `answer_is_supervised` và `question_is_masked` đều true.
Template giữ nội dung reasoning khi assistant có `<think>`. Corpus hiện tại chỉ
có JSON, không có reasoning trace; không dùng kết quả này để tuyên bố bảo toàn
suy luận sau fine-tune.

Độ dài đo trên 250 mẫu: mean 93.1, p50 93, p95 98, p99 100, max 101 token.
Hàm của lab gợi ý `max_length=256`. Giữ trần 1024 của tier để recipe phần cứng
không đổi giữa các run. Dữ liệu token hoá không pad cố định lên 1024; với batch 1,
trần này không tạo padding 1024 token, và không mẫu nào bị cắt ở trần đó.

`check_mask_agreement.py` phát hiện template không có `{% generation %}`, nên
mask assistant do tokenizer trả về có 0/31 token; mask của lab có 9/31 token và
chứa JSON + EOS. Vì vậy train dùng `input_ids`, `attention_mask`, `labels` đã
token hoá và kiểm chứng ở NB1, không bật `assistant_only_loss` của thư viện.

Trên Windows, Git đã chuyển JSONL sang CRLF. Chỉ khôi phục LF sau khi xác nhận
hash LF khớp hoàn toàn `data/checksums.json`; không sửa nội dung hay nhãn. File
`.gitattributes` giữ LF cho lần checkout sau. Không thay checksum tham chiếu.

## Quy tắc so sánh đã định trước

1. Chạy NB2 trước NB3/NB4; đóng băng cả score, prompt SHA, thời điểm UTC và SHA256
   của hai tập eval. Không sửa prompt tối ưu hay tập eval sau huấn luyện.
2. Dùng toàn bộ 50 target và 15 regression; không đặt `EVAL_LIMIT`.
3. `correct`: text-linear, r=16, alpha=32, LR=0.0001, trọng số base fp16.
4. `attn_only`: q/v, rank tính bằng `matched_rank`, alpha=2r; ngân sách tham số
   phải lệch dưới 5%; các cấu hình còn lại như `correct`.
5. `wrong_lr`: chỉ giảm LR xuống 0.00001.
6. `qlora`: chỉ chuyển trọng số base sang NF4 4-bit; compute vẫn theo thiết bị.
7. Mọi run dùng cùng số step tính từ 225 mẫu, batch hiệu dụng 8 và hai epoch:
   `ceil(225/8) * 2 = 58` step. Kiểm tra số step thực tế trong log/trainer state
   và bảng `results/runs.csv` trước khi kết luận.
8. Sinh greedy, tối đa 160 token cho target, 96 token cho regression, batch 4.
   Baseline (a) và fine-tune dùng prompt ngắn; baseline (b) dùng nguyên prompt tối
   ưu của lab. Dùng cùng scorer cho tất cả run.

## Điều kiện phán quyết

Cổng yêu cầu target fine-tune **cao hơn** baseline (b), đồng thời regression
không giảm quá 0.02. Target là độ chính xác trung bình bốn trường, regression là
keyword recall, format là tỷ lệ khoá bắt buộc có trong JSON trích xuất được.
Format scorer này khá rộng: không đồng nghĩa mọi đầu ra là JSON thuần đúng hợp
đồng. Latency là thời gian sinh trung bình mỗi mẫu trong batch, không phải độ trễ
yêu cầu đơn lẻ; không dùng nó để cam kết SLA.

Xếp hạng bốn run bằng target, đối chiếu loss huấn luyện như chỉ số phụ. Lưu nguyên
đầu ra baseline và fine-tune để chọn ít nhất năm ví dụ, bao gồm ít nhất hai trường
hợp fine-tune thực sự thua baseline (b), nếu các trường hợp đó tồn tại. Không tạo
ca thua giả nếu quan sát không có đủ; báo số lượng thực tế và hạn chế tương ứng.

PASS hoặc FAIL đều là kết quả hợp lệ. Không nới cổng, đổi tập eval hoặc làm yếu
prompt (b) để tạo ra chiến thắng. Quyết định deploy chỉ được viết sau khi có đủ
số đo trong `results/`.
