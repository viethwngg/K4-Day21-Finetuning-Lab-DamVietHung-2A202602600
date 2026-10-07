# Reflection — Lab 21

Ghi chú kỹ thuật từ công việc đã làm với Codex. Training chưa chạy; học viên
cần bổ sung trải nghiệm cá nhân sau khi hoàn thành thí nghiệm.

**1. Điều đáng chú ý nhất?**

Template render bình thường nhưng mask assistant tự sinh có 0 token. Mask của
lab trên cùng ví dụ có 9/31 token. Tên cờ không bảo đảm câu trả lời vào loss;
cần giải mã labels thật.

**2. Thời gian tập trung ở đâu?**

Setup CUDA PyTorch và tải model là nút thắt hiện tại: timeout, DNS và tải dở.
NB1 đã hoàn thành mà chưa tải được trọng số. Chưa có dữ liệu so thời gian setup
với thời gian train/evaluate của cả lab.

**3. Giả định nào cần xem lại?**

Loss giảm không chứng minh fine-tune tốt. Cần baseline prompt tử tế, regression
và ca thua thật. So hai vị trí ở cùng rank cũng không mặc nhiên là ngân sách
công bằng; phải tính số tham số.

**4. AI assistant được dùng vào việc gì?**

Codex đọc project/rubric, kiểm tra phần cứng, setup, chạy mask proof và tests,
sửa predictions/checksums/step logs, chuẩn bị Colab runner và dựng report/ZIP.
Cách tải CUDA thông thường không hoàn tất do kết nối; range download có tiến
triển nhưng cũng gặp DNS. Không điền target, VRAM hoặc verdict khi chưa đo.

**5. Bước đầu với khách hàng thật?**

Xác định schema, nhãn và chi phí sai từng trường; tạo tập eval tự nhiên đã
khử nhiễm, đóng băng nó và đo baseline prompt trước train. Sau đó chứng minh
mask trên tokenizer thật và đặt giới hạn regression trước khi chọn LoRA.
