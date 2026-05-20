# EXACT 2026 - Tổng hợp Thông tin và Yêu cầu Cuộc thi

Tài liệu này tổng hợp toàn bộ thông tin quan trọng từ buổi Kick-off Workshop và văn bản Q&A chính thức của cuộc thi **EXACT 2026** (The 2nd International XAΙ Challenge for Transparent Educational Question-Answering).

---

## 1. Thông tin Chung
* **Tên cuộc thi:** EXACT 2026, thuộc khuôn khổ hội nghị IEEE IJCNN 2026 và WCCI 2026 [cite: 5, 6, 85, 87].
* **Đơn vị tổ chức:** Đại học Bách khoa TP.HCM (HCMUT), VNU-HCM và Đại học Naples Parthenope, Ý [cite: 82, 83].
* **Mục tiêu:** Xây dựng hệ thống Hỏi-Đáp minh bạch trong giáo dục, tập trung vào khả năng giải thích (Explainable AI) [cite: 80].

## 2. Lịch trình Quan trọng (2026)
| Ngày | Cột mốc |
| :--- | :--- |
| 10/04 - 10/05 | Đăng ký đội thi [cite: 142]. |
| 09/05 | Kick-off Workshop & Công bố dữ liệu [cite: 143, 698]. |
| 10/05 - 30/05 | Giai đoạn thi đấu chính [cite: 144]. |
| 01/06 - 02/06 | Đánh giá & Phản hồi Giai đoạn 1 [cite: 145]. |
| 03/06 - 04/06 | Giai đoạn tinh chỉnh mô hình [cite: 146]. |
| 05/06 - 07/06 | Đánh giá Giai đoạn 2 [cite: 147]. |
| 10/06 | Công bố Top 10 [cite: 148]. |
| 15/06 | Public Test Day (Demo trực tiếp trước ban giám khảo) [cite: 149]. |
| 30/06 - 15/07 | Nộp bài báo (Dành cho Top 10) [cite: 150]. |
| 16/11 - 18/11 | Thuyết trình tại CSoNet 2026 [cite: 151]. |

## 3. Quy định về Mô hình & Kỹ thuật
### 3.1. Giới hạn Tham số (Rule of Thumb)
* **Mô hình 8B:** Chỉ chấp nhận LLM mã nguồn mở thuộc nhóm "8B-class" theo tên gọi chính thức (ví dụ: Qwen3-8B với 8.19B thực tế vẫn được chấp nhận) [cite: 576, 578].
* **Mixture-of-Experts (MoE):** Giới hạn 8B áp dụng cho **tổng số tham số**, không phải tham số hoạt động (active parameters). Do đó, các mô hình như Qwen3-30B-A3B không hợp lệ [cite: 581, 582, 585].
* **Cấm mô hình đóng:** Không được sử dụng GPT, Claude, Gemini hoặc các API dịch vụ bên thứ ba (Groq, Together AI, v.v.) khi suy luận [cite: 183, 607, 617].

### 3.2. Cấu trúc Hệ thống (Pipeline)
* **Sử dụng nhiều mô hình:** Có thể dùng các LLM khác nhau cho từng tác vụ (ví dụ: một cái cho Type 1, một cái cho Type 2) nhưng phải chạy **nối tiếp** (sequential). Tại một thời điểm suy luận, chỉ được phép nạp và chạy duy nhất một LLM <= 8B [cite: 589, 591, 594].
* **Công cụ hỗ trợ:** Khuyến khích sử dụng công cụ bên ngoài như Code execution (Python, SymPy), Solver (Z3, Prover9), RAG và tìm kiếm internet [cite: 179, 627, 628, 638, 641].
* **Hosting:** Phải tự host mô hình (Self-host) trên hạ tầng riêng (GPU cá nhân, Cloud VM, Kaggle, Colab) và triển khai qua **vLLM** (hoặc framework tương thích OpenAI API) để BTC kiểm tra qua endpoint `/v1/models` [cite: 610, 611, 615, 678, 679].

## 4. Dữ liệu Cuộc thi
Hệ thống phải xử lý một luồng truy vấn hợp nhất bao gồm hai loại [cite: 503, 711]:

### Type 1: Logic-Based Educational Queries
* **Nội dung:** Quy chế học vụ (điểm số, đăng ký, học bổng) [cite: 254].
* **Yêu cầu:** Giải thích dựa trên Logic bậc nhất (FOL) và trích dẫn điều khoản cụ thể [cite: 254, 258].
* **Quy mô:** 411 bản ghi, 808 câu hỏi [cite: 268].

### Type 2: Physics Problems
* **Nội dung:** Bài toán vật lý về mạch điện và tĩnh điện [cite: 327].
* **Yêu cầu:** Suy luận Chain-of-Thought từng bước, tính toán chính xác và đi kèm đơn vị SI [cite: 327, 332].
* **Lưu ý:** Cần tự lọc bỏ 401 mẫu có mã ID bắt đầu bằng "QA" (đây là lỗi chú thích) [cite: 721, 724, 725]. Tập dữ liệu hợp lệ còn lại là 1.354 bài toán [cite: 726].

## 5. Tiêu chí Đánh giá
Điểm số cuối cùng là tổng trọng số của 3 tiêu chí [cite: 734]:
1.  **P1 - Correctness:** Độ chính xác của đáp án (so khớp tự động) [cite: 196, 735].
2.  **P2 - Explanation Quality:** Chất lượng câu giải thích bằng ngôn ngữ tự nhiên (Ban giám khảo đánh giá độ rõ ràng, trung thực) [cite: 196, 736].
3.  **P3 - Reasoning Depth:** Điểm cộng cho các bằng chứng suy luận có thể kiểm chứng (FOL derivations, CoT steps, trích dẫn premises) [cite: 196, 737].

## 6. Quy định nộp bài
Thí sinh nộp gói hồ sơ qua website [https://ura.hcmut.edu.vn/exact](https://ura.hcmut.edu.vn/exact) bao gồm [cite: 749, 751]:
1.  **URL của API endpoint:** Phải hoạt động trực tuyến trong suốt các cửa sổ đánh giá [cite: 756]. Thời gian phản hồi tối đa 60 giây/yêu cầu [cite: 214, 671].
2.  **Solution Description (PDF):** Tài liệu 1 trang mô tả giải pháp, mô hình và công cụ [cite: 210, 749].
3.  **Data Disclosure Document (PDF):** Bắt buộc khai báo mọi nguồn dữ liệu ngoài, dữ liệu thu thập hoặc dữ liệu tổng hợp từ mô hình đóng dùng để huấn luyện [cite: 652, 653, 750].

## 7. Giải thưởng & Cơ hội
* **Top 5:** Giải thưởng tiền mặt + Mời thuyết trình tại CSoNet 2026 [cite: 228].
* **Top 10:** Bài báo được Springer xuất bản (LNCS/LNAI), được chỉ mục bởi Scopus, DBLP, v.v. [cite: 229].
* **Chứng chỉ:** Chứng chỉ chính thức cho mọi đội thi có bài dự thi hợp lệ [cite: 231].

---
*Lưu ý: Báo cáo các lỗi trong bộ dữ liệu trên Discord để nhận điểm thưởng [cite: 741, 742].*
