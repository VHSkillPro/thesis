# Khoá luận tốt nghiệp

## Tên đề tài:
Nghiên cứu và ứng dụng kỹ thuật nhận dạng khuôn mặt kết hợp chưng cất tri thức trong hệ thống điểm danh và xác thực tự động

## Sinh viên thực hiện: 
- Mã sinh viên: 21T1020340
- Họ tên: Ngô Văn Hải
- Lớp: Công nghệ thông tin K45F
- Chuyên ngành: Khoa học máy tính
- Giảng viên hướng dẫn: TS. Lê Quang Chiến

## Tóm tắt:
Trong bài báo này, chúng tôi trình bày một hệ thống điểm danh tự động dựa trên nhận dạng khuôn mặt, đặc biệt được thiết kế tối ưu cho thiết bị di động. Hệ thống sử dụng mô hình phát hiện khuôn mặt YuNet và mô hình nhận dạng khuôn mặt SFace. Để cân bằng giữa hiệu suất và độ chính xác, chúng tôi áp dụng kỹ thuật Knowledge Distillation với hai teacher models (ArcFace và SFace OpenCV) và fine-tuning trên dữ liệu thực tế. Kết quả cho thấy Equal Error Rate (EER) giảm còn 9.613% (từ 11.067%), Accuracy và F1-score được cải thiện. Hệ thống đạt tốc độ suy luận ~25ms và với dung lượng <5MB, phù hợp cho các ứng dụng di động. 

## Abstract:
In this paper, we present an automated attendance system based on face recognition, specifically optimized for mobile devices. The system employs YuNet for face detection and SFace for face recognition. To balance between performance and accuracy, we apply Knowledge Distillation using two teacher models (ArcFace and SFace-OpenCV), followed by fine-tuning on real-world data. Experimental results show that the Equal Error Rate (EER) is reduced to 9.613% (from 11.067%), with improvements observed in both Accuracy and F1-score. The system achieves an inference speed of approximately 25 ms and has a model size of less than 5MB, making it suitable for mobile applications.

