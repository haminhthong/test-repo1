"""Script khởi tạo bộ dữ liệu tài liệu nội bộ doanh nghiệp mẫu (Enterprise Multi-Department Corpus).

Bao gồm hơn 15 tài liệu thuộc 3 khối:
- Nhân sự (HR): Nghỉ phép 2025, Nghỉ phép 2026 (Hard Negative theo năm), Nghỉ ốm, Làm việc từ xa, Thử việc.
- Tài chính (Finance): Công tác phí, Hoàn ứng văn phòng phẩm, Mua sắm trang thiết bị, Thẻ tín dụng doanh nghiệp, Hướng dẫn hóa đơn VAT.
- Bảo mật (Security): Mật khẩu, Xác thực 2FA/Tài khoản, Ứng cứu sự cố, Bảo mật thiết bị, Phân loại dữ liệu.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Đảm bảo Windows console in đúng UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (OSError, ValueError):
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("download_data")

SAMPLES: dict[str, str] = {
    # --------------------------------------------------------------------------
    # KHỐI NHÂN SỰ (HR POLICIES)
    # --------------------------------------------------------------------------
    "policy_leave.txt": (
        "CHÍNH SÁCH NGHỈ PHÉP NỘI BỘ DOANH NGHIỆP\n\n"
        "1. Quyền lợi nghỉ phép:\n"
        "Nhân viên toàn thời gian chính thức có 12 ngày phép năm hưởng nguyên lương.\n"
        "Nhân viên có thâm niên từ 5 năm trở lên được cộng thêm 1 ngày phép cho mỗi năm tiếp theo.\n\n"
        "2. Quy định đăng ký:\n"
        "Đơn xin nghỉ phép từ 3 ngày liên tiếp trở lên cần gửi đăng ký trước tối thiểu 5 ngày làm việc.\n"
        "Trường hợp khẩn cấp (ốm đau, việc gia đình đột xuất), nhân viên có thể báo trực tiếp cho quản lý "
        "và bổ sung đơn nghỉ trên hệ thống trong vòng 24 giờ sau khi quay lại làm việc."
    ),
    "annual_leave_policy_2025.txt": (
        "CHÍNH SÁCH PHÉP NĂM ÁP DỤNG NĂM 2025 (HẾT HIỆU LỰC)\n\n"
        "1. Tiêu chuẩn phép năm 2025:\n"
        "Trong năm 2025, nhân viên chính thức được cấp 10 ngày nghỉ phép năm có lương.\n"
        "Thời gian thử việc là 2 tháng và không được tính phép năm trong thời gian thử việc.\n\n"
        "2. Chuyển tiếp phép năm 2025:\n"
        "Phép năm 2025 chưa sử dụng hết chỉ được chuyển tối đa 3 ngày sang quý 1 năm sau."
    ),
    "annual_leave_policy_2026.txt": (
        "CHÍNH SÁCH PHÉP NĂM ÁP DỤNG TỪ NĂM 2026 (HIỆU LỰC HIỆN HÀNH)\n\n"
        "1. Tiêu chuẩn phép năm 2026:\n"
        "Từ ngày 01/01/2026, nhân viên chính thức được nâng tiêu chuẩn lên 12 ngày phép năm hưởng nguyên lương.\n"
        "Thời gian thử việc rút ngắn còn 1 tháng đối với vị trí kỹ thuật.\n\n"
        "2. Chuyển tiếp phép năm 2026:\n"
        "Toàn bộ ngày phép năm 2026 còn tồn đọng được phép chuyển tối đa 5 ngày sang năm tiếp theo, "
        "thời hạn sử dụng đến hết ngày 31/03/2027."
    ),
    "sick_leave_policy.txt": (
        "QUY ĐỊNH NGHỈ ỐM ĐAU VÀ CHẾ ĐỘ BẢO HIỂM XÃ HỘI\n\n"
        "1. Điều kiện hưởng trợ cấp nghỉ ốm:\n"
        "Nhân viên nghỉ ốm từ 1 ngày trở lên bắt buộc phải nộp Giấy chứng nhận nghỉ việc hưởng BHXH (Mẫu C65-HD) "
        "hoặc giấy ra viện hợp lệ từ cơ sở y tế có thẩm quyền.\n\n"
        "2. Quyền lợi lương:\n"
        "Bảo hiểm xã hội chi trả 75% mức tiền lương đóng BHXH của tháng liền kề trước khi nghỉ.\n"
        "Công ty hỗ trợ thêm 25% lương cơ bản trong 3 ngày ốm đầu tiên của mỗi năm dương lịch."
    ),
    "remote_work_policy.txt": (
        "QUY CHẾ LÀM VIỆC TỪ XA VÀ LINH HOẠT (REMOTE WORK)\n\n"
        "1. Hạn mức làm việc từ xa:\n"
        "Nhân viên sau khi kết thúc thử việc được đăng ký tối đa 2 ngày làm việc từ xa mỗi tuần.\n"
        "Cần đăng ký trên hệ thống nhân sự trước 17:00 của ngày làm việc liền kề và được Quản lý trực tiếp duyệt.\n\n"
        "2. Tiêu chuẩn kết nối và bảo mật khi remote:\n"
        "Bắt buộc kết nối qua mạng riêng ảo công ty (Company VPN) khi truy cập tài nguyên máy chủ nội bộ.\n"
        "Phải duy trì trạng thái online trên Slack trong khung giờ hành chính từ 08:30 đến 17:30."
    ),
    "onboarding_policy.txt": (
        "QUY TRÌNH TIẾP NHẬN NHÂN VIÊN MỚI VÀ ĐÁNH GIÁ THỬ VIỆC\n\n"
        "1. Chương trình Buddy:\n"
        "Mỗi nhân viên mới sẽ được phân công một người hướng dẫn (Buddy) trong 30 ngày đầu tiên.\n"
        "Buddy chịu trách nhiệm hỗ trợ hội nhập văn hóa và bàn giao thiết bị làm việc.\n\n"
        "2. Mốc đánh giá thử việc:\n"
        "Buổi đánh giá thử việc (Probation Review) được tổ chức vào ngày làm việc thứ 25 của kỳ thử việc.\n"
        "Kết quả đánh giá Đạt sẽ kích hoạt ký hợp đồng lao động chính thức thời hạn 1 năm."
    ),
    # --------------------------------------------------------------------------
    # KHỐI TÀI CHÍNH & KẾ TOÁN (FINANCE & EXPENSE POLICIES)
    # --------------------------------------------------------------------------
    "expense_policy.txt": (
        "CHÍNH SÁCH VÀ QUY TRÌNH HOÀN ỨNG CHI PHÍ CÔNG TÁC\n\n"
        "1. Điều kiện hoàn ứng:\n"
        "Mọi chi phí công tác phát sinh chỉ được hoàn ứng khi có hóa đơn tài chính hợp lệ (hóa đơn đỏ/VAT).\n"
        "Đối với các khoản chi phí phát sinh trên 5.000.000 VND (năm triệu đồng), nhân viên cần phải "
        "có phê duyệt trước (Pre-approval) bằng văn bản hoặc email từ Quản lý trực tiếp.\n\n"
        "2. Thời hạn nộp hồ sơ:\n"
        "Hồ sơ thanh toán hoàn ứng phải được nộp cho phòng Kế toán trong vòng 10 ngày làm việc "
        "kể từ ngày kết thúc chuyến công tác."
    ),
    "travel_expense_policy.txt": (
        "QUY ĐỊNH CÔNG TÁC PHÍ VÀ ĐẶT VÉ MÁY BAY\n\n"
        "1. Hạn mức khách sạn và phụ cấp lưu trú:\n"
        "Hạn mức phòng khách sạn tại Hà Nội và TP.HCM tối đa 1.200.000 VND/đêm; các tỉnh khác tối đa 800.000 VND/đêm.\n"
        "Phụ cấp tiền ăn (Per diem) công tác nội địa là 300.000 VND/ngày không cần nộp hóa đơn lẻ.\n\n"
        "2. Đặt vé máy bay:\n"
        "Bắt buộc đặt vé máy bay hạng phổ thông (Economy Class) cho mọi chuyến bay dưới 6 giờ bay.\n"
        "Yêu cầu đặt vé trước ngày khởi hành tối thiểu 7 ngày để tối ưu chi phí."
    ),
    "reimbursement_policy.txt": (
        "QUY TRÌNH HOÀN ỨNG MUA SẮM VĂN PHÒNG PHẨM VÀ TIỆC TEAM\n\n"
        "1. Hoàn ứng chi phí gắn kết đội ngũ (Team Building):\n"
        "Ngân sách tiệc team định kỳ hàng tháng là 200.000 VND/người/tháng.\n"
        "Không được cộng dồn ngân sách của tháng trước sang tháng sau nếu không sử dụng.\n\n"
        "2. Mua sắm văn phòng phẩm khẩn cấp:\n"
        "Chỉ được hoàn ứng các vật phẩm dưới 500.000 VND; trên mức này phải yêu cầu phòng Mua sắm cấp phát."
    ),
    "procurement_policy.txt": (
        "QUY TRÌNH MUA SẮM TẬP TRUNG VÀ ĐẤU THẦU NHÀ CUNG CẤP\n\n"
        "1. Quy định chào giá cạnh tranh:\n"
        "Mọi đơn hàng thiết bị công nghệ hoặc dịch vụ phần mềm trên 20.000.000 VND bắt buộc phải "
        "có tối thiểu 3 báo giá độc lập từ 3 nhà cung cấp khác nhau.\n\n"
        "2. Thẩm quyền phê duyệt:\n"
        "Đơn hàng từ 20 đến 100 triệu do Giám đốc Khối duyệt. Trên 100 triệu bắt buộc có chữ ký của Tổng Giám đốc."
    ),
    "corporate_card_policy.txt": (
        "QUY CHẾ SỬ DỤNG THẺ TÍN DỤNG DOANH NGHIỆP (CORPORATE CARD)\n\n"
        "1. Đối tượng cấp phát:\n"
        "Thẻ tín dụng công ty chỉ được cấp cho cấp bậc Giám đốc (Director) trở lên phục vụ tiếp khách và chi phí đối ngoại.\n\n"
        "2. Hành vi nghiêm cấm:\n"
        "Nghiêm cấm tuyệt đối sử dụng thẻ tín dụng doanh nghiệp vào mục đích chi tiêu cá nhân.\n"
        "Bảng kê sao kê ngân hàng phải được hoàn tất chứng từ đối soát trước ngày 20 hàng tháng."
    ),
    "vat_invoice_guide.txt": (
        "HƯỚNG DẪN XUẤT VÀ KIỂM TRA HÓA ĐƠN ĐIỆN TỬ VAT\n\n"
        "1. Thông tin xuất hóa đơn công ty:\n"
        "Tên đơn vị: CÔNG TY CỔ PHẦN CÔNG NGHỆ TRI THỨC DOANH NGHIỆP\n"
        "Mã số thuế: 0101234567\n"
        "Địa chỉ: Tầng 18, Tòa nhà Innovation, Phường Cầu Giấy, Hà Nội.\n\n"
        "2. Định dạng hóa đơn hợp lệ:\n"
        "Kế toán chỉ chấp nhận file định dạng hóa đơn điện tử gốc kèm file XML nén tra cứu hợp lệ từ Tổng cục Thuế."
    ),
    # --------------------------------------------------------------------------
    # KHỐI BẢO MẬT & CNTT (SECURITY & IT POLICIES)
    # --------------------------------------------------------------------------
    "security_guide.txt": (
        "HƯỚNG DẪN BẢO MẬT AN THÔNG TIN\n\n"
        "1. Quản lý tài khoản và mật khẩu:\n"
        "Tuyệt đối không chia sẻ mật khẩu cá nhân hoặc tài khoản nội bộ cho bất kỳ ai.\n"
        "Bắt buộc bật xác thực hai yếu tố (2FA) cho toàn bộ tài khoản công ty.\n\n"
        "2. Quy trình xử lý sự cố an ninh mạng:\n"
        "Khi phát hiện nghi ngờ lộ thông tin hoặc tấn công mạng, nhân viên phải báo lập tức "
        "cho đội Bảo mật (Security Team) trong vòng 30 phút kể từ khi phát hiện qua email security@company.com "
        "hoặc kênh Slack #incident-report."
    ),
    "password_policy.txt": (
        "TIÊU CHUẨN ĐẶT MẬT KHẨU VÀ QUẢN LÝ KHÓA TRUY CẬP\n\n"
        "1. Độ phức tạp mật khẩu:\n"
        "Mật khẩu tài khoản phải có độ dài tối thiểu 12 ký tự, bao gồm chữ hoa, chữ thường, chữ số và ký tự đặc biệt.\n"
        "Mật khẩu tự động hết hạn sau 90 ngày; không được sử dụng lại 5 mật khẩu gần nhất.\n\n"
        "2. Lưu trữ mật khẩu:\n"
        "Nghiêm cấm ghi mật khẩu lên giấy hoặc lưu dạng plain text trong máy tính. Bắt buộc dùng 1Password nội bộ."
    ),
    "account_security.txt": (
        "QUY ĐỊNH XÁC THỰC HAI YẾU TỐ (2FA) VÀ PHÂN QUYỀN TRUY CẬP\n\n"
        "1. Tiêu chuẩn xác thực 2FA:\n"
        "Chỉ chấp nhận 2FA qua ứng dụng Authenticator (TOTP như Google Authenticator, Duo). Không dùng SMS 2FA.\n\n"
        "2. Khóa màn hình tự động:\n"
        "Máy tính xách tay của nhân viên bắt buộc phải kích hoạt chế độ tự động khóa màn hình sau tối đa 5 phút không hoạt động."
    ),
    "incident_response.txt": (
        "QUY TRÌNH ỨNG CỨU SỰ CỐ AN NINH MẠNG VÀ RÒ RỈ DỮ LIỆU\n\n"
        "1. Phân loại mức độ sự cố:\n"
        "- Mức P1 (Nghiêm trọng): Dịch vụ chính ngừng trệ hoặc lộ dữ liệu khách hàng. Thời gian phản hồi: dưới 15 phút.\n"
        "- Mức P2 (Cao): Nhiễm mã độc máy trạm nội bộ. Thời gian phản hồi: dưới 1 giờ.\n\n"
        "2. Đầu mối báo cáo khẩn cấp:\n"
        "Đường dây nóng trực ban an ninh mạng: 024-8888-9999 hoặc Slack bot @sec-ops-pager."
    ),
    "device_security.txt": (
        "CHÍNH SÁCH BẢO MẬT THIẾT BỊ ĐẦU CUỐI (LAPTOP & DI ĐỘNG)\n\n"
        "1. Mã hóa ổ đĩa bắt buộc:\n"
        "Mọi máy tính xách tay do công ty cấp phải bật mã hóa BitLocker (Windows) hoặc FileVault (macOS).\n\n"
        "2. Cổng kết nối ngoại vi:\n"
        "Cổng USB trên máy trạm bị vô hiệu hóa tính năng sao chép dữ liệu ra ngoài (USB Mass Storage Blocked) "
        "để ngăn chặn rò rỉ dữ liệu tài sản trí tuệ."
    ),
    "data_classification.txt": (
        "CHÍNH SÁCH PHÂN LOẠI DỮ LIỆU VÀ QUYỀN HẠN TRUY CẬP\n\n"
        "1. Bốn cấp độ dữ liệu:\n"
        "- Public: Thông tin công bố rộng rãi trên website.\n"
        "- Internal: Thông tin quy trình nội bộ dùng chung cho toàn thể nhân viên.\n"
        "- Confidential: Dữ liệu tài chính, tiền lương, bảng lương chỉ dành cho HR và Ban Giám đốc.\n"
        "- Restricted: Khóa mã hóa, mã nguồn cốt lõi, mật khẩu cơ sở dữ liệu production.\n\n"
        "2. Xử lý vi phạm:\n"
        "Mọi hành vi chia sẻ tài liệu cấp độ Confidential ra bên ngoài sẽ bị sa thải ngay lập tức."
    ),
}


def create_sample_data(data_dir: str = "data/raw") -> None:
    """Tạo toàn bộ các tệp tài liệu mẫu doanh nghiệp tại thư mục quy định."""
    output_path = Path(data_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for filename, text_content in SAMPLES.items():
        file_path = output_path / filename
        file_path.write_text(text_content.strip() + "\n", encoding="utf-8")
        LOGGER.info("Đã tạo tệp tài liệu: %s", file_path.name)

    print(
        f"\n[OK] Đã tạo thành công {len(SAMPLES)} tệp tài liệu doanh nghiệp phong phú tại '{output_path.resolve()}'"
    )


if __name__ == "__main__":
    create_sample_data()
