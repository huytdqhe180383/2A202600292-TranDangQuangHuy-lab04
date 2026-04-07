from langchain_core.tools import tool

# =====================================================================
# MOCK DATA - Dữ liệu giả lập hệ thống du lịch
# Lưu ý: Giá cả có logic (VD: cuối tuần đắt hơn, hạng cao hơn đắt hơn)
# Sinh viên cần đọc hiểu data để debug test cases.
# =====================================================================

FLIGHTS_DB = {
    ("Hà Nội", "Đà Nẵng"): [
        {"airline": "Vietnam Airlines", "departure": "06:00", "arrival": "07:20", "price": 1_450_000, "class": "economy"},
        {"airline": "Vietnam Airlines", "departure": "14:00", "arrival": "15:20", "price": 2_800_000, "class": "business"},
        {"airline": "VietJet Air", "departure": "08:30", "arrival": "09:50", "price": 890_000, "class": "economy"},
        {"airline": "Bamboo Airways", "departure": "11:00", "arrival": "12:20", "price": 1_200_000, "class": "economy"},
    ],
    ("Hà Nội", "Phú Quốc"): [
        {"airline": "Vietnam Airlines", "departure": "07:00", "arrival": "09:15", "price": 2_100_000, "class": "economy"},
        {"airline": "VietJet Air", "departure": "10:00", "arrival": "12:15", "price": 1_350_000, "class": "economy"},
        {"airline": "VietJet Air", "departure": "16:00", "arrival": "18:15", "price": 1_100_000, "class": "economy"},
    ],
    ("Hà Nội", "Hồ Chí Minh"): [
        {"airline": "Vietnam Airlines", "departure": "06:00", "arrival": "08:10", "price": 1_600_000, "class": "economy"},
        {"airline": "VietJet Air", "departure": "07:30", "arrival": "09:40", "price": 950_000, "class": "economy"},
        {"airline": "Bamboo Airways", "departure": "12:00", "arrival": "14:10", "price": 1_300_000, "class": "economy"},
        {"airline": "Vietnam Airlines", "departure": "18:00", "arrival": "20:10", "price": 3_200_000, "class": "business"},
    ],
    ("Hồ Chí Minh", "Đà Nẵng"): [
        {"airline": "Vietnam Airlines", "departure": "09:00", "arrival": "10:20", "price": 1_300_000, "class": "economy"},
        {"airline": "VietJet Air", "departure": "13:00", "arrival": "14:20", "price": 780_000, "class": "economy"},
    ],
    ("Hồ Chí Minh", "Phú Quốc"): [
        {"airline": "Vietnam Airlines", "departure": "08:00", "arrival": "09:00", "price": 1_100_000, "class": "economy"},
        {"airline": "VietJet Air", "departure": "15:00", "arrival": "16:00", "price": 650_000, "class": "economy"},
    ],
}

HOTELS_DB = {
    "Đà Nẵng": [
        {"name": "Mường Thanh Luxury", "stars": 5, "price_per_night": 1_800_000, "area": "Mỹ Khê", "rating": 4.5},
        {"name": "Sala Danang Beach", "stars": 4, "price_per_night": 1_200_000, "area": "Mỹ Khê", "rating": 4.3},
        {"name": "Fivitel Danang", "stars": 3, "price_per_night": 650_000, "area": "Sơn Trà", "rating": 4.1},
        {"name": "Memory Hostel", "stars": 2, "price_per_night": 250_000, "area": "Hải Châu", "rating": 4.6},
        {"name": "Christina's Homestay", "stars": 2, "price_per_night": 350_000, "area": "An Thượng", "rating": 4.7},
    ],
    "Phú Quốc": [
        {"name": "Vinpearl Resort", "stars": 5, "price_per_night": 3_500_000, "area": "Bãi Dài", "rating": 4.4},
        {"name": "Sol by Meliá", "stars": 4, "price_per_night": 1_500_000, "area": "Bãi Trường", "rating": 4.2},
        {"name": "Lahana Resort", "stars": 3, "price_per_night": 800_000, "area": "Dương Đông", "rating": 4.0},
        {"name": "9Station Hostel", "stars": 2, "price_per_night": 200_000, "area": "Dương Đông", "rating": 4.5},
    ],
    "Hồ Chí Minh": [
        {"name": "Rex Hotel", "stars": 5, "price_per_night": 2_800_000, "area": "Quận 1", "rating": 4.3},
        {"name": "Liberty Central", "stars": 4, "price_per_night": 1_400_000, "area": "Quận 1", "rating": 4.1},
        {"name": "Cochin Zen Hotel", "stars": 3, "price_per_night": 550_000, "area": "Quận 3", "rating": 4.4},
        {"name": "The Common Room", "stars": 2, "price_per_night": 180_000, "area": "Quận 1", "rating": 4.6},
    ]
}

@tool
def search_flights(origin: str, destination: str) -> str:
    """
    Tìm kiếm các chuyến bay giữa hai thành phố.
    Tham số:
    - origin: thành phố khởi hành (VD: 'Hà Nội', 'Hồ Chí Minh')
    - destination: thành phố đến (VD: 'Đà Nẵng', 'Phú Quốc')
    Trả về danh sách chuyến bay với hãng, giờ bay, giá vé.
    Nếu không tìm thấy chuyến bay, trả về thông báo không có chuyến.
    """
    key = (origin, destination)
    reverse_key = (destination, origin)
    
    flights = FLIGHTS_DB.get(key)
    
    if not flights:
        if reverse_key in FLIGHTS_DB:
            return f"Không tìm thấy chuyến bay từ {origin} đến {destination} nhưng có chuyến bay theo chiều ngược lại."
        return f"Không tìm thấy chuyến bay từ {origin} đến {destination}."
        
    result = []
    for f in flights:
        price_str = f"{f['price']:,}".replace(",", ".")
        result.append(f"- Hãng: {f['airline']}, Khởi hành: {f['departure']}, Đến: {f['arrival']}, Giá vé: {price_str}đ, Hạng: {f['class']}")
    
    return "\n".join(result)

@tool
def search_hotels(city: str, max_price_per_night: int = 99999999) -> str:
    """
    Tìm kiếm khách sạn tại một thành phố, có thể lọc theo giá tối đa mỗi đêm.
    Tham số:
    - city: tên thành phố (VD: 'Đà Nẵng', 'Phú Quốc', 'Hồ Chí Minh')
    - max_price_per_night: giá tối đa mỗi đêm (VNĐ), mặc định không giới hạn
    Trả về danh sách khách sạn phù hợp với tên, số sao, giá, khu vực, rating.
    """
    hotels = HOTELS_DB.get(city)
    
    if not hotels:
        return f"Không tìm thấy thông tin khách sạn tại {city}."
        
    filtered = [h for h in hotels if h['price_per_night'] <= max_price_per_night]
    
    if not filtered:
        price_str = f"{max_price_per_night:,}".replace(",", ".")
        return f"Không tìm thấy khách sạn tại {city} với giá dưới {price_str}đ/đêm. Hãy thử tăng ngân sách."
        
    filtered.sort(key=lambda x: x['rating'], reverse=True)
    
    result = []
    for h in filtered:
        price_str = f"{h['price_per_night']:,}".replace(",", ".")
        result.append(f"- Tên: {h['name']}, {h['stars']} sao, Giá: {price_str}đ/đêm, Khu vực: {h['area']}, Rating: {h['rating']}")
        
    return "\n".join(result)

@tool
def calculate_budget(total_budget: int, expenses: str) -> str:
    """
    Tính toán ngân sách còn lại sau khi trừ các khoản chi phí.
    Tham số:
    - total_budget: tổng ngân sách ban đầu (VNĐ)
    - expenses: chuỗi mô tả các khoản chi, mỗi khoản cách nhau bởi dấu phẩy, 
      định dạng 'tên_khoản:số_tiền' (VD: 'vé_máy_bay:890000,khách_sạn:650000')
    Trả về bảng chi tiết các khoản chi và số tiền còn lại.
    Nếu vượt ngân sách, cảnh báo rõ ràng số tiền thiếu.
    """
    try:
        expense_dict = {}
        for item in expenses.split(','):
            item = item.strip()
            if not item: continue
            name, amount = item.split(':')
            expense_dict[name.strip()] = int(amount.strip())
            
        total_expenses = sum(expense_dict.values())
        remaining = total_budget - total_expenses
        
        result = ["Bảng chi phí:"]
        for name, amount in expense_dict.items():
            formatted_name = name.replace('_', ' ').capitalize()
            # Special cases from the example
            if formatted_name == "Vé máy bay": formatted_name = "Vé máy bay"
            elif formatted_name == "Khách sạn": formatted_name = "Khách sạn"
                
            amount_str = f"{amount:,}".replace(",", ".")
            result.append(f"- {formatted_name}: {amount_str}đ")
            
        result.append("---")
        result.append(f"Tổng chi: {f'{total_expenses:,}'.replace(',', '.')}đ")
        result.append(f"Ngân sách: {f'{total_budget:,}'.replace(',', '.')}đ")
        result.append(f"Còn lại: {f'{remaining:,}'.replace(',', '.')}đ")
        
        if remaining < 0:
            result.append(f"Vượt ngân sách {f'{-remaining:,}'.replace(',', '.')} đồng! Cần điều chỉnh.")
            
        return "\n".join(result)
    except Exception as e:
        return "Lỗi xử lý định dạng expenses. Vui lòng kiểm tra lại định dạng (VD: 'vé_máy_bay:890000,khách_sạn:650000')"
