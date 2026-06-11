import json
import re
import os

def determine_options(query: str, answer: str) -> list[str]:
    # 1. Quét các lựa chọn MCQ bắt đầu ở đầu dòng (ví dụ: \nA. hoặc \nB))
    found_options = re.findall(r"(?:\n|^)\s*([A-F])[\.\)]\s", query)
    
    if found_options:
        options_list = sorted(list(set(found_options)))
        # Luôn luôn thêm "Uncertain" vào cuối danh sách lựa chọn MCQ
        options_list.append("Uncertain")
        return options_list
        
    # 2. Dạng câu hỏi Đúng/Sai/Chưa xác định (Yes/No/Uncertain)
    if answer in ["Yes", "No", "Uncertain"]:
        return ["Yes", "No", "Uncertain"]
        
    # 3. Dạng câu hỏi tự do (Free-form) điền số/chuỗi ngắn
    return []

def main():
    input_file = r"d:\Education\exact_2026\data\processed\Logic_SFT.json"
    output_file = r"d:\Education\exact_2026\data\processed\Logic_SFT_with_options.json"
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        return
        
    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    print("Processing options field for each question...")
    updated_records = []
    for idx, r in enumerate(records):
        queries = r.get("question", [])
        answers = r.get("answer", [])
        
        options_for_record = []
        for q, ans in zip(queries, answers):
            opts = determine_options(q, ans)
            options_for_record.append(opts)
            
        r["options"] = options_for_record
        updated_records.append(r)
        
    # Ghi ra file mới để an toàn, tránh ghi đè trực tiếp lên file gốc của bạn
    print(f"Saving processed data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(updated_records, f, ensure_ascii=False, indent=2)
        
    print("Success! You can now check the new file.")
    print("If you are satisfied, you can replace the original file with the new one.")

if __name__ == "__main__":
    main()
