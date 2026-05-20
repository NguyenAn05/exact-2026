# EXACT 2026 — Pipeline Explained (Final Architecture)

> Tài liệu này mô tả kiến trúc **đã được quyết định cuối cùng**, bao gồm đầy đủ input/output từng bước.

---

## Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────┐
│                   5 GIAI ĐOẠN CHÍNH                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [RAW DATA]                                                 │
│      │                                                      │
│      ▼  Giai đoạn 1: Data Preparation                      │
│  [SFT DATASET]  ← chuẩn hóa FOL + sinh python_code        │
│      │                                                      │
│      ▼  Giai đoạn 2: SFT Training                         │
│  [LoRA 1 + LoRA 2 weights]                                 │
│      │                                                      │
│      ▼  Giai đoạn 3: GRPO (RL với code execution)         │
│  [LoRA 1 + LoRA 2 weights — improved]                      │
│      │                                                      │
│      ▼  Giai đoạn 4: Inference Pipeline                    │
│  [FastAPI + FOL→Z3 Converter + Python Sandbox]             │
│      │                                                      │
│      ▼  Giai đoạn 5: Deployment                           │
│  [vLLM endpoint — BTC gọi để chấm điểm]                   │
└─────────────────────────────────────────────────────────────┘
```

### Quyết định kiến trúc chính

| Quyết định | Chi tiết |
|-----------|---------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Logic FOL format | `ForAll()` style (Z3-compatible), chuẩn hóa trước khi train |
| Logic verification | Developer viết FOL→Z3 converter, không dùng LLM sinh Z3 code |
| Physics verification | Python sandbox (`subprocess.run`) |
| GRPO reward | Cả 2 loại đều dùng code execution thực sự |
| Output BTC | Strip `conclusion_fol` trước khi trả về |

---

## Giai Đoạn 1: Data Preparation

### Bước 1A — Chuẩn hóa FOL Format (Logic)

Dataset gốc có **2 format trộn lẫn nhau**:

```
Format A (43%): ∀x (WT(x) → O(x))           ← Unicode math
Format B (33%): ForAll(x, Implies(WT(x), O(x))) ← Z3-like
Ground atoms (21%): completed_core_curriculum(Sophia)
```

**Hành động:** Convert tất cả về **ForAll() style duy nhất** để model học 1 format nhất quán.

```python
# normalize_fol.py
def normalize_fol(fol_str: str) -> str:
    """Convert Unicode FOL sang ForAll() style."""
    s = fol_str.strip()

    # ∀x P(x) → ForAll(x, P(x))
    s = re.sub(r'∀(\w+)\s+([^(].*)', r'ForAll(\1, \2)', s)

    # ∀x (P(x) → Q(x)) → ForAll(x, Implies(P(x), Q(x)))
    s = re.sub(r'∀(\w+)\s+\((.+)\)', lambda m: convert_body(m), s)

    # ∃x P(x) → Exists(x, P(x))
    s = re.sub(r'∃(\w+)\s+(.+)', r'Exists(\1, \2)', s)

    # → → Implies
    s = s.replace(' → ', ', ').replace('→', ', ')

    # ∧ → And, ∨ → Or, ¬ → Not
    # (xử lý từng nested level)
    return s
```

### Bước 1B — Sinh Training Data Type 1 (Logic)

**INPUT** — 1 record từ dataset (sau khi chuẩn hóa FOL):
```json
{
  "idx": [[4, 5, 6, 7, 1, 2, 3], [4, 5, 6, 7, 1, 2, 3]],
  "premises-FOL": [
    "ForAll(x, Implies(And(CC(x), PS(x)), QA(x)))",
    "ForAll(x, Implies(And(QA(x), RM(x)), EI(x)))",
    "ForAll(x, Implies(And(EI(x), CP(x)), HD(x)))",
    "CC(Sophia)",
    "PS(Sophia)",
    "RM(Sophia)",
    "CP(Sophia)"
  ],
  "premises-NL": [
    "Students who complete core curriculum and pass science are qualified for advanced courses.",
    "..."
  ],
  "questions": ["Is Sophia awarded an honors diploma?"],
  "answers": ["Yes"],
  "explanation": ["Sophia completes CC and PS → QA. QA + RM → EI. EI + CP → HD."]
}
```

**OUTPUT** — 1 training sample trong `logic_train.jsonl`:
```json
{
  "instruction": "You are a logical reasoning expert. Given premises in FOL format, answer the question. Generate FOL derivation using ForAll/Exists/Implies/And/Not/Or syntax, identify the conclusion FOL, provide chain-of-thought and explanation.\n\nPremises:\n1. ForAll(x, Implies(And(CC(x), PS(x)), QA(x)))\n2. ForAll(x, Implies(And(QA(x), RM(x)), EI(x)))\n3. ForAll(x, Implies(And(EI(x), CP(x)), HD(x)))\n4. CC(Sophia)\n5. PS(Sophia)\n6. RM(Sophia)\n7. CP(Sophia)\n\nQuestion: Is Sophia awarded an honors diploma?",
  "input": "",
  "output": "{\"answer\": \"Yes\", \"fol\": [\"ForAll(x, Implies(And(CC(x), PS(x)), QA(x)))\", \"CC(Sophia)\", \"PS(Sophia)\", \"ForAll(x, Implies(And(QA(x), RM(x)), EI(x)))\", \"RM(Sophia)\", \"ForAll(x, Implies(And(EI(x), CP(x)), HD(x)))\", \"CP(Sophia)\"], \"conclusion_fol\": \"HD(Sophia)\", \"cot\": [\"Step 1: CC(Sophia) ∧ PS(Sophia) + Premise 1 → QA(Sophia)\", \"Step 2: QA(Sophia) ∧ RM(Sophia) + Premise 2 → EI(Sophia)\", \"Step 3: EI(Sophia) ∧ CP(Sophia) + Premise 3 → HD(Sophia)\"], \"explanation\": \"Sophia has CC and PS, so she qualifies for advanced courses. With RM she becomes eligible for international program. With CP she is awarded honors diploma.\", \"premises_used\": [1, 2, 3, 4, 5, 6, 7], \"confidence\": 0.97}"
}
```

> **`conclusion_fol`** là field đặc biệt — model sinh ra dạng `HD(Sophia)` hoặc `ForAll(x, O(x))`. FOL→Z3 converter dùng field này để negate và check UNSAT.

### Bước 1C — Sinh Training Data Type 2 (Physics)

**INPUT** — 1 row từ CSV (chỉ lấy valid rows, bỏ id bắt đầu "QA"):
```
id: TD401
question: "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
cot: "Step 1: E = 0.5 * C * U^2 ..."
answer: "45"
unit: "J"
```

**OUTPUT** — 1 training sample trong `physics_train.jsonl`:
```json
{
  "instruction": "You are a physics problem solver. Solve step-by-step and generate Python code that prints the numerical answer.\n\nProblem: Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V.",
  "input": "",
  "output": "{\"python_code\": \"C = 100e-6\\nU = 30\\nE = 0.5 * C * U**2\\nprint(E)\", \"answer\": \"45\", \"unit\": \"J\", \"cot\": [\"Step 1: C = 100μF = 1e-4 F, U = 30V.\", \"Step 2: E = 0.5 × C × U² = 0.5 × 1e-4 × 900.\", \"Step 3: E = 0.045 J = 45 mJ.\"], \"explanation\": \"Using E = ½CU², with C = 10⁻⁴ F and U = 30V, stored energy = 45 mJ.\", \"confidence\": 0.98}"
}
```

### Bước 1D — Output Files

```
data/processed/
├── logic_train.jsonl    (~644 samples, từ 329 records × ~2 q/record)
├── logic_val.jsonl      (~81 samples)
├── logic_test.jsonl     (~83 samples)
├── physics_train.jsonl  (~1,218 samples, từ 1,354 valid × 0.9 split)
├── physics_val.jsonl    (~68 samples)
└── physics_test.jsonl   (~68 samples)
```

---

## Giai Đoạn 2: SFT Training

### LoRA 1 — Logic Reasoning

Model học sinh ra JSON output có các fields:
- `answer`: Yes / No / Uncertain / A / B / C / D
- `fol[]`: danh sách FOL strings dạng ForAll() cho premises liên quan
- `conclusion_fol`: FOL của kết luận (dùng để negate trong Z3)
- `cot[]`: chain of thought từng bước
- `explanation`: giải thích ngôn ngữ tự nhiên
- `premises_used[]`: chỉ số premises dùng

**Prompt format (Qwen 2.5 chat template):**
```
<|im_start|>system
You are a logical reasoning expert. Given premises in FOL (ForAll/Exists/Implies/And/Not/Or) format, answer the question. Always respond in valid JSON.
<|im_end|>
<|im_start|>user
Premises:
1. ForAll(x, Implies(And(CC(x), PS(x)), QA(x)))
...
Question: Is Sophia awarded an honors diploma?
<|im_end|>
<|im_start|>assistant
{"answer": "Yes", "fol": [...], "conclusion_fol": "HD(Sophia)", ...}
<|im_end|>
```

**Config:** QLoRA 4-bit, r=16, alpha=32, 3 epochs, LR=2e-4, bf16.
**Kết quả lưu:** `models/lora1_logic_sft/`

### LoRA 2 — Physics Solver

Model học sinh ra JSON với `python_code` thực thi được.

**Config:** Giống LoRA 1.
**Kết quả lưu:** `models/lora2_physics_sft/`

---

## Giai Đoạn 3: FOL→Z3 Converter (Developer viết)

> **Quan trọng:** Đây là hàm do lập trình viên viết — KHÔNG phải LLM sinh. Viết sau khi SFT xong, trước GRPO.

### Cơ chế hoạt động

```python
# fol_to_z3.py

def run_z3_verification(fol_list: list[str], conclusion_fol: str) -> str:
    """
    Nhận FOL strings + conclusion → chạy Z3 → trả về Yes/No/Uncertain.

    Args:
        fol_list: ["ForAll(x, Implies(CC(x), QA(x)))", "CC(Sophia)", ...]
        conclusion_fol: "QA(Sophia)" hoặc "ForAll(x, O(x))"

    Returns:
        "Yes" | "No" | "Uncertain"
    """
    # Bước 1: Collect tất cả predicate names và constants
    predicates, constants, variables = extract_symbols(fol_list + [conclusion_fol])

    # Bước 2: Sinh Z3 declarations
    z3_code = generate_declarations(predicates, constants, variables)

    # Bước 3: Add premises vào solver
    z3_code += "\ns = Solver()\n"
    for fol in fol_list:
        z3_expr = fol_to_z3_expr(fol)
        z3_code += f"s.add({z3_expr})\n"

    # Bước 4: Negate conclusion để check UNSAT
    conclusion_expr = fol_to_z3_expr(conclusion_fol)
    z3_code += f"s.add(Not({conclusion_expr}))\n"
    z3_code += "result = s.check()\n"
    z3_code += "print('RESULT:', result)\n"

    # Bước 5: Chạy trong sandbox
    output = run_in_sandbox(z3_code, timeout=10)

    # Bước 6: Parse kết quả
    if "unsat" in output:   return "Yes"      # Không thể phủ định → True
    elif "sat" in output:   return "No"       # Có thể phủ định → False
    else:                   return "Uncertain" # Unknown
```

### Ví dụ converter end-to-end

**Input:**
```python
fol_list = [
    "ForAll(x, Implies(And(CC(x), PS(x)), QA(x)))",
    "CC(Sophia)",
    "PS(Sophia)"
]
conclusion_fol = "QA(Sophia)"
```

**Converter sinh ra Z3 code:**
```python
from z3 import *

# Declarations
Entity = DeclareSort('Entity')
x = Const('x', Entity)

# Predicates (1-arg functions)
CC = Function('CC', Entity, BoolSort())
PS = Function('PS', Entity, BoolSort())
QA = Function('QA', Entity, BoolSort())

# Constants (uppercase = constants in our convention)
Sophia = Const('Sophia', Entity)

# Solver
s = Solver()
s.add(ForAll([x], Implies(And(CC(x), PS(x)), QA(x))))
s.add(CC(Sophia))
s.add(PS(Sophia))

# Negate conclusion
s.add(Not(QA(Sophia)))

result = s.check()
print('RESULT:', result)
```

**Output:** `RESULT: unsat` → `"Yes"` ✅

---

## Giai Đoạn 4: GRPO (Reinforcement Learning)

### Cơ chế thưởng/phạt — cả 2 loại dùng code execution

```
Với mỗi prompt, sinh G=8 responses:

─── TYPE 1 (Logic) ───────────────────────────────────────

Response[i] = {"answer":"Yes", "fol":[...], "conclusion_fol":"HD(Sophia)", ...}

Reward_i = 0.50 × R_correct
         + 0.25 × R_explanation
         + 0.15 × R_format
         + 0.10 × R_z3_execution

R_correct:
  Chạy converter → get z3_answer
  z3_answer == ground_truth → 1.0
  Nếu converter fail → dùng LLM answer == ground_truth → 0.7 (partial)
  Cả hai sai → 0.0

R_z3_execution:
  converter thành công + z3 không crash → 1.0
  converter thành công + z3 trả unknown → 0.5
  converter fail hoặc z3 crash → 0.0

─── TYPE 2 (Physics) ──────────────────────────────────────

Response[i] = {"answer":"45", "python_code":"C=100e-6; ...; print(E)", ...}

R_correct:
  Chạy python_code → stdout → parse float → so với ground_truth
  |pred - gt| / gt ≤ 1%  → 1.0
  |pred - gt| / gt ≤ 5%  → 0.6
  |pred - gt| / gt ≤ 10% → 0.3
  > 10% hoặc crash       → 0.0

R_code_execution:
  subprocess thành công, không exception → 1.0
  timeout hoặc exception → 0.0
```

### Cơ chế cập nhật gradient

```
Bước 1: Sinh 8 responses (temp=0.8 để diverse)
Bước 2: Tính reward R[0..7] cho từng response
Bước 3: Advantage A[i] = R[i] - mean(R[0..7])
         A[i] > 0 → THƯỞNG → tăng xác suất sinh response này
         A[i] < 0 → PHẠT  → giảm xác suất sinh response này
Bước 4: Loss = -mean(A[i] × log P(response[i] | prompt))
Bước 5: Backprop → chỉ update LoRA adapter weights
         Base Qwen 2.5 7B weights FROZEN
         KL penalty β=0.04 giữ không drift xa SFT
Bước 6: Lặp với prompt tiếp theo
```

**Kết quả lưu:** `models/lora1_logic_grpo_final/` và `models/lora2_physics_grpo_final/`

---

## Giai Đoạn 5: Inference Pipeline

### Luồng đầy đủ khi BTC gửi request

```
POST /predict  {JSON request}
       │
       ▼
  ┌────────────────────────────────┐
  │ ROUTER (rule-based, ~0ms)     │
  │ "premises-NL" in request?     │
  │   Có  → Type 1 (Logic)       │
  │   Không → Type 2 (Physics)   │
  └────────────────────────────────┘
       │
  ┌────┴───────────────────────────────┐
  │                                    │
  ▼ Type 1                     Type 2 ▼
  │                                    │
  LoRA 1 generate              LoRA 2 generate
  (~10-30s)                    (~10-30s)
  │                                    │
  ▼                                    ▼
  Parse JSON                  Parse JSON
  Lấy fol[] + conclusion_fol  Lấy python_code
  │                                    │
  ▼                                    ▼
  FOL→Z3 Converter             Python Sandbox
  Sinh Z3 Python code          subprocess.run()
  Chạy trong sandbox           Capture stdout
  (~1-3s)                      (~1-5s)
  │                                    │
  ▼                                    ▼
  Z3 result:                   exec_answer =
  unsat → "Yes"                stdout.strip()
  sat   → "No"                 (last line)
  unknown → "Uncertain"
  error → giữ LLM answer
  │                                    │
  └──────────────┬─────────────────────┘
                 │
                 ▼
         Override parsed["answer"]
         với kết quả từ solver/sandbox
                 │
                 ▼
         del parsed["conclusion_fol"]   ← strip trước khi gửi BTC
                 │
                 ▼
         Return JSON sạch (≤60s total)
```

### Ví dụ Type 1 — end-to-end

**BTC gửi:**
```json
{
  "premises-NL": [
    "Students who complete core curriculum and pass science are qualified.",
    "Students who are qualified and complete research methodology are eligible.",
    "Students eligible and who complete capstone are awarded honors diploma.",
    "Sophia has completed core curriculum.",
    "Sophia has passed science assessment.",
    "Sophia has completed research methodology.",
    "Sophia has completed a capstone project."
  ],
  "question": "Is Sophia awarded an honors diploma?"
}
```

**LoRA 1 raw output:**
```json
{
  "answer": "Yes",
  "fol": [
    "ForAll(x, Implies(And(CC(x), PS(x)), QA(x)))",
    "ForAll(x, Implies(And(QA(x), RM(x)), EI(x)))",
    "ForAll(x, Implies(And(EI(x), CP(x)), HD(x)))",
    "CC(Sophia)", "PS(Sophia)", "RM(Sophia)", "CP(Sophia)"
  ],
  "conclusion_fol": "HD(Sophia)",
  "cot": [
    "Step 1: CC(Sophia) ∧ PS(Sophia) + P1 → QA(Sophia)",
    "Step 2: QA(Sophia) ∧ RM(Sophia) + P2 → EI(Sophia)",
    "Step 3: EI(Sophia) ∧ CP(Sophia) + P3 → HD(Sophia)"
  ],
  "explanation": "Sophia completes CC and PS → qualified. With RM → eligible. With CP → honors diploma.",
  "premises_used": [1, 2, 3, 4, 5, 6, 7],
  "confidence": 0.97
}
```

**Converter chạy Z3:** `unsat` → `"Yes"` ✅ (khớp với LLM answer)

**Response trả BTC (sau khi strip `conclusion_fol`):**
```json
{
  "answer": "Yes",
  "fol": [
    "ForAll(x, Implies(And(CC(x), PS(x)), QA(x)))",
    "ForAll(x, Implies(And(QA(x), RM(x)), EI(x)))",
    "ForAll(x, Implies(And(EI(x), CP(x)), HD(x)))",
    "CC(Sophia)", "PS(Sophia)", "RM(Sophia)", "CP(Sophia)"
  ],
  "cot": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "explanation": "Sophia completes CC and PS → qualified...",
  "premises_used": [1, 2, 3, 4, 5, 6, 7],
  "confidence": 0.97
}
```

### Ví dụ Type 2 — end-to-end

**BTC gửi:**
```json
{
  "question": "A parallel circuit has R1 = 30 Ω, R2 = 60 Ω. Find the equivalent resistance."
}
```

**LoRA 2 raw output:**
```json
{
  "python_code": "R1 = 30\nR2 = 60\nR_eq = (R1 * R2) / (R1 + R2)\nprint(R_eq)",
  "answer": "20.0",
  "unit": "Ω",
  "cot": ["Step 1: 1/R_eq = 1/30 + 1/60 = 3/60", "Step 2: R_eq = 20 Ω"],
  "explanation": "For parallel resistors: R_eq = R1·R2/(R1+R2) = 20 Ω.",
  "confidence": 0.99
}
```

**Sandbox chạy:** stdout = `"20.0"` ✅

**Response trả BTC:**
```json
{
  "answer": "20.0",
  "unit": "Ω",
  "cot": ["Step 1: ...", "Step 2: ..."],
  "explanation": "For parallel resistors: R_eq = 20 Ω.",
  "confidence": 0.99
}
```

---

## Tiêu Chí Chấm Điểm (BTC)

| Tiêu chí | Mô tả | Field liên quan |
|----------|-------|-----------------|
| **P1 Correctness** | Đáp án đúng (auto-match) | `answer` |
| **P2 Explanation** | Chất lượng giải thích NL | `explanation` |
| **P3 Reasoning Depth** | FOL + CoT có thể kiểm chứng | `fol`, `cot`, `premises_used` |

---

## Lịch Trình 10 Ngày

```
Ngày 1   (20/5): Chuẩn hóa FOL format → 1 ForAll() style
Ngày 2-3 (21-22): Viết FOL→Z3 converter + unit test trên 411 records
Ngày 3-4 (22-23): prepare_data.py → logic_train.jsonl + physics_train.jsonl
Ngày 4-6 (23-25): SFT LoRA 1 (Logic) — ~4-6h A100
Ngày 5-7 (24-26): SFT LoRA 2 (Physics) — ~6-8h A100 (song song nếu có)
Ngày 7-8 (26-27): GRPO cho cả 2 LoRA — ~8-12h mỗi cái
Ngày 8-9 (27-28): Build FastAPI + test inference pipeline end-to-end
Ngày 9   (28/5):  Stress test (latency < 60s, error handling)
Ngày 10  (29/5):  Deploy vLLM + submit URL + viết solution description
```

---

## Cảnh Báo Quan Trọng

> [!WARNING]
> **Label "Unknown"** — Dataset dùng `"Unknown"`, không phải `"Uncertain"`. Training data phải dùng đúng label này.

> [!WARNING]
> **Physics: loại 401 rows QA** — Rows có `id` bắt đầu `"QA"` có `answer = NaN`. Phải drop trước khi tạo dataset.

> [!IMPORTANT]
> **`conclusion_fol`** phải được strip khỏi response trước khi gửi BTC. BTC không cần field này.

> [!IMPORTANT]
> **FOL→Z3 converter phải có fallback** — Nếu converter fail (FOL không parse được), giữ nguyên `answer` từ LLM thay vì throw error.

> [!TIP]
> **Convention phân biệt constant vs variable:** Uppercase = constant (`Sophia`), lowercase = variable (`x`). Converter dùng rule này để declare `Const()` vs `Const()` variable.
