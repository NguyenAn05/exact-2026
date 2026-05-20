# Kế Hoạch Chi Tiết: Training Pipeline EXACT 2026

---

## Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Định Nghĩa Input / Output JSON](#2-định-nghĩa-input--output-json)
3. [Chuẩn Bị Dữ Liệu](#3-chuẩn-bị-dữ-liệu)
4. [Giai Đoạn 1: SFT (Supervised Fine-Tuning)](#4-giai-đoạn-1-sft-supervised-fine-tuning)
5. [Giai Đoạn 2: GRPO (Reinforcement Learning)](#5-giai-đoạn-2-grpo-reinforcement-learning)
6. [Pipeline Inference](#6-pipeline-inference)
7. [Lịch Trình &amp; Checklist](#7-lịch-trình--checklist)
8. [Cấu Hình Cloud &amp; Tài Nguyên](#8-cấu-hình-cloud--tài-nguyên)

---

## 1. Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────┐
│                  Nhận Request (Input JSON)               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Bộ Định Tuyến (Rule-based Router)           │
│   Check: "premises-NL" key có tồn tại trong JSON?        │
│   → Có: Type 1 (Logic)  |  Không: Type 2 (Physics)      │
└───────────────┬─────────────────────────┬───────────────┘
                │                         │
         Type 1 │                         │ Type 2
                ▼                         ▼
┌───────────────────────┐   ┌─────────────────────────────┐
│  LoRA 1: NL → FOL     │   │  LoRA 2: Sinh Python Code   │
│  (Qwen 2.5 7B + SFT)  │   │  (Qwen 2.5 7B + SFT)        │
└──────────┬────────────┘   └──────────────┬──────────────┘
           │                               │
           ▼                               ▼
┌─────────────────────┐     ┌──────────────────────────────┐
│   Z3 Solver (Logic) │     │  Sandbox Exec (Python/SymPy) │
│   → Answer: Y/N/U   │     │  → Numerical Answer + Unit   │
└──────────┬──────────┘     └──────────────┬───────────────┘
           │   [Nếu UNSAT/Error]           │  [Nếu Error]
           │   ↓ Fallback LLM              │  ↓ Fallback LLM
           └──────────────┬────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│          Sinh Giải Thích (Qwen 2.5 7B LLM)              │
│          Tạo explanation + cot + fol (nếu Type 1)        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│               Format & Validate JSON Output              │
└─────────────────────────────────────────────────────────┘
```

### BaLoRA Adapter riêng biệt trên cùng base model:

| Adapter          | Nhiệm vụ                               | Training Data     | Output                           |
| ---------------- | ---------------------------------------- | ----------------- | -------------------------------- |
| **LoRA 1** | NL premises → FOL + Answer (Type 1)     | 411 records JSON  | FOL string + Yes/No/Uncertain/MC |
| **LoRA 2** | Physics question → Python code (Type 2) | 1,755 records CSV | Python code → numerical answer  |
| **LoRA 3** | Sinh giải thích tự nhiên             | Cả hai dataset   | Explanation string               |

> **Lưu ý:** LoRA 3 (sinh giải thích) có thể gộp vào LoRA 1 hoặc LoRA 2, hoặc tách riêng tùy VRAM.

---

## 2. Định Nghĩa Input / Output JSON

### 2.1 API Input — Type 1 (Logic-Based)

Dựa trực tiếp từ file `Logic_Based_Educational_Queries.json`:

```json
{
  "type": "logic",
  "premises-NL": [
    "Students who have completed the core curriculum and passed the science assessment are qualified for advanced courses.",
    "Students who are qualified for advanced courses and have completed research methodology are eligible for the international program.",
    "Students who are eligible for the international program and have completed a capstone project are awarded an honors diploma.",
    "Students who have been awarded an honors diploma and have completed community service qualify for the university scholarship.",
    "Sophia has completed the core curriculum.",
    "Sophia has passed the science assessment.",
    "Sophia has completed research methodology.",
    "Sophia has completed a capstone project.",
    "Sophia has completed community service."
  ],
  "question": "Does Sophia qualify for the university scholarship?"
}
```

### 2.2 API Input — Type 2 (Physics)

Dựa từ file `Physics_Problems_Text_Only.csv`:

```json
{
  "type": "physics",
  "question": "A parallel circuit has R1 = 30 Ω, R2 = 60 Ω, U = 12 V. Calculate the equivalent resistance."
}
```

> **Lưu ý thực tế:** Trong test set chính thức, không có field `"type"`. Router sẽ tự phát hiện dựa trên sự tồn tại của `"premises-NL"`.

### 2.3 API Input — Thực tế từ ban tổ chức (Unified Format)

```json
// Type 1 — có premises-NL
{
  "premises-NL": [
    "If a Python code is well-tested, then the project is optimized.",
    "All Python code is well-tested.",
    "If a Python project is optimized, then it has clean and readable code."
  ],
  "question": "Does all Python code have clean and readable code?"
}

// Type 2 — chỉ có question
{
  "question": "Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V."
}
```

### 2.4 API Output — Type 1 (Logic)

```json
{
  "answer": "Yes",
  "explanation": "Based on the given premises, all Python code is well-tested (Premise 2). Since it is well-tested, the project is optimized (Premise 1). Since it is optimized, it has clean and readable code (Premise 3). Therefore, all Python code has clean and readable code.",
  "fol": [
    "∀x (WT(x) → O(x))",
    "∀x (WT(x))",
    "∀x (O(x) → CR(x))",
    "⊢ ∀x (CR(x))"
  ],
  "cot": [
    "Step 1: From Premise 2, establish ∀x WT(x) — all Python code is well-tested.",
    "Step 2: Apply Premise 1: WT(x) → O(x), therefore ∀x O(x) — all code is optimized.",
    "Step 3: Apply Premise 3: O(x) → CR(x), therefore ∀x CR(x) — all code is clean and readable.",
    "Step 4: Conclusion: Yes, all Python code has clean and readable code."
  ],
  "premises_used": [1, 2, 3],
  "confidence": 0.95,
  "solver": "z3"
}
```

### 2.5 API Output — Type 2 (Physics)

```json
{
  "answer": "45.0",
  "unit": "J",
  "explanation": "The energy stored in a capacitor is calculated using the formula E = 0.5 × C × U². With C = 100 μF = 1×10⁻⁴ F and U = 30 V, substituting gives E = 0.5 × 1×10⁻⁴ × 900 = 0.045 J = 45 mJ.",
  "cot": [
    "Step 1: Identify given values: C = 100 μF = 1×10⁻⁴ F, U = 30 V.",
    "Step 2: Recall energy formula: E = 0.5 × C × U².",
    "Step 3: Substitute: E = 0.5 × (1×10⁻⁴) × (30)² = 0.5 × 10⁻⁴ × 900.",
    "Step 4: Calculate: E = 0.5 × 0.09 = 0.045 J = 45 mJ."
  ],
  "python_code": "C = 100e-6  # Farads\nU = 30      # Volts\nE = 0.5 * C * U**2\nprint(f'E = {E} J')",
  "confidence": 0.98,
  "solver": "python_exec"
}
```

### 2.6 Output — Multiple Choice (MC)

```json
{
  "answer": "C",
  "explanation": "Option C states 'If a Python project is well-tested, then it must be clean and readable.' This follows directly from: Premise 4 (WT→WT), Premise 1 (WT→O), and Premise 9 (O→CR), requiring 3 steps.",
  "cot": [
    "Step 1: Evaluate Option A — requires 1 premise (contrapositive of P1). Valid but check others.",
    "Step 2: Evaluate Option B — optimization does not imply well-structured. Invalid.",
    "Step 3: Evaluate Option C — follows from P4→P1→P9, requires 3 steps.",
    "Step 4: Evaluate Option D — follows from P1+P6, 2 steps.",
    "Step 5: The question asks which follows with FEWEST premises → Option A (1 premise)."
  ],
  "fol": "¬O(x) → ¬WT(x)  [contrapositive of WT(x) → O(x)]",
  "premises_used": [1],
  "confidence": 0.87,
  "solver": "z3"
}
```

---

## 3. Chuẩn Bị Dữ Liệu

### 3.1 Phân chia Dataset

#### Type 1 — Logic (411 records, 808 questions)

```
Train : Val : Test = 329 : 41 : 41 records
                   (~644 : ~81 : ~83 questions)
```

#### Type 2 — Physics (1,755 records)

```
Train : Val : Test = 1,404 : 175 : 176 records
```

### 3.2 Format SFT Dataset — Type 1

Chuyển mỗi record JSON thành Instruction-Response format:

```json
{
  "instruction": "You are a logical reasoning expert. Given the following premises, answer the question and provide a step-by-step explanation with FOL derivation.\n\nPremises:\n1. If a Python code is well-tested, then the project is optimized.\n2. All Python code is well-tested.\n3. If a Python project is optimized, then it has clean and readable code.\n\nQuestion: Does all Python code have clean and readable code?",

  "input": "",

  "output": "{\n  \"answer\": \"Yes\",\n  \"explanation\": \"All Python code is well-tested (P2). Since well-tested code is optimized (P1), all code is optimized. Since optimized code is clean and readable (P3), all code is clean and readable.\",\n  \"fol\": [\"∀x WT(x)\", \"∀x (WT(x) → O(x))\", \"∀x (O(x) → CR(x))\", \"⊢ ∀x CR(x)\"],\n  \"cot\": [\"Step 1: P2 gives ∀x WT(x).\", \"Step 2: P1 gives ∀x O(x).\", \"Step 3: P3 gives ∀x CR(x).\"],\n  \"premises_used\": [1, 2, 3],\n  \"confidence\": 0.95\n}"
}
```

### 3.3 Format SFT Dataset — Type 2

```json
{
  "instruction": "You are a physics problem solver. Solve the following problem step-by-step using Chain-of-Thought reasoning. Generate Python code to compute the answer, then provide a clear explanation.\n\nProblem: Calculate the energy stored in capacitor C when C = 100 μF and U = 30 V.",

  "input": "",

  "output": "{\n  \"python_code\": \"C = 100e-6\\nU = 30\\nE = 0.5 * C * U**2\\nprint(E)\",\n  \"answer\": \"45.0\",\n  \"unit\": \"J\",\n  \"cot\": [\"Step 1: C = 100 μF = 1e-4 F, U = 30 V.\", \"Step 2: E = 0.5 × C × U² = 0.5 × 1e-4 × 900.\", \"Step 3: E = 0.045 J = 45 mJ.\"],\n  \"explanation\": \"Using E = 0.5CU², with C = 1×10⁻⁴ F and U = 30 V, E = 45 mJ.\",\n  \"confidence\": 0.98\n}"
}
```

### 3.4 Script Chuẩn Bị Dữ Liệu

```python
# prepare_data.py
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# ── TYPE 1 ──────────────────────────────────────────────
with open('Logic_Based_Educational_Queries.json', 'r') as f:
    logic_data = json.load(f)

def format_logic_sample(record):
    samples = []
    premises_str = "\n".join(
        f"{i+1}. {p}" for i, p in enumerate(record['premises-NL'])
    )
    for q, a, e in zip(record['questions'], record['answers'], record['explanation']):
        instruction = (
            f"You are a logical reasoning expert. Given the following premises, "
            f"answer the question and provide FOL derivation.\n\n"
            f"Premises:\n{premises_str}\n\nQuestion: {q}"
        )
        output = json.dumps({
            "answer": a,
            "explanation": e,
            "cot": [f"Step {i+1}: Analyze premise {i+1}." for i in range(min(3, len(record['premises-NL'])))],
            "premises_used": list(range(1, len(record['premises-NL'])+1)),
            "confidence": 0.9
        }, ensure_ascii=False)
        samples.append({"instruction": instruction, "input": "", "output": output})
    return samples

logic_samples = []
for record in logic_data:
    logic_samples.extend(format_logic_sample(record))

# ── TYPE 2 ──────────────────────────────────────────────
physics_df = pd.read_csv('Physics_Problems_Text_Only.csv').dropna(subset=['answer'])

def format_physics_sample(row):
    instruction = (
        f"You are a physics problem solver. Solve using Chain-of-Thought "
        f"and generate Python code for the computation.\n\nProblem: {row['question']}"
    )
    cot_steps = [s.strip() for s in str(row['cot']).split('\n') if s.strip()]
    output = json.dumps({
        "answer": str(row['answer']),
        "unit": str(row['unit']) if pd.notna(row['unit']) else "",
        "cot": cot_steps,
        "explanation": f"The answer is {row['answer']} {row['unit'] if pd.notna(row['unit']) else ''}.",
        "confidence": 0.9
    }, ensure_ascii=False)
    return {"instruction": instruction, "input": "", "output": output}

physics_samples = [format_physics_sample(row) for _, row in physics_df.iterrows()]

# ── SPLIT & SAVE ─────────────────────────────────────────
def split_save(samples, prefix):
    train, val = train_test_split(samples, test_size=0.1, random_state=42)
    val, test = train_test_split(val, test_size=0.5, random_state=42)
    for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
        with open(f'{prefix}_{split_name}.json', 'w') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
    print(f"{prefix}: train={len(train)}, val={len(val)}, test={len(test)}")

split_save(logic_samples, 'logic')
split_save(physics_samples, 'physics')
```

---

## 4. Giai Đoạn 1: SFT (Supervised Fine-Tuning)

### 4.1 Cấu Hình Chung

```python
# sft_config.py
from transformers import TrainingArguments
from peft import LoraConfig

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# LoRA Config — dùng cho cả LoRA 1 và LoRA 2
LORA_CONFIG = LoraConfig(
    r=16,                          # Rank — tăng lên 32 nếu VRAM > 24GB
    lora_alpha=32,                 # Alpha = 2 × r
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

TRAINING_ARGS = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=2,     # Với A100 40GB
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,     # Effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=False,
    bf16=True,                         # Qwen 2.5 dùng bf16
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    evaluation_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="wandb",                 # Hoặc "tensorboard"
    gradient_checkpointing=True,       # BẮT BUỘC cho 7B trên cloud
    optim="paged_adamw_8bit",         # Tiết kiệm VRAM
    max_grad_norm=1.0,
    dataloader_num_workers=4,
)
```

### 4.2 Training Script — LoRA 1 (Type 1 Logic)

```python
# train_lora1_logic.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
import torch

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 2048

# Load model với 4-bit quantization (QLoRA)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()
# Dự kiến: ~0.7% params trainable (~53M/7B)

# Load dataset
dataset = load_dataset("json", data_files={
    "train": "logic_train.json",
    "validation": "logic_val.json"
})

def format_prompt(sample):
    return f"""<|im_start|>system
You are a logical reasoning expert. Always respond in valid JSON format.<|im_end|>
<|im_start|>user
{sample['instruction']}<|im_end|>
<|im_start|>assistant
{sample['output']}<|im_end|>"""

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    args=TRAINING_ARGS,
    formatting_func=format_prompt,
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

trainer.train()
trainer.save_model("./lora1_logic_sft")
tokenizer.save_pretrained("./lora1_logic_sft")
```

### 4.3 Training Script — LoRA 2 (Type 2 Physics)

```python
# train_lora2_physics.py
# Tương tự LoRA 1 nhưng đổi prompt system và dataset

SYSTEM_PROMPT = """You are a physics problem solver specializing in electric circuits and electrostatics. 
Generate Python code to compute the answer, then explain step-by-step. 
Always respond in valid JSON format with fields: python_code, answer, unit, cot, explanation."""

def format_prompt_physics(sample):
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{sample['instruction']}<|im_end|>
<|im_start|>assistant
{sample['output']}<|im_end|>"""

# Load physics dataset
dataset = load_dataset("json", data_files={
    "train": "physics_train.json",
    "validation": "physics_val.json"
})

# Train (cấu hình giống LoRA 1)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    args=TRAINING_ARGS,
    formatting_func=format_prompt_physics,
    max_seq_length=MAX_SEQ_LEN,
)

trainer.train()
trainer.save_model("./lora2_physics_sft")
```

### 4.4 Metrics Đánh Giá Sau SFT

| Metric                        | Type 1 Target | Type 2 Target |
| ----------------------------- | ------------- | ------------- |
| Answer Accuracy (Exact Match) | ≥ 70%        | ≥ 65%        |
| JSON Parse Rate               | ≥ 95%        | ≥ 95%        |
| ROUGE-L (explanation)         | ≥ 0.35       | ≥ 0.40       |
| FOL Valid Rate (Z3 parseable) | ≥ 60%        | N/A           |
| Code Exec Success Rate        | N/A           | ≥ 75%        |

---

## 5. Giai Đoạn 2: GRPO (Reinforcement Learning)

### 5.1 Tổng Quan GRPO

GRPO (Group Relative Policy Optimization) tối ưu trực tiếp trên reward, không cần reward model riêng. Với mỗi prompt, sinh G=8 responses, tính relative reward trong nhóm.

```
R_total(response) = α·R_correct + β·R_explain + γ·R_format + δ·R_fol

Trong đó:
  α = 0.50  (correctness — quan trọng nhất → P1)
  β = 0.25  (explanation quality → P2)
  γ = 0.15  (JSON format valid → kỹ thuật)
  δ = 0.10  (FOL validity, chỉ Type 1 → P3)
```

### 5.2 Định Nghĩa Reward Functions

```python
# reward_functions.py
import json
import re
from z3 import *
from rouge_score import rouge_scorer
import subprocess, tempfile, os

# ── R1: Correctness ─────────────────────────────────────
def reward_correctness(response: str, ground_truth: str, q_type: str) -> float:
    """
    Trả về 1.0 nếu đúng, 0.0 nếu sai, 0.3 nếu gần đúng (physics).
    """
    try:
        parsed = json.loads(response)
        pred_answer = str(parsed.get("answer", "")).strip().lower()
        gt = str(ground_truth).strip().lower()

        if q_type == "logic":
            # Exact match cho Yes/No/Uncertain/MC
            return 1.0 if pred_answer == gt else 0.0

        elif q_type == "physics":
            # Numeric tolerance ±1%
            try:
                pred_val = float(pred_answer)
                gt_val = float(gt)
                rel_error = abs(pred_val - gt_val) / (abs(gt_val) + 1e-9)
                if rel_error <= 0.01:   return 1.0   # ±1%
                elif rel_error <= 0.05: return 0.6   # ±5%
                elif rel_error <= 0.10: return 0.3   # ±10%
                else:                   return 0.0
            except:
                return 0.0
    except:
        return 0.0

# ── R2: Explanation Quality ──────────────────────────────
scorer_rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def reward_explanation(response: str, reference_explanation: str) -> float:
    """
    ROUGE-L score của explanation so với human-written explanation.
    """
    try:
        parsed = json.loads(response)
        pred_exp = parsed.get("explanation", "")
        if not pred_exp:
            return 0.0
        score = scorer_rouge.score(reference_explanation, pred_exp)
        rouge_l = score['rougeL'].fmeasure
        # Bonus nếu CoT có đủ steps
        cot = parsed.get("cot", [])
        cot_bonus = min(len(cot) * 0.02, 0.1)  # max +0.1
        return min(rouge_l + cot_bonus, 1.0)
    except:
        return 0.0

# ── R3: JSON Format Valid ────────────────────────────────
def reward_format(response: str, q_type: str) -> float:
    """
    Kiểm tra JSON hợp lệ và có đủ required fields.
    """
    try:
        parsed = json.loads(response)
        required_fields = ["answer", "explanation", "cot"]
        optional_bonus = 0.0

        # Check required fields
        for field in required_fields:
            if field not in parsed:
                return 0.0

        # Bonus cho optional fields
        if q_type == "logic":
            if "fol" in parsed:        optional_bonus += 0.1
            if "premises_used" in parsed: optional_bonus += 0.05
        elif q_type == "physics":
            if "unit" in parsed:       optional_bonus += 0.1
            if "python_code" in parsed: optional_bonus += 0.05

        return min(0.85 + optional_bonus, 1.0)
    except json.JSONDecodeError:
        return 0.0

# ── R4: FOL Validity (Type 1 only) ──────────────────────
def reward_fol_validity(response: str) -> float:
    """
    Kiểm tra FOL strings có thể parse được bởi Z3.
    Đây là heuristic đơn giản — parser FOL đầy đủ phức tạp hơn.
    """
    try:
        parsed = json.loads(response)
        fol_list = parsed.get("fol", [])
        if not fol_list:
            return 0.0
        # Heuristic: check FOL có chứa quantifier hợp lệ
        valid_count = 0
        for fol in fol_list:
            if any(sym in fol for sym in ["∀", "∃", "ForAll", "Exists", "→", "->"]):
                valid_count += 1
        return valid_count / max(len(fol_list), 1)
    except:
        return 0.0

# ── R5: Code Execution (Type 2 only) ────────────────────
def reward_code_execution(response: str, expected_answer: float) -> float:
    """
    Chạy Python code trong sandbox, so sánh kết quả.
    """
    try:
        parsed = json.loads(response)
        code = parsed.get("python_code", "")
        if not code:
            return 0.0

        # Sandbox execution với timeout
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout.strip()
            exec_val = float(output.split('\n')[-1])
            rel_error = abs(exec_val - expected_answer) / (abs(expected_answer) + 1e-9)
            return 1.0 if rel_error <= 0.01 else 0.3
        except:
            return 0.0
        finally:
            os.unlink(tmp_path)
    except:
        return 0.0

# ── COMBINED REWARD ──────────────────────────────────────
def compute_total_reward(response, ground_truth, reference_exp, q_type, expected_num=None):
    r_correct  = reward_correctness(response, ground_truth, q_type)
    r_explain  = reward_explanation(response, reference_exp)
    r_format   = reward_format(response, q_type)

    if q_type == "logic":
        r_fol = reward_fol_validity(response)
        total = 0.50*r_correct + 0.25*r_explain + 0.15*r_format + 0.10*r_fol
    else:  # physics
        r_code = reward_code_execution(response, expected_num) if expected_num else 0.0
        total = 0.50*r_correct + 0.25*r_explain + 0.15*r_format + 0.10*r_code

    return {
        "total": round(total, 4),
        "correctness": round(r_correct, 4),
        "explanation": round(r_explain, 4),
        "format": round(r_format, 4),
    }
```

### 5.3 GRPO Training Script

```python
# train_grpo.py
from trl import GRPOConfig, GRPOTrainer
from peft import PeftModel
import torch

# Load SFT model đã train
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
)
# Load LoRA 1 adapter (train GRPO riêng cho từng type)
model = PeftModel.from_pretrained(base_model, "./lora1_logic_sft")

# GRPO Config
grpo_config = GRPOConfig(
    output_dir="./lora1_logic_grpo",
    num_train_epochs=1,                # GRPO không cần nhiều epoch
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,                # Nhỏ hơn SFT nhiều
    num_generations=8,                 # G=8 responses per prompt
    max_new_tokens=512,
    temperature=0.8,                   # Diverse generation
    beta=0.04,                         # KL penalty (giữ gần SFT policy)
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    save_steps=50,
    report_to="wandb",
)

# Reward function wrapper cho GRPO Trainer
def reward_fn(completions, prompts, **kwargs):
    rewards = []
    for completion, prompt_data in zip(completions, kwargs.get('ground_truths', [])):
        r = compute_total_reward(
            response=completion,
            ground_truth=prompt_data['answer'],
            reference_exp=prompt_data['explanation'],
            q_type="logic"
        )
        rewards.append(r['total'])
    return rewards

grpo_trainer = GRPOTrainer(
    model=model,
    config=grpo_config,
    reward_funcs=reward_fn,
    train_dataset=grpo_logic_dataset,  # Dataset đã format
)

grpo_trainer.train()
grpo_trainer.save_model("./lora1_logic_grpo_final")
```

### 5.4 Kỳ Vọng Cải Thiện Sau GRPO

| Metric                   | Sau SFT | Sau GRPO | Cải thiện |
| ------------------------ | ------- | -------- | ----------- |
| Answer Accuracy (Type 1) | ~70%    | ~80%     | +10%        |
| Answer Accuracy (Type 2) | ~65%    | ~76%     | +11%        |
| JSON Parse Rate          | ~95%    | ~99%     | +4%         |
| ROUGE-L Explanation      | ~0.35   | ~0.45    | +0.10       |
| FOL Valid Rate           | ~60%    | ~75%     | +15%        |
| Code Exec Success        | ~75%    | ~87%     | +12%        |

---

## 6. Pipeline Inference

### 6.1 Router (Rule-based, không cần ML)

```python
# router.py
def classify_query(input_json: dict) -> str:
    """
    Rule-based: nhanh, chính xác 100% cho format chuẩn EXACT 2026.
    """
    if "premises-NL" in input_json and input_json["premises-NL"]:
        return "logic"
    return "physics"
```

### 6.2 Z3 Solver Wrapper

```python
# z3_solver.py
from z3 import *

def solve_with_z3(fol_strings: list, question_type: str) -> dict:
    """
    Giải bài toán logic với Z3.
    Trả về: {"answer": "Yes/No/Uncertain", "proof": [...]}
    """
    solver = Solver()
    # Parse FOL strings → Z3 assertions
    # (đây là phần phức tạp nhất, cần custom parser)
    try:
        # ... parse và add assertions ...
        result = solver.check()
        if result == sat:
            return {"answer": "Yes", "proof": str(solver.model())}
        elif result == unsat:
            return {"answer": "No", "proof": "UNSAT"}
        else:
            return {"answer": "Uncertain", "proof": "UNKNOWN"}
    except Exception as e:
        return {"answer": "fallback", "error": str(e)}
```

### 6.3 Main Inference API

```python
# inference_api.py
from fastapi import FastAPI
from peft import PeftModel
import torch, json, subprocess, tempfile, os

app = FastAPI()

# Load models once at startup
@app.on_event("startup")
async def load_models():
    global tokenizer, lora1_model, lora2_model, base_model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    lora1_model = PeftModel.from_pretrained(base_model, "./lora1_logic_grpo_final")
    lora2_model = PeftModel.from_pretrained(base_model, "./lora2_physics_grpo_final")

@app.post("/predict")
async def predict(request: dict):
    q_type = classify_query(request)

    if q_type == "logic":
        # Step 1: Generate FOL với LoRA 1
        raw_output = generate_response(lora1_model, tokenizer, request, "logic")

        # Step 2: Parse output
        parsed = safe_json_parse(raw_output)

        # Step 3: Verify với Z3
        if parsed and "fol" in parsed:
            z3_result = solve_with_z3(parsed["fol"], q_type)
            if z3_result["answer"] != "fallback":
                parsed["answer"] = z3_result["answer"]
                parsed["solver"] = "z3"
            else:
                parsed["solver"] = "llm_fallback"

    else:  # physics
        # Step 1: Generate Python code với LoRA 2
        raw_output = generate_response(lora2_model, tokenizer, request, "physics")
        parsed = safe_json_parse(raw_output)

        # Step 2: Execute code để lấy numerical answer
        if parsed and "python_code" in parsed:
            exec_result = safe_execute_python(parsed["python_code"])
            if exec_result["success"]:
                parsed["answer"] = exec_result["output"]
                parsed["solver"] = "python_exec"
            else:
                parsed["solver"] = "llm_fallback"

    return parsed or {"answer": "Uncertain", "explanation": "Processing failed.", "confidence": 0.0}

def generate_response(model, tokenizer, request, q_type, max_new_tokens=512):
    if q_type == "logic":
        premises = "\n".join(f"{i+1}. {p}" for i, p in enumerate(request["premises-NL"]))
        prompt = f"<|im_start|>system\nYou are a logical reasoning expert. Respond in JSON.<|im_end|>\n<|im_start|>user\nPremises:\n{premises}\n\nQuestion: {request['question']}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>system\nYou are a physics solver. Respond in JSON.<|im_end|>\n<|im_start|>user\n{request['question']}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,       # Thấp để output ổn định
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def safe_json_parse(text: str) -> dict | None:
    # Thử parse trực tiếp
    try:
        return json.loads(text)
    except:
        pass
    # Thử extract JSON block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return None

def safe_execute_python(code: str, timeout: int = 10) -> dict:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            tmp = f.name
        result = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=timeout)
        os.unlink(tmp)
        return {"success": True, "output": result.stdout.strip().split('\n')[-1]}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## 7. Lịch Trình & Checklist

### Timeline Chi Tiết (10/5 → 30/5)

```
Tuần 1 (10/5 – 16/5): Data & SFT Setup
├── [ ] Day 1-2: Cài đặt môi trường cloud, clone repo Qwen 2.5
├── [ ] Day 2-3: Chạy prepare_data.py, kiểm tra format JSON
├── [ ] Day 3-4: Train LoRA 1 (Logic SFT) — ~4-6h A100
├── [ ] Day 5-6: Train LoRA 2 (Physics SFT) — ~6-8h A100
└── [ ] Day 7:   Evaluate SFT, fix bugs, re-train nếu cần

Tuần 2 (17/5 – 23/5): GRPO & Integration
├── [ ] Day 8-9:  Implement reward functions, unit test từng hàm
├── [ ] Day 10-11: GRPO LoRA 1 — ~8-12h (chậm hơn SFT)
├── [ ] Day 12-13: GRPO LoRA 2 — ~8-12h
└── [ ] Day 14:   Tích hợp Z3 Solver + Python Sandbox

Tuần 3 (24/5 – 30/5): Pipeline & Submission
├── [ ] Day 15-16: Build FastAPI endpoint, test end-to-end
├── [ ] Day 17-18: Stress test (latency < 60s, error handling)
├── [ ] Day 19:   Evaluation trên val set, so sánh SFT vs GRPO
├── [ ] Day 20:   Viết solution description (1 trang)
└── [ ] Day 21:   Final submission
```

### Checklist Kỹ Thuật

**Data Preparation**

- [ ] JSON format đúng cấu trúc Instruction-Response
- [ ] Train/Val/Test split không bị data leak
- [ ] Tokenization test (không vượt MAX_SEQ_LEN)
- [ ] Special tokens được xử lý đúng cho Qwen 2.5

**SFT**

- [ ] LoRA 1 JSON parse rate ≥ 95% trên val
- [ ] LoRA 2 code exec success ≥ 75% trên val
- [ ] Checkpoint saved đúng format PEFT
- [ ] Training loss hội tụ (không diverge)

**GRPO**

- [ ] Mỗi reward function đều unit-tested
- [ ] Reward không bị sparse (ít nhất 30% response có R > 0.5)
- [ ] KL divergence không vượt quá ngưỡng (monitor wandb)
- [ ] GRPO cải thiện so với SFT trên val set

**Inference**

- [ ] API trả về response trong < 60 giây
- [ ] JSON output hợp lệ 100% (có fallback)
- [ ] Router classify đúng 100% (rule-based)
- [ ] Z3 Solver fallback hoạt động khi parse lỗi
- [ ] Python sandbox có timeout + exception handling

---


## 📌 Tóm Tắt Nhanh

| Giai đoạn     | Model       | Thời gian          | Output                                       |
| --------------- | ----------- | ------------------- | -------------------------------------------- |
| Data Prep       | —          | 2 ngày             | `logic_train.json`, `physics_train.json` |
| SFT LoRA 1      | Qwen 2.5 7B | 4-6h                | `lora1_logic_sft/`                         |
| SFT LoRA 2      | Qwen 2.5 7B | 6-8h                | `lora2_physics_sft/`                       |
| GRPO LoRA 1     | Qwen 2.5 7B | 8-12h               | `lora1_logic_grpo_final/`                  |
| GRPO LoRA 2     | Qwen 2.5 7B | 8-12h               | `lora2_physics_grpo_final/`                |
| Integration     | FastAPI     | 3 ngày             | API endpoint                                 |
| **Total** |             | **~21 ngày** | **Submission ready**                   |
