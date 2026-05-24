import os
import json
import random
from typing import List, Dict, Any

def format_premises(premises_nl: List[str]) -> str:
    """Format premises as a numbered list string."""
    return "\n".join(f"{i + 1}. {premise}" for i, premise in enumerate(premises_nl))

def prepare_logic_sft(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """Process normalized logic data and prepare SFT dataset for DeepSeek-R1."""
    print(f"Loading normalized logic data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    samples = []
    skipped_count = 0

    # System prompt for Logic reasoning
    system_prompt = (
        "You are a logical reasoning expert. Given premises in natural language, convert them to "
        "First-Order Logic (FOL) formulas using ForAll, Exists, Implies, And, Or, Not syntax, "
        "identify the conclusion FOL, identify the premises used, and answer the question step-by-step. "
        "Always respond in valid JSON format."
    )

    for record_idx, record in enumerate(records):
        premises_nl = record.get("premises-NL", [])
        premises_fol = record.get("premises-FOL", [])
        questions = record.get("questions", [])
        answers = record.get("answers", [])
        explanations = record.get("explanation", [])
        indices_list = record.get("idx", [])
        conclusion_fol_list = record.get("conclusion_fol", [])

        # Format natural language premises
        formatted_premises = format_premises(premises_nl)

        for q_idx in range(len(questions)):
            try:
                question = questions[q_idx]
                answer = answers[q_idx]
                explanation = explanations[q_idx]
                used_indices = indices_list[q_idx]
                raw_conclusion_fol = conclusion_fol_list[q_idx]

                # Extract the corresponding FOL premises used
                fol_list = []
                for idx in used_indices:
                    # In dataset, indices are 1-based
                    if 1 <= idx <= len(premises_fol):
                        fol_list.append(premises_fol[idx - 1])
                    else:
                        print(f"Warning: Index {idx} out of range for premises-FOL in record {record_idx}")

                # Extract conclusion FOL based on question type (MCQ vs Yes/No)
                if isinstance(raw_conclusion_fol, dict):
                    # MCQ: extract the FOL of the correct answer option
                    conclusion_fol_for_q = raw_conclusion_fol.get(answer)
                    if not conclusion_fol_for_q:
                        # Fallback: if not found, use the first key or stringify
                        print(f"Warning: Correct answer '{answer}' not found in conclusion_fol dict for record {record_idx}, q_idx {q_idx}")
                        conclusion_fol_for_q = list(raw_conclusion_fol.values())[0] if raw_conclusion_fol else ""
                else:
                    # Yes/No: raw_conclusion_fol is directly a string
                    conclusion_fol_for_q = raw_conclusion_fol

                # Build the target JSON output
                json_data = {
                    "answer": answer,
                    "fol": fol_list,
                    "conclusion_fol": conclusion_fol_for_q,
                    "premises_used": used_indices,
                    "confidence": 1.0,
                    "explanation": explanation
                }
                
                # Serialized target JSON string
                json_output_str = json.dumps(json_data, ensure_ascii=False, indent=2)

                # Format conversational messages compatible with HF SFTTrainer chat_template
                # For DeepSeek-R1, the assistant content contains thinking inside <think> and JSON outside
                assistant_content = f"<think>\n{explanation}\n</think>\n{json_output_str}"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Premises:\n{formatted_premises}\n\nQuestion: {question}"},
                    {"role": "assistant", "content": assistant_content}
                ]

                samples.append({"messages": messages})

            except Exception as e:
                print(f"Error processing record {record_idx}, question {q_idx}: {e}")
                skipped_count += 1

    print(f"Processed {len(samples)} total logic training samples successfully. Skipped {skipped_count} samples due to errors.")

    # Shuffle and split dataset
    random.seed(seed)
    random.shuffle(samples)

    total_samples = len(samples)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    print(f"Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSONL files
    splits = {
        "logic_train.jsonl": train_samples,
        "logic_val.jsonl": val_samples,
        "logic_test.jsonl": test_samples
    }

    for filename, data in splits.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} samples to {filepath}")

if __name__ == "__main__":
    # Define absolute paths based on user's workspace
    workspace_dir = r"d:\Education\exact_2026"
    input_json = os.path.join(workspace_dir, "data", "processed", "Logic_Based_Educational_Queries_Normalization.json")
    output_directory = os.path.join(workspace_dir, "data", "processed")

    prepare_logic_sft(
        input_file=input_json,
        output_dir=output_directory,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
