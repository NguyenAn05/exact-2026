import sys
import os
import json
import re
import contextlib
import io
try:
    from z3.z3 import *
except ImportError:
    from z3 import *

# Helper functions for evaluating Z3 code in Logic_SFT

def check_property(solver, prop):
    """
    Check a boolean property under the solver's constraints.
    Returns:
        "Yes" if the property is proven to be True (Not(prop) is unsat)
        "No" if the property is proven to be False (prop is unsat)
        "Unknown" otherwise.
    """
    # Set a timeout (2000ms) to prevent hanging
    solver.set("timeout", 2000)
    
    # Try to prove prop is True (by showing Not(prop) is inconsistent with constraints)
    solver.push()
    solver.add(Not(prop))
    res = solver.check()
    solver.pop()
    if res == unsat:
        return "Yes"
        
    # Try to prove prop is False (by showing prop is inconsistent with constraints)
    solver.push()
    solver.add(prop)
    res = solver.check()
    solver.pop()
    if res == unsat:
        return "No"
        
    return "Unknown"


def check_mcq(solver, options_dict):
    """
    Evaluate multiple choice options under the solver's constraints.
    Returns:
        The key of the option (e.g., 'A', 'B', 'C', 'D') that is proven to be True.
        "Unknown" if none or multiple are proven.
    """
    # Set a timeout (2000ms) to prevent hanging
    solver.set("timeout", 2000)
    
    proved = []
    for opt, opt_expr in options_dict.items():
        solver.push()
        solver.add(Not(opt_expr))
        res = solver.check()
        solver.pop()
        if res == unsat:
            proved.append(opt)
            
    if len(proved) == 1:
        return proved[0]
    elif len(proved) > 1:
        # If multiple options are proven, log and return the first one
        return proved[0]
    return "Unknown"


def run_z3_code(z3_code_str: str) -> str:
    """
    Execute a Z3 code block string and return the extracted Answer.
    """
    # Prepare the execution namespace
    globals_dict = {
        'check_mcq': check_mcq,
        'check_property': check_property,
        '__builtins__': __builtins__
    }
    
    # Try importing from z3.z3 (specific to this environment) or fallback to standard z3 module
    try:
        import z3.z3 as z3_mod
    except ImportError:
        try:
            import z3 as z3_mod
        except ImportError as e:
            return f"Error: Z3 module not found ({e})"
            
    # Inject Z3 symbols into globals_dict for execution
    for name in dir(z3_mod):
        if not name.startswith('__'):
            globals_dict[name] = getattr(z3_mod, name)
            
    # Redirect stdout to capture the printed answer
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(z3_code_str, globals_dict)
    except Exception as e:
        return f"Error: {e}"
        
    output = stdout_capture.getvalue()
    
    # Extract the answer from printed output (e.g., "Answer: A" or "Answer: Yes")
    match = re.search(r"Answer:\s*(\w+)", output, re.IGNORECASE)
    if match:
        return match.group(1)
        
    # Check for printed output style like "Answer, Yes" or "Answer: Yes" with multiple spaces
    match_comma = re.search(r"Answer\s*,\s*(\w+)", output, re.IGNORECASE)
    if match_comma:
        return match_comma.group(1)
        
    return "Unknown"


def run_verification_on_dataset(dataset_path: str):
    """
    Load dataset, run all z3_code snippets, and compare with ground-truth answers.
    """
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    total_questions = 0
    correct_verifications = 0
    failed_verifications = 0
    errors_count = 0
    
    for record_idx, record in enumerate(data):
        z3_codes = record.get("z3_code", [])
        answers = record.get("answers", [])
        
        for q_idx, (code, expected_ans) in enumerate(zip(z3_codes, answers)):
            total_questions += 1
            predicted_ans = run_z3_code(code)
            
            # Normalize for comparison
            pred_norm = str(predicted_ans).strip().lower()
            exp_norm = str(expected_ans).strip().lower()
            
            if pred_norm == exp_norm:
                correct_verifications += 1
            else:
                if "error" in pred_norm:
                    errors_count += 1
                    print(f"Record {record_idx}, Question {q_idx}: Execution error: {predicted_ans}")
                else:
                    failed_verifications += 1
                    print(f"Record {record_idx}, Question {q_idx}: Mismatch. Expected: {expected_ans}, Got: {predicted_ans}")
                    
    print("\n" + "=" * 40)
    print("Verification Summary:")
    print(f"Total Questions Evaluated: {total_questions}")
    print(f"Successfully Verified (Matched Gold): {correct_verifications} ({correct_verifications / total_questions * 100:.2f}%)")
    print(f"Mismatches: {failed_verifications}")
    print(f"Execution Errors: {errors_count}")
    print("=" * 40)
