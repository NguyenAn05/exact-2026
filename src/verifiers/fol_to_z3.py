import re
import ast
from z3 import z3 as _z3
from typing import List, Dict, Any, Union

# Define Universal Real Sort for generic entity domain in Z3
UniversalSort = _z3.RealSort()

def coerce_to_bool(expr):
    """Gradually coerce the expression to a boolean context for Z3 compatibility"""
    if isinstance(expr, bool):
        return expr
    if hasattr(expr, 'sort'):
        sort = expr.sort()
        if sort == _z3.BoolSort():
            return expr
        elif sort == _z3.RealSort() or sort == _z3.IntSort():
            return expr != 0
    return expr

# Custom Z3 operators wrapping original Z3 operators with type coercion
def CustomAnd(*args):
    coerced = [coerce_to_bool(x) for x in args]
    return _z3.And(*coerced)

def CustomOr(*args):
    coerced = [coerce_to_bool(x) for x in args]
    return _z3.Or(*coerced)

def CustomNot(arg):
    return _z3.Not(coerce_to_bool(arg))

def CustomImplies(a, b):
    return _z3.Implies(coerce_to_bool(a), coerce_to_bool(b))

def CustomIff(a, b):
    return coerce_to_bool(a) == coerce_to_bool(b)

def CustomForAll(vars_list, body):
    return _z3.ForAll(vars_list, coerce_to_bool(body))

def CustomExists(vars_list, body):
    return _z3.Exists(vars_list, coerce_to_bool(body))

def clean_and_register_names(fol_str: str, string_map: dict) -> str:
    fol_clean = fol_str.strip().rstrip('.$')
    string_literals = re.findall(r"'([^']*)'", fol_clean)
    for s_lit in string_literals:
        existing_var = None
        for var, val in string_map.items():
            if val == s_lit:
                existing_var = var
                break
        if existing_var:
            fol_clean = fol_clean.replace(f"'{s_lit}'", existing_var)
        else:
            var_name = f'str_lit_{len(string_map)}'
            fol_clean = fol_clean.replace(f"'{s_lit}'", var_name)
            string_map[var_name] = s_lit
    return fol_clean

def get_clean_question_text(question_text: str) -> str:
    """Isolate the actual question prompt by stripping answer options (A., B., C., D.) from the text.
    This prevents keywords inside option text from polluting classification."""
    if not question_text:
        return ""
    parts = re.split(r'\n(?=[A-D]\.| *\([A-D]\)| *\[[A-D]\])', question_text)
    return parts[0].strip()

def classify_mcq_strategy(question_text: str) -> str:
    """Classify the MCQ tie-breaking strategy from the *cleaned* question text.
    
    Returns one of:
      'fewest_premises'      – pick the option provable with the smallest unsat core
      'cannot_be_concluded'  – pick the option that is NOT provable
      'strongest'            – (default) pick the option with the deepest reasoning chain
    """
    if not question_text:
        return "strongest"
    q_clean = get_clean_question_text(question_text)
    q_lower = q_clean.lower()
    
    # --- "cannot be concluded" family ---
    if any(kw in q_lower for kw in [
        "cannot be concluded", "cannot be inferred", "not follow",
        "not be concluded", "not be inferred", "cannot follow",
    ]):
        return "cannot_be_concluded"
    
    # --- "fewest premises" family ---
    if "fewest" in q_lower:
        return "fewest_premises"
    # "least" but NOT "at least" (existential quantifier phrasing)
    if "least" in q_lower and not re.search(r'\bat\s+least\b', q_lower):
        return "fewest_premises"
    
    return "strongest"

class FOLASTTransformer(ast.NodeTransformer):
    """AST Transformer to map dynamic functions to Z3 equivalents with arity information."""
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in {'ForAll', 'Exists', 'Implies', 'And', 'Or', 'Not', 'Iff', 'Power'}:
                arity = len(node.args)
                node.func.id = f'{func_name}_arity_{arity}'
        return node

def run_single_z3_verification(premises: List[str], conclusion_fol: Union[str, Dict[str, str]], timeout_ms: int = 2000, llm_answer: str = None, question_text: str = None) -> str:
    """
    Convert FOL premises and a conclusion to Z3, solve it, and return the verified logical answer.
    
    Args:
        premises: List of normalized FOL strings.
        conclusion_fol: A string (for Yes/No) or a dict mapping options (A, B, C, D) to FOL strings (for MCQ).
        timeout_ms: Timeout limit for Z3 solver in milliseconds.
        llm_answer: Optional LLM-provided answer to trust if it is among proved options.
        question_text: Optional raw question text used for classifying the MCQ strategy.
        
    Returns:
        "Yes", "No", "Unknown", specific MCQ option (e.g. "A", "B", "C", "D"), or "Error".
    """
    all_fol_strings = list(premises)
    if isinstance(conclusion_fol, dict):
        all_fol_strings.extend(conclusion_fol.values())
    elif isinstance(conclusion_fol, str):
        all_fol_strings.append(conclusion_fol)
        
    string_map = {}
    cleaned_trees = []
    record_names = set()
    record_functions = {}
    math_e_detected = False
    function_return_sorts = {}
    
    try:
        # Step 1: Collect variable, constant, and function information through AST Walk
        for fol in all_fol_strings:
            cleaned = clean_and_register_names(fol, string_map)
            tree = ast.parse(cleaned, mode='eval')
            ast.fix_missing_locations(tree)
            transformed_tree = FOLASTTransformer().visit(tree)
            ast.fix_missing_locations(transformed_tree)
            cleaned_trees.append(transformed_tree)
            
            # AST type inference to deduce if functions return Bool or Real based on context
            for node in ast.walk(transformed_tree):
                if isinstance(node, ast.Name):
                    if node.id not in {'ForAll', 'Exists', 'Implies', 'And', 'Or', 'Not', 'Iff', 'Power'}:
                        record_names.add(node.id)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in {'ForAll', 'Exists', 'Implies', 'And', 'Or', 'Not', 'Iff', 'Power'}:
                        if '_arity_' in func_name:
                            parts = func_name.split('_arity_')
                            arity = int(parts[-1])
                            record_functions[func_name] = arity
                        else:
                            record_functions[func_name] = len(node.args)
                        
                        if func_name not in function_return_sorts:
                            function_return_sorts[func_name] = _z3.BoolSort()
                    elif func_name == 'Power':
                        if isinstance(node.args[0], ast.Name) and node.args[0].id == 'e':
                            math_e_detected = True
                            
                # If a function appears in a numeric comparison context or is compared to a numeric literal
                elif isinstance(node, ast.Compare):
                    is_numeric_op = any(isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)) for op in node.ops)
                    if not is_numeric_op:
                        all_sides = [node.left] + node.comparators
                        has_number = any(
                            isinstance(side, (ast.Num, ast.Constant)) and isinstance(getattr(side, 'n', getattr(side, 'value', None)), (int, float))
                            for side in all_sides
                        )
                        if has_number:
                            is_numeric_op = True
                            
                    if is_numeric_op:
                        for subnode in ast.walk(node):
                            if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                                f_name = subnode.func.id
                                function_return_sorts[f_name] = UniversalSort
                                
                # If a function is involved in an arithmetic operation
                elif isinstance(node, ast.BinOp):
                    if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)):
                        for subnode in [node.left, node.right]:
                            for call_node in ast.walk(subnode):
                                if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
                                    f_name = call_node.func.id
                                    function_return_sorts[f_name] = UniversalSort
                                    
                # If a function appears as an argument to Power
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Power':
                    for arg in node.args:
                        for call_node in ast.walk(arg):
                            if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
                                f_name = call_node.func.id
                                function_return_sorts[f_name] = UniversalSort
                            
        # Step 2: Set up a namespace that preserves the inferred types for Z3 execution
        ns = {
            'ForAll': CustomForAll,
            'Exists': CustomExists,
            'Implies': CustomImplies,
            'And': CustomAnd,
            'Or': CustomOr,
            'Not': CustomNot,
            'Iff': CustomIff,
            'Power': _z3.Function('Power', _z3.RealSort(), _z3.RealSort(), _z3.RealSort())
        }
        
        # Identify predicates (functions returning Bool) and functions (returning Real)
        for func_name, arity in record_functions.items():
            arg_sorts = [UniversalSort] * arity
            ret_sort = function_return_sorts.get(func_name, _z3.BoolSort())
            ns[func_name] = _z3.Function(func_name, *arg_sorts, ret_sort)
        
        # Identify constants and entities
        for name in record_names:
            if name in ns: 
                continue
            if name == 'e' and math_e_detected:
                ns[name] = _z3.RealVal(2.71828)
            elif name.lower() == 'pi':
                ns[name] = _z3.RealVal(3.14159)
            elif name.isdigit():
                ns[name] = _z3.RealVal(int(name))
            elif name in record_functions:
                continue
            else:
                mapped_name = string_map.get(name, name)
                ns[name] = _z3.Const(mapped_name, UniversalSort)
                
        # Step 3: Compile premises into Z3 expressions
        premises_exprs = []
        for fol in premises:
            cleaned = clean_and_register_names(fol, string_map)
            tree = ast.parse(cleaned, mode='eval')
            ast.fix_missing_locations(tree)
            transformed_tree = FOLASTTransformer().visit(tree)
            ast.fix_missing_locations(transformed_tree)
            code = compile(transformed_tree, filename='<ast>', mode='eval')
            res = eval(code, {'__builtins__': {}}, ns)
            premises_exprs.append(coerce_to_bool(res))
            
        # Step 4: Setup Solver and Evaluate
        base_solver = _z3.Solver()
        base_solver.set("timeout", timeout_ms)
        for prem_expr in premises_exprs:
            base_solver.add(prem_expr)
            
        # Check the consistency of the premises first
        if base_solver.check() == _z3.unsat:
            return "Error: Inconsistent Premises"
            
        # Case A: Multiple Choice Question with options in a dictionary
        if isinstance(conclusion_fol, dict):
            # Create a tracking solver to identify minimal premise usage via Unsat Core
            tracking_solver = _z3.Solver()
            tracking_solver.set("timeout", timeout_ms)
            tracking_literals = []
            
            for idx, prem_expr in enumerate(premises_exprs):
                p = _z3.Bool(f'p_{idx}')
                tracking_literals.append(p)
                tracking_solver.add(_z3.Implies(p, prem_expr))
                
            proved_options_cores = {}
            opt_exprs = {}
            for opt, opt_fol in conclusion_fol.items():
                cleaned_opt = clean_and_register_names(opt_fol, string_map)
                tree = ast.parse(cleaned_opt, mode='eval')
                ast.fix_missing_locations(tree)
                transformed_tree = FOLASTTransformer().visit(tree)
                ast.fix_missing_locations(transformed_tree)
                code = compile(transformed_tree, filename='<ast>', mode='eval')
                opt_expr = coerce_to_bool(eval(code, {'__builtins__': {}}, ns))
                opt_exprs[opt] = opt_expr
                
                tracking_solver.push()
                tracking_solver.add(_z3.Not(opt_expr))
                res = tracking_solver.check(*tracking_literals)
                
                if res == _z3.unsat:
                    core = tracking_solver.unsat_core()
                    proved_options_cores[opt] = {str(x) for x in core}
                tracking_solver.pop()
                
            # If the LLM's answer is explicitly provided and is among proved options, trust it
            if llm_answer is not None and llm_answer in proved_options_cores:
                return llm_answer

            # Classify the MCQ strategy from the question text
            strategy = classify_mcq_strategy(question_text) if question_text else "strongest"
            
            proved_options = list(proved_options_cores.keys())
            all_option_keys = list(conclusion_fol.keys())
            
            # ── Strategy: cannot_be_concluded ──
            # Pick the option(s) that Z3 could NOT prove from the premises.
            if strategy == "cannot_be_concluded":
                unproved = [opt for opt in all_option_keys if opt not in proved_options_cores]
                if len(unproved) == 1:
                    return unproved[0]
                elif len(unproved) > 1:
                    return unproved[0]  # fallback: first unproved
                else:
                    return "Unknown"  # all proved – unexpected for this question type
            
            # ── Common path for strongest / fewest_premises ──
            if len(proved_options) == 1:
                return proved_options[0]
            elif len(proved_options) > 1:
                # Step 1: Structural Strength Check
                scores = {opt: 0 for opt in proved_options}
                for opt_i in proved_options:
                    for opt_j in proved_options:
                        if opt_i == opt_j:
                            continue
                        expr_i = opt_exprs[opt_i]
                        expr_j = opt_exprs[opt_j]
                        struct_solver = _z3.Solver()
                        struct_solver.set("timeout", timeout_ms)
                        struct_solver.add(_z3.Not(_z3.Implies(expr_i, expr_j)))
                        if struct_solver.check() == _z3.unsat:
                            scores[opt_i] += 1
                            
                if strategy == "strongest":
                    # For strongest: prefer highest implication score
                    max_score = max(scores.values())
                    best_by_strength = [opt for opt, score in scores.items() if score == max_score]
                else:
                    # For fewest_premises: prefer lowest implication score (shallowest)
                    min_score = min(scores.values())
                    best_by_strength = [opt for opt, score in scores.items() if score == min_score]
                
                if len(best_by_strength) == 1:
                    return best_by_strength[0]
                else:
                    # Step 2: Unsat Core Subset Inclusion Check
                    best_by_inclusion = []
                    has_any_subset_relation = False
                    for opt_i in proved_options:
                        core_i = proved_options_cores[opt_i]
                        is_superset = False
                        for opt_j in proved_options:
                            if opt_i == opt_j:
                                continue
                            core_j = proved_options_cores[opt_j]
                            if core_j.issubset(core_i) and core_j != core_i:
                                is_superset = True
                                has_any_subset_relation = True
                                break
                        if strategy == "strongest":
                            # Keep supersets (deeper chains)
                            if is_superset:
                                best_by_inclusion.append(opt_i)
                        else:
                            # fewest_premises: keep subsets (shallower chains)
                            if not is_superset:
                                best_by_inclusion.append(opt_i)
                    
                    # Only apply inclusion filter when at least 1 subset pair exists
                    # AND the filter does not eliminate everything
                    if has_any_subset_relation and len(best_by_inclusion) > 0:
                        proved_options = best_by_inclusion
                        
                    # Step 3: Core Size Fallback
                    if strategy == "strongest":
                        target_size = max(len(proved_options_cores[opt]) for opt in proved_options)
                    else:
                        target_size = min(len(proved_options_cores[opt]) for opt in proved_options)
                    best_by_core = [opt for opt in proved_options if len(proved_options_cores[opt]) == target_size]
                    
                    if len(best_by_core) == 1:
                        return best_by_core[0]
                    else:
                        return "Unknown"
            else:
                return "Unknown"
                
        # Case B: Yes/No Question with a single conclusion string
        elif isinstance(conclusion_fol, str):
            cleaned_conc = clean_and_register_names(conclusion_fol, string_map)
            tree = ast.parse(cleaned_conc, mode='eval')
            ast.fix_missing_locations(tree)
            transformed_tree = FOLASTTransformer().visit(tree)
            ast.fix_missing_locations(transformed_tree)
            code = compile(transformed_tree, filename='<ast>', mode='eval')
            conc_expr = coerce_to_bool(eval(code, {'__builtins__': {}}, ns))
            
            # Try to prove conc_expr is true by showing that Not(conc_expr) leads to unsat
            base_solver.push()
            base_solver.add(_z3.Not(conc_expr))
            res_yes = base_solver.check()
            base_solver.pop()
            
            if res_yes == _z3.unsat:
                return "Yes"
            else:
                # Try to prove Not(conc_expr) is true by showing that conc_expr leads to unsat
                base_solver.push()
                base_solver.add(conc_expr)
                res_no = base_solver.check()
                base_solver.pop()
                
                if res_no == _z3.unsat:
                    return "No"
                else:
                    return "Unknown"
                    
    except Exception as e:
        print(f"Error in Z3 verification: {e}")
        return "Error"
