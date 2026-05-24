import re
import ast
import z3 as _z3
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

def clean_and_register_names(fol_str: str, string_map: Dict[str, str]) -> str:
    """Standardize string formatting, handle multi-word predicates, and extract string literals."""
    # Join multi-word predicates with _ (e.g., passed science assessment(x) -> passed_science_assessment(x))
    fol_clean = re.sub(r'\b([a-z]+(?:\s+[a-z]+)+)\s*\(', lambda m: m.group(1).replace(' ', '_') + '(', fol_str)
    fol_clean = fol_clean.strip().rstrip('.$')
    fol_clean = fol_clean.replace("^", "**")
    
    string_literals = re.findall(r"'([^']*)'", fol_clean)
    for idx_lit, s_lit in enumerate(string_literals):
        var_name = f'str_lit_{idx_lit}'
        fol_clean = fol_clean.replace(f"'{s_lit}'", var_name)
        string_map[var_name] = s_lit
        
    return fol_clean

class FOLASTTransformer(ast.NodeTransformer):
    """AST Transformer to map Python logic operators and dynamic functions to Z3 equivalents."""
    def visit_Compare(self, node):
        if isinstance(node.left, ast.BinOp) and isinstance(node.left.op, (ast.BitAnd, ast.BitOr)):
            binop = node.left
            new_compare = ast.Compare(
                left=binop.right,
                ops=node.ops,
                comparators=node.comparators
            )
            ast.copy_location(new_compare, node)
            func_name = 'And' if isinstance(binop.op, ast.BitAnd) else 'Or'
            call_node = ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=[binop.left, new_compare],
                keywords=[]
            )
            ast.copy_location(call_node, node)
            return self.visit(call_node)
        
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            return ast.Call(
                func=ast.Name(id='Power', ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
        elif isinstance(node.op, ast.BitAnd):
            return ast.Call(
                func=ast.Name(id='And', ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
        elif isinstance(node.op, ast.BitOr):
            return ast.Call(
                func=ast.Name(id='Or', ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[]
            )
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in {'ForAll', 'Exists', 'Implies', 'And', 'Or', 'Not', 'Iff', 'Power'}:
                arity = len(node.args)
                node.func.id = f'{func_name}_arity_{arity}'
        return node

def run_single_z3_verification(premises: List[str], conclusion_fol: Union[str, Dict[str, str]], timeout_ms: int = 2000) -> str:
    """
    Convert FOL premises and a conclusion to Z3, solve it, and return the verified logical answer.
    
    Args:
        premises: List of normalized FOL strings.
        conclusion_fol: A string (for Yes/No) or a dict mapping options (A, B, C, D) to FOL strings (for MCQ).
        timeout_ms: Timeout limit for Z3 solver in milliseconds.
        
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
            proved_options = []
            for opt, opt_fol in conclusion_fol.items():
                cleaned_opt = clean_and_register_names(opt_fol, string_map)
                tree = ast.parse(cleaned_opt, mode='eval')
                ast.fix_missing_locations(tree)
                transformed_tree = FOLASTTransformer().visit(tree)
                ast.fix_missing_locations(transformed_tree)
                code = compile(transformed_tree, filename='<ast>', mode='eval')
                opt_expr = coerce_to_bool(eval(code, {'__builtins__': {}}, ns))
                
                # Prove opt_expr is true by showing that Not(opt_expr) leads to unsat
                base_solver.push()
                base_solver.add(_z3.Not(opt_expr))
                res = base_solver.check()
                base_solver.pop()
                
                if res == _z3.unsat:
                    proved_options.append(opt)
            if len(proved_options) == 1:
                return proved_options[0]
            elif len(proved_options) > 1:
                return f"Error: Multiple True Options ({', '.join(proved_options)})"
            else:
                return "Unknown"  # "Cannot Prove Any Option" -> return Unknown
                
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
