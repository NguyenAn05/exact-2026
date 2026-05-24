import re
from typing import List


class SimpleUniversalNormalizer:
    """Normalize FOL using safe string pre-processing + contextual pattern matching.

    Converts various FOL notation styles into Z3-compatible prefix syntax:
        - ForAll x (P(x) → Q(x))  →  ForAll([x], Implies(P(x), Q(x)))
        - ∀x (P(x) ∧ Q(x))        →  ForAll([x], And(P(x), Q(x)))
        - Exists m1, Exists m2, R  →  Exists([m1], Exists([m2], R))
        - Exists(x, P(x)) → Q     →  Implies(Exists([x], P(x)), Q)
        - ForAll x(P(x)→Q(x))     →  ForAll([x], Implies(P(x), Q(x)))
        - A → B → C               →  Implies(A, Implies(B, C))
    """

    MAX_DEPTH = 50
    VAR_PATTERN = r'[a-zA-Z]\w*'

    def __init__(self):
        self.operator_map = {
            '⇒': '→', '⊃': '→',
            '⇔': '↔',
        }
        self.comp_map = {
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
        }

    def normalize(self, fol_str: str, _depth: int = 0) -> str:
        if not fol_str or not fol_str.strip():
            return fol_str

        if _depth > self.MAX_DEPTH:
            return fol_str

        try:
            s = fol_str.strip()

            # Step 1: Replace various symbols and notations with standardized forms before deeper parsing
            s = self._replace_symbols(s)
            s = self._fix_prefix_comparisons(s)
            s = self._fix_argument_comparisons(s)
            s = self._normalize_arrow_spacing(s)

            # Step 2: If already in normalized prefix format, just ensure it's fully normalized inside
            if self._is_normalized(s):
                open_idx = s.index('(')
                close_idx = self._matching_paren(s, open_idx)
                if close_idx == len(s) - 1:
                    return self._normalize_prefix_format(s, _depth)

            # Step 3: If it's a simple ground atom, return as is
            if self._is_ground_atom(s):
                return s

            # Step 4: Process the body structure (Handle operators from lowest to highest precedence)
            return self._normalize_body(s, _depth)

        except Exception as e:
            print(f"Warning: Normalization failed for '{fol_str}': {e}")
            return fol_str

    # Symbol replacement

    def _replace_set_membership(self, s: str) -> str:
        pattern = r'([A-Za-z_]\w*(?:\([^)]*\))?)\s*(?:∈|\bin\b)\s*\{([^}]+)\}'
        def repl(match):
            left = match.group(1).strip()
            elems = [e.strip() for e in match.group(2).split(',')]
            eqs = [f"{left} == {e}" for e in elems]  # Z3 chuẩn sử dụng ==
            return " ( " + " ∨ ".join(eqs) + " ) "
        return re.sub(pattern, repl, s)

    def _replace_symbols(self, s: str) -> str:
        s = self._replace_set_membership(s)
        s = re.sub(r'\bpass\b', 'passed', s)
        s = re.sub(r'∀\s*', 'ForAll ', s)
        s = re.sub(r'∃\s*', 'Exists ', s)
        s = re.sub(r'-\s*>', '→', s)

        for old, new in self.operator_map.items():
            s = s.replace(old, new)

        s = re.sub(r'\bimplies\b', '→', s)
        s = re.sub(r'\bnot\b\s*', '¬', s)
        s = re.sub(r'~(?=\s*[A-Za-z(¬])', '¬', s)
        s = re.sub(r'\s*&\s*', ' ∧ ', s)
        
        # Sửa lỗi Regex của toán tử '|' để không nhận diện sai ký tự ống dẫn trong toán tử khác
        s = re.sub(r'(?<=[\s)])\s*\|\s*(?=[\s(])', ' ∨ ', s)
        s = re.sub(r'(?<=\))\s*\|\s*', ' ∨ ', s)

        for old, new in self.comp_map.items():
            s = s.replace(old, new)

        return s

    def _normalize_arrow_spacing(self, s: str) -> str:
        s = re.sub(r'(?<!\s)→', ' →', s)
        s = re.sub(r'→(?!\s)', '→ ', s)
        return s

    # Detection methods

    def _is_normalized(self, s: str) -> bool:
        return bool(re.match(r'^(ForAll|Exists)\s*\(', s))

    def _starts_with_quantifier(self, s: str) -> bool:
        return s.startswith('ForAll ') or s.startswith('Exists ')

    def _is_ground_atom(self, s: str) -> bool:
        has_quant = bool(re.search(r'\b(ForAll|Exists)\b', s))
        has_logic_op = bool(re.search(r'[→∧∨¬↔]', s))
        has_comp = bool(re.search(r'(≠|≤|≥|==|!=|<=|>=|(?<![<>=!])[<>=!])', s))
        return not (has_quant or has_logic_op or has_comp)

    # Prefix format normalization

    def _normalize_prefix_format(self, s: str, _depth: int = 0) -> str:
        var_pat = self.VAR_PATTERN

        # Pattern 1: ForAll([x], body)
        match1 = re.match(rf'^(ForAll|Exists)\s*\(\s*\[({var_pat})\]\s*,\s*', s)
        if match1:
            quant = match1.group(1)
            var = match1.group(2)
            body = s[match1.end():-1].strip() # Cắt an toàn đến trước dấu ngoặc đóng cuối cùng
            normalized_body = self._normalize_body(body, _depth + 1)
            return f"{quant}([{var}], {normalized_body})"

        # Pattern 2: ForAll(x, body)
        match2 = re.match(rf'^(ForAll|Exists)\s*\(\s*({var_pat})\s*,\s*', s)
        if match2:
            quant = match2.group(1)
            var = match2.group(2)
            body = s[match2.end():-1].strip()
            normalized_body = self._normalize_body(body, _depth + 1)
            return f"{quant}([{var}], {normalized_body})"

        return s

    # Quantifier normalization

    def _normalize_quantifiers(self, s: str, _depth: int = 0) -> str:
        """Normalize quantifier structures from infix to Z3 prefix format"""
        if _depth > self.MAX_DEPTH:
            return s

        s = s.strip()
        var_pat = self.VAR_PATTERN

        # Pattern A: Handle comma-separated quantifiers: ForAll x, Exists y, body
        match_comma = re.match(rf'^(ForAll|Exists)\s+({var_pat})\s*,\s*(.+)$', s)
        if match_comma:
            quant, var, rest = match_comma.groups()
            rest = rest.strip()
            if self._starts_with_quantifier(rest):
                return f"{quant}([{var}], {self._normalize_quantifiers(rest, _depth + 1)})"
            return f"{quant}([{var}], {self._normalize_body(rest, _depth + 1)})"

        # Pattern B: Handle ForAll x(body) - Use accurate parenthesis matching function
        match_paren = re.match(rf'^(ForAll|Exists)\s+({var_pat})\(', s)
        if match_paren:
            quant, var = match_paren.group(1), match_paren.group(2)
            open_idx = match_paren.end() - 1
            close_idx = self._matching_paren(s, open_idx)
            if close_idx > 0:
                body = s[open_idx + 1:close_idx].strip()
                rest_after = s[close_idx + 1:].strip()
                normalized_body = self._normalize_body(body, _depth + 1)
                quant_part = f"{quant}([{var}], {normalized_body})"
                if not rest_after:
                    return quant_part

                return self._normalize_body(f"{quant_part} {rest_after}", _depth + 1)

        # Pattern C: Handle space-separated quantifiers: ForAll x body
        match_space = re.match(rf'^(ForAll|Exists)\s+({var_pat})\s+(.+)$', s)
        if match_space:
            quant, var, body = match_space.groups()
            body = body.strip()
            if self._starts_with_quantifier(body):
                return f"{quant}([{var}], {self._normalize_quantifiers(body, _depth + 1)})"
            return f"{quant}([{var}], {self._normalize_body(body, _depth + 1)})"

        return s

    # Body normalization

    def _normalize_body(self, s: str, _depth: int = 0) -> str:
        if _depth > self.MAX_DEPTH:
            return s

        s = s.strip()
        if not s:
            return s

        s = self._strip_outer_parens(s)

        # If the body starts with a quantifier, normalize it first to ensure correct handling of nested structures
        if self._starts_with_quantifier(s):
            return self._normalize_quantifiers(s, _depth + 1)
        
        if self._is_normalized(s):
            open_idx = s.index('(')
            close_idx = self._matching_paren(s, open_idx)
            if close_idx == len(s) - 1:
                return self._normalize_prefix_format(s, _depth + 1)

        # Biconditional (Priority: Lowest) - Handle first to avoid splitting implications inside biconditionals
        parts = self._split_by_op(s, '↔')
        if len(parts) == 2:
            left = self._normalize_body(parts[0], _depth + 1)
            right = self._normalize_body(parts[1], _depth + 1)
            return f"Iff({left}, {right})"

        # Implication (Right-associative) - Split only on the first top-level -> to preserve right-associativity
        parts = self._split_by_op(s, '→')
        if len(parts) >= 2:
            left = self._normalize_body(parts[0], _depth + 1)
            right_str = ' → '.join(parts[1:])
            right = self._normalize_body(right_str, _depth + 1)
            return f"Implies({left}, {right})"

        # Disjunction ∨
        parts = self._split_by_op(s, '∨')
        if len(parts) > 1:
            normalized = [self._normalize_body(p, _depth + 1) for p in parts]
            return f"Or({', '.join(normalized)})"

        # Conjunction ∧
        parts = self._split_by_op(s, '∧')
        if len(parts) > 1:
            normalized = [self._normalize_body(p, _depth + 1) for p in parts]
            return f"And({', '.join(normalized)})"

        # Negation ¬ 
        neg_match = re.match(r'^¬\s*(.+)$', s)
        if neg_match:
            operand = neg_match.group(1).strip()
            return self._normalize_negation(operand, _depth)

        # Structures like Implies(...), And(...), Or(...), Not(...), Iff(...) - Just normalize their arguments
        func_match = re.match(r'^(Implies|And|Or|Not|Iff)\s*\(', s)
        if func_match:
            func_name = func_match.group(1)
            open_idx = func_match.end() - 1
            close_idx = self._matching_paren(s, open_idx)
            if close_idx == len(s) - 1:
                args_str = s[open_idx + 1:close_idx]
                args = self._split_args(args_str)
                normalized_args = [self._normalize_body(a.strip(), _depth + 1) for a in args]
                return f"{func_name}({', '.join(normalized_args)})"

        # Operators: ==, !=, <=, >=, <, > (Keep as is for Z3 compatibility)
        comp_match = re.match(r'^(.+?)\s*(==|!=|<=|>=|<|>|=)\s*(.+)$', s)
        if comp_match:
            left = comp_match.group(1).strip()
            op = comp_match.group(2)
            right = comp_match.group(3).strip()
            if op == '=':
                op = '=='
            return f"{left} {op} {right}"

        return s

    def _normalize_negation(self, operand: str, _depth: int) -> str:
        if re.match(r'^(ForAll|Exists)\s*\(', operand):
            return f"Not({self._normalize_body(operand, _depth + 1)})"
        if self._starts_with_quantifier(operand):
            return f"Not({self._normalize_quantifiers(operand, _depth + 1)})"

        if operand.startswith('('):
            close = self._matching_paren(operand, 0)
            if close == len(operand) - 1:
                inner = operand[1:-1].strip()
                return f"Not({self._normalize_body(inner, _depth + 1)})"
            elif close > 0:
                neg_part = operand[1:close].strip()
                rest = operand[close + 1:].strip()
                # Phối hợp ngược lại cấu trúc tổng để tránh mất dữ liệu phía sau dấu ngoặc đóng
                return self._normalize_body(f"Not({self._normalize_body(neg_part, _depth + 1)}) {rest}", _depth + 1)
        
        return f"Not({self._normalize_body(operand, _depth + 1)})"

    # Utility methods

    def _split_by_op(self, s: str, op: str) -> List[str]:
        parts: List[str] = []
        current: List[str] = []
        depth = 0
        i = 0

        while i < len(s):
            char = s[i]
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif depth == 0 and s[i:i + len(op)] == op:
                parts.append(''.join(current).strip())
                current = []
                i += len(op)
                continue
            else:
                current.append(char)
            i += 1

        if current:
            parts.append(''.join(current).strip())

        return parts if len(parts) > 1 else [s]

    def _split_args(self, s: str) -> List[str]:
        args: List[str] = []
        current: List[str] = []
        depth = 0

        for char in s:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            args.append(''.join(current).strip())
        return args

    def _matching_paren(self, s: str, start: int) -> int:
        depth = 1
        i = start + 1
        while i < len(s) and depth > 0:
            if s[i] == '(':
                depth += 1
            elif s[i] == ')':
                depth -= 1
            i += 1
        return (i - 1) if depth == 0 else -1

    def _strip_outer_parens(self, s: str) -> str:
        while s.startswith('(') and s.endswith(')'):
            if self._matching_paren(s, 0) == len(s) - 1:
                s = s[1:-1].strip()
            else:
                break
        return s

    def _fix_prefix_comparisons(self, s: str) -> str:
        """Convert '>= (A, B)' to 'A >= B' safely."""
        ops = ["==", "!=", ">=", "<=", ">", "<"]
        i = 0
        while i < len(s):
            found_op = None
            for op in ops:
                if s[i:].startswith(op + "(") or s[i:].startswith(op + " ("):
                    found_op = op
                    break
            
            if found_op:
                if i > 0 and (s[i-1].isalnum() or s[i-1] in {'_', '>', '<', '=', '!'}):
                    i += 1
                    continue
                    
                op_len = len(found_op)
                open_idx = i + op_len + (1 if s[i + op_len] == ' ' else 0)
                close_idx = self._matching_paren(s, open_idx)
                
                if close_idx > 0:
                    args_str = s[open_idx + 1:close_idx]
                    args = self._split_args(args_str)
                    if len(args) == 2:
                        infix = f"{args[0]} {found_op} {args[1]}"
                        s = s[:i] + infix + s[close_idx + 1:]
                        i += len(infix)
                        continue
            i += 1
        return s

    def _fix_argument_comparisons(self, s: str) -> str:
        """Convert 'Studies(x, y, >= 10)' to 'Studies(x, y) >= 10' safely."""
        i = 0
        while i < len(s):
            if i == 0 or not (s[i-1].isalnum() or s[i-1] == '_'):
                match = re.match(r'^([A-Za-z_]\w*)\(', s[i:])
                if match:
                    func_name = match.group(1)
                    if func_name not in {"ForAll", "Exists", "Implies", "And", "Or", "Not", "Iff"}:
                        open_idx = i + match.end() - 1
                        close_idx = self._matching_paren(s, open_idx)
                        if close_idx > 0:
                            args_str = s[open_idx + 1:close_idx]
                            args = self._split_args(args_str)
                            
                            if args:
                                last_arg = args[-1]
                                comp_match = re.match(r'^(>=|<=|==|!=|>|<)\s*(.+)$', last_arg)
                                if comp_match:
                                    op = comp_match.group(1)
                                    val = comp_match.group(2).strip()
                                    remaining_args = args[:-1]
                                    new_call = f"{func_name}({', '.join(remaining_args)}) {op} {val}"
                                    s = s[:i] + new_call + s[close_idx + 1:]
                                    i += len(new_call)
                                    continue
            i += 1
        return s