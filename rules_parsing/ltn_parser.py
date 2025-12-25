from rules_parser import RulesParserInterface

class LTNParser(RulesParserInterface):
    def parse_expression(self, expr):
        op = expr.get("operator")
        name = expr.get("name")

        if op is None and name:
            return f"{name}(x)"

        if op == "NOT":
            return f"Not({name}(x))"

        if op == "AND" or op == "OR":
            parsed_arg1 = self.parse_expression(expr["arg1"])
            parsed_arg2 = self.parse_expression(expr["arg2"])
            return f"{op.lower().capitalize()}({parsed_arg1}, {parsed_arg2})"

        raise ValueError(f"Unsupported operator: {op}")

    def post_process_conclusion(self, s): # TODO: This is a hack - to fix and remove
        if s.endswith("))"):
            return s.replace("))", ",f(x)))")
        else:
            return s.replace(")", ",f(x))")

    def parse_rule(self, rule):
        premise = self.parse_expression(rule["if_part"])
        conclusion = self.parse_expression(rule["then_part"])
        return f"Implies({premise}, {self.post_process_conclusion(conclusion)})"
    
    