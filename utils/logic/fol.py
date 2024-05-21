from lark import Lark, Transformer
import sys

### FOL classes ### 

class Constant:
    def __init__(self, name):
        self.name = name

    def __repr__(self) -> str:
        return self.name


class Variable:
    def __init__(self, name):
        self.name = name

    def __repr__(self) -> str:
        return self.name


class Predicate:
    def __init__(self, name, *args):
        self.name = name
        self.args = args

    def __repr__(self) -> str:
        args = ", ".join(map(str, self.args))

        return f"{self.name}({args})"


class Function:
    def __init__(self, name, *args):
        self.name = name
        self.args = args

    def __repr__(self) -> str:
        args = ", ".join(map(str, self.args))

        return f"{self.name}({args})"


class Not:
    def __init__(self, arg):
        self.arg = arg

    def __repr__(self) -> str:
        return f"~{str(self.arg)}"


class And:
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __repr__(self) -> str:
        return f"{str(self.arg1)} && {str(self.arg2)}"


class Or:
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __repr__(self) -> str:
        return f"{str(self.arg1)} || {str(self.arg2)}"


class Implies:
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self) -> str:
        return f"{str(self.lhs)} => {str(self.rhs)}"


class ForAll:
    def __init__(self, var, body):
        self.var = var
        self.body = body

    def __repr__(self) -> str:
        return f"FOR_ALL {self.var}, {str(self.body)}"


class Exists:
    def __init__(self, var, body):
        self.var = var
        self.body = body

    def __repr__(self) -> str:
        return f"EXISTS {self.var}, {str(self.body)}"

### FOL parser ### 

class FOLTransformer(Transformer):
    def CONSTANT(self, c):
        return Constant(c)

    def VARIABLE(self, v):
        return Variable(v)

    def function(self, children):
        name = children[0].value
        args = children[1:]

        return Function(name, *args)

    def predicate(self, children):
        name = children[0].value
        args = children[1:]

        return Predicate(name, *args)

    def negation(self, children):
        arg = children[-1]

        return Not(arg)

    def conjunction(self, children):
        arg1, arg2 = children[0], children[-1]

        return And(arg1, arg2)

    def disjunction(self, children):
        arg1, arg2 = children[0], children[-1]

        return Or(arg1, arg2)

    def implication(self, children):
        lhs, rhs = children[0], children[-1]

        return Implies(lhs, rhs)

    def quantifier(self, children):
        quantifier = children[0].value
        var = children[1]
        body = children[2]

        if quantifier == "FOR_ALL":
            return ForAll(var, body)
        else:
            return Exists(var, body)

with open("utils/logic/fol.lark", "r") as f:
    parser = Lark(f, start="sentence")

transformer = FOLTransformer()


def parse_fol(s):
    tree = parser.parse(s)
    fol = transformer.transform(tree)

    return tree, fol

### FOL functions ### 

def and_list(conjuncts):
    result = None

    for conjunct in conjuncts:
        if result:
            result = And(result, conjunct)
        else:
            result = conjunct 
    
    return result

def or_list(disjuncts):
    result = None

    for disjunct in disjuncts:
        if result:
            result = Or(result, disjunct)
        else:
            result = disjunct 
    
    return result

def flatten_and(sentence):
    if not isinstance(sentence, And):
        return [sentence]

    return flatten_and(sentence.arg1) + flatten_and(sentence.arg2)

def flatten_or(sentence):
    if not isinstance(sentence, Or):
        return [sentence]

    return flatten_or(sentence.arg1) + flatten_or(sentence.arg2)



if __name__ == "__main__":
    #s = "FOR_ALL x, meat_pasta(x) && watching_weight(x) && ~spicy(x) => likes(U, x)"
    s = "FOR_ALL x, meat_pasta(x) && watching_weight(x) && ~spicy(x) => likes(U, x)"
    tree, fol = parse_fol(s)
    
    print(isinstance(fol.body.lhs, And))
    sys.exit()


    print(tree.pretty())
    print(type(fol.body.rhs.args[1]))
    print()

    print(type(fol))
    print(f"{type(fol.var)} -> {fol.var}")
    print(f"{type(fol.body)} -> {fol.body}")
    print(f"{type(fol.body.lhs)} -> {fol.body.lhs}")
    print(f"{type(fol.body.lhs.arg1)} -> {fol.body.lhs.arg1}")
    # you can keep checking the types and their contents...  
    
    # alternatively, define a list of conjuncts 
    print("\nalternative:")
    conjuncts = [
        Predicate("meat_pasta", Variable("x")),
        Predicate("watching_weight", Variable("x")),
        Not(Predicate("spicy", Constant("r1")))
    ]
    ands = and_list(conjuncts)
    flatenned_list = flatten_and(ands)
    print(flatenned_list[0].name)
    print(flatenned_list[0].args[0])
    print(type(flatenned_list[-1].arg.args[0]))
    print(isinstance(flatenned_list[-1].arg.args[0], Constant))
    print((Not(Predicate("spicy", Variable("x")))).arg.name)
    print(Predicate("spicy", Variable("x")).args)
    print(Predicate("spicy", Constant("r1")).args)