import pytest
import os
from itertools import combinations, permutations, product
from cspsat import *
from .alldefs import *

def norm(f):
    if f is None or isinstance(f, (int,str,Bool,Var)):
        return f
    if isinstance(f, dict):
        f = [ (k,norm(g)) for (k,g) in f.items() ]
    else:
        f = [ norm(g) for g in iter(f) ]
    return tuple(sorted(f))

def verify(csp, encoder=None, vars=[], tester=None, num=0, k=None):
    sols = list(solutionsCSP(csp, encoder=encoder, num=num))
    if k is not None and len(sols) != k:
        print(f"# num of solutions {len(sols)} != expected {k} in {csp}")
        return False
    for sol in sols:
        values = [ sol[v] for v in vars ]
        if not tester(*values):
            print(f"# values {values} not satisfy {csp}")
            return False
    return True

def test_var_wsum():
    (x, _y) = (Var("x"), Var())
    assert str(x) == "x"
    assert str(x(1)) == "x(1)"
    assert str(x(1,2)) == "x(1,2)"
    assert str(_y).startswith("?")
    assert not x.isAux()
    assert _y.isAux()
    wsum = Wsum(x(1)).mul(2).add(x(2)).add(3)
    assert wsum.toExpr() == ['+', ['*', x(1), 2], ['*', x(2), 1], 3]

def test_basic_constraints():
    (p, x) = (Bool("p"), Var("x"))
    xx = [ x(i) for i in range(10) ]
    csp = [ ["int", x, -2, 2] for x in xx[:2] ]
    for e in ["d", "o", "l"]:
        assert verify([*csp, ["==", x(0), x(1)]], encoder=e, k= 5, vars=xx[:2], tester=lambda x0,x1: x0 == x1)
        assert verify([*csp, ["!=", x(0), x(1)]], encoder=e, k=20, vars=xx[:2], tester=lambda x0,x1: x0 != x1)
        assert verify([*csp, [">=", x(0), x(1)]], encoder=e, k=15, vars=xx[:2], tester=lambda x0,x1: x0 >= x1)
        assert verify([*csp, [">" , x(0), x(1)]], encoder=e, k=10, vars=xx[:2], tester=lambda x0,x1: x0 >  x1)
        assert verify([*csp, ["<=", x(0), x(1)]], encoder=e, k=15, vars=xx[:2], tester=lambda x0,x1: x0 <= x1)
        assert verify([*csp, ["<" , x(0), x(1)]], encoder=e, k=10, vars=xx[:2], tester=lambda x0,x1: x0 <  x1)
        assert verify([*csp, ["==", x(0), ["-", x(1)]]], encoder=e, k=5, vars=xx[:2], tester=lambda x0,x1: x0 == -x1)
        assert verify([*csp, ["==", x(0), ["+", x(1), 2]]], encoder=e, k=3, vars=xx[:2], tester=lambda x0,x1: x0 == x1+2)
        assert verify([*csp, ["==", x(0), ["-", x(1), 2]]], encoder=e, k=3, vars=xx[:2], tester=lambda x0,x1: x0 == x1-2)
        assert verify([*csp, ["==", x(0), ["*", x(1), 2]]], encoder=e, k=3, vars=xx[:2], tester=lambda x0,x1: x0 == x1*2)
        assert verify([*csp, ["==", x(0), ["div", x(1), 2]]], encoder=e, k=5, vars=xx[:2], tester=lambda x0,x1: x0 == x1//2)
        assert verify([*csp, ["==", x(0), ["mod", x(1), 2]]], encoder=e, k=5, vars=xx[:2], tester=lambda x0,x1: x0 == x1%2)
        assert verify([*csp, ["==", x(0), ["min", x(1), 0]]], encoder=e, k=5, vars=xx[:2], tester=lambda x0,x1: x0 == min(x1,0))
        assert verify([*csp, ["==", x(0), ["max", x(1), 0]]], encoder=e, k=5, vars=xx[:2], tester=lambda x0,x1: x0 == max(x1,0))
        assert verify([*csp, ["==", x(0), ["if", [">",x(1),0], 1, 0]]], encoder=e, k=5, vars=xx[:2], tester=lambda x0,x1: x0 == (1 if x1>0 else 0))
    m = 25
    assert verify([*csp, ["not", ["==", x(0), x(1)]]], k=m- 5, vars=xx[:2], tester=lambda x0,x1: not x0 == x1)
    assert verify([*csp, ["not", ["!=", x(0), x(1)]]], k=m-20, vars=xx[:2], tester=lambda x0,x1: not x0 != x1)
    assert verify([*csp, ["not", [">=", x(0), x(1)]]], k=m-15, vars=xx[:2], tester=lambda x0,x1: not x0 >= x1)
    assert verify([*csp, ["not", [">" , x(0), x(1)]]], k=m-10, vars=xx[:2], tester=lambda x0,x1: not x0 >  x1)
    assert verify([*csp, ["not", ["<=", x(0), x(1)]]], k=m-15, vars=xx[:2], tester=lambda x0,x1: not x0 <= x1)
    assert verify([*csp, ["not", ["<" , x(0), x(1)]]], k=m-10, vars=xx[:2], tester=lambda x0,x1: not x0 <  x1)

def test_cardinality_constraints():
    p = Bool("p")
    pp = [ p(i) for i in range(5) ]
    assert verify([ ["eqK", pp, 1] ], k= 5, vars=pp, tester=lambda *pp: sum(pp) == 1)
    assert verify([ ["neK", pp, 1] ], k=27, vars=pp, tester=lambda *pp: sum(pp) != 1)
    assert verify([ ["geK", pp, 4] ], k= 6, vars=pp, tester=lambda *pp: sum(pp) >= 4)
    assert verify([ ["gtK", pp, 4] ], k= 1, vars=pp, tester=lambda *pp: sum(pp) >  4)
    assert verify([ ["leK", pp, 2] ], k=16, vars=pp, tester=lambda *pp: sum(pp) <= 2)
    assert verify([ ["ltK", pp, 2] ], k= 6, vars=pp, tester=lambda *pp: sum(pp) <  2)
    m=32
    assert verify([ ["not", ["eqK", pp, 1]] ], k=m- 5, vars=pp, tester=lambda *pp: not sum(pp) == 1)
    assert verify([ ["not", ["neK", pp, 1]] ], k=m-27, vars=pp, tester=lambda *pp: not sum(pp) != 1)
    assert verify([ ["not", ["geK", pp, 4]] ], k=m- 6, vars=pp, tester=lambda *pp: not sum(pp) >= 4)
    assert verify([ ["not", ["gtK", pp, 4]] ], k=m- 1, vars=pp, tester=lambda *pp: not sum(pp) >  4)
    assert verify([ ["not", ["leK", pp, 2]] ], k=m-16, vars=pp, tester=lambda *pp: not sum(pp) <= 2)
    assert verify([ ["not", ["ltK", pp, 2]] ], k=m- 6, vars=pp, tester=lambda *pp: not sum(pp) <  2)

def test_global_constraints():
    (x, y) = (Var("x"), Var("y"))
    xx = [ x(i) for i in range(4) ]
    yy = [ y(i) for i in range(4) ]
    cx = [ ["int", x, 0, 3] for x in xx ]
    cy = [ ["int", y, 0, 3] for y in yy ]
    assert verify([*cx, ["alldifferent", *xx]], k=24, vars=xx, tester=lambda *aa: all(a != b for (a,b) in combinations(aa,2)))
    (m, n) = (4, 3)
    xx = [ x(i) for i in range(m) ]
    yy = [ y(i) for i in range(n) ]
    cx = [ ["int", x, 0, 1] for x in xx ]
    cy = [ ["int", y, 0, 1] for y in yy ]
    (mm, nn) = (1 << m, 1 << n)
    eq = nn
    ne = mm*nn - eq
    ge = nn*(nn+1)//2 * (1 << (m-n))
    lt = mm*nn - ge
    le = lt + eq
    gt = mm*nn - le
    assert verify([*cx, *cy, ["lexCmp", "==", xx, yy]], k=eq, vars=[*xx,*yy], tester=lambda *zz: tuple(zz[:m]) == tuple([*zz[m:], 0]))
    assert verify([*cx, *cy, ["lexCmp", "!=", xx, yy]], k=ne, vars=[*xx,*yy], tester=lambda *zz: tuple(zz[:m]) != tuple([*zz[m:], 0]))
    assert verify([*cx, *cy, ["lexCmp", ">=", xx, yy]], k=ge, vars=[*xx,*yy], tester=lambda *zz: tuple(zz[:m]) >= tuple([*zz[m:], 0]))
    assert verify([*cx, *cy, ["lexCmp", ">", xx, yy]], k=gt, vars=[*xx,*yy], tester=lambda *zz: tuple(zz[:m]) >  tuple([*zz[m:], 0]))
    assert verify([*cx, *cy, ["lexCmp", "<=", xx, yy]], k=le, vars=[*xx,*yy], tester=lambda *zz: tuple(zz[:m]) <= tuple([*zz[m:], 0]))
    assert verify([*cx, *cy, ["lexCmp", "<", xx, yy]], k=lt, vars=[*xx,*yy], tester=lambda *zz: tuple(zz[:m]) <  tuple([*zz[m:], 0]))

def test_binary():
    (x, y, z) = (Var("x"), Var("y"), Var("z"))
    xx = [x,y,z]
    (cx, cy, cz) = (["int", x, 0, 3], ["int", y, 0, 3], ["int", z, 0, 27])
    assert verify([cx, cy, cz, ["mulCmp", "==", x, y, z]], encoder="l", k=16, vars=(x,y,z), tester=lambda x,y,z: x*y == z)
    assert verify([cx, cz, ["powCmp", "==", x, 3, z]], encoder="l", k=4, vars=(x,z), tester=lambda x,z: x**3 == z)

def test_csp_magicSquare():
    n = 3
    s = n*(n*n+1)//2
    x = Var("x")
    xx = [ x(i,j) for i in range(n) for j in range(n) ]
    def tester(*zz):
        for i in range(n):
            if sum([ zz[n*i+j] for j in range(n) ]) != s:
                return False
        for j in range(n):
            if sum([ zz[n*i+j] for i in range(n) ]) != s:
                return False
        if sum([ zz[n*i+i] for i in range(n) ]) != s:
            return False
        if sum([ zz[n*i+(n-i-1)] for i in range(n) ]) != s:
            return False
        return True
    for e in ["d", "o", "l"]:
        csp = magicSquare(n)
        assert verify(csp, encoder=e, k=1, vars=xx, tester=tester)

def test_csp_prime():
    for e in ["d", "o", "l"]:
        assert { sol[Var("n")] for sol in solutionsCSP(prime(10), encoder=e, num=0) } == {2, 3, 5, 7}
        assert { sol[Var("n")] for sol in solutionsCSP(composite(10), encoder=e, num=0) } == {4, 6, 8, 9, 10}

def test_csp_sudoku():
    n = 9
    clues = [ [(j if i == 0 else 0) for j in range(n)] for i in range(n) ]
    x = Var("x")
    xx = [ x(i,j) for i in range(n) for j in range(n) ]
    def tester(*zz):
        def alldiff(aa):
            return all(a != b for (a,b) in combinations(aa, 2))
        for i in range(n):
            if not alldiff([ zz[n*i+j] for j in range(n) ]):
                return False
        for j in range(n):
            if not alldiff([ zz[n*i+j] for i in range(n) ]):
                return False
        for (i,j) in product(range(0,9,3), range(0,9,3)):
            if not alldiff([ zz[n*(i+a)+(j+b)] for a in range(3) for b in range(3) ]):
                return False
        return True
    for e in ["d", "o", "l"]:
        csp = sudoku(clues)
        assert verify(csp, encoder=e, num=10, k=10, vars=xx, tester=tester)

def test_cop():
    for e in ["d", "o", "l"]:
        (x, y) = (Var("x"), Var("y"))
        csp = [ ["int", x, 1, 5], ["int", y, 1, 5] ]
        cop = [ *csp, ["minimize", x] ]
        solveCSP(cop, encoder=e)
        assert status()["result"] == "MINIMUM 1"
        assert verify(cop, encoder=e, num=0, k=5, vars=[x,y], tester=lambda x,y: x == 1)
        cop = [ *csp, ["maximize", x] ]
        solveCSP(cop, encoder=e)
        assert status()["result"] == "MAXIMUM 5"
        assert verify(cop, encoder=e, num=0, k=5, vars=[x,y], tester=lambda x,y: x == 5)
        cop = [ *csp, [">", x, 5], ["maximize", x] ]
        solveCSP(cop, encoder=e)
        assert status()["result"] == "UNSATISFIABLE"

def test_save_load_csp():
    cspFile = SAT._tempfileName(".csp")
    csp = list(queenDominationOpt(8))
    print(list(solutionsCSP(csp)), file=sys.stderr) # DEBUG
    [sol1] = solutionsCSP(csp)
    stat1 = status()
    saveCSP(csp, cspFile)
    csp = loadCSP(cspFile)
    [sol2] = solutionsCSP(csp)
    stat2 = status()
    print([stat1, stat2])
    assert norm(sol1) == norm(sol2)
    os.remove(cspFile)
