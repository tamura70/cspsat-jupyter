import pytest
import sys
import os
from cspsat import *

def norm(f):
    if f is None or isinstance(f, (int,str,Bool,Var)):
        return f
    if isinstance(f, dict):
        f = [ (k,norm(g)) for (k,g) in f.items() ]
    else:
        f = [ norm(g) for g in iter(f) ]
    return tuple(sorted(f))

def binaryEq(zz, n):
    for i in range(len(zz)):
        yield [ zz[i] ] if (n>>i)&1 else [ ~zz[i] ] 

def factor(n=18446743979220271189, m=32):
    (x, y, z) = (Bool("x"), Bool("y"), Bool("z"))
    xx = [ x(i) for i in range(m) ]
    yy = [ y(i) for i in range(m) ]
    zz = [ z(i) for i in range(2*m) ]
    binary = Binary()
    binary.addMul(xx, yy)
    yield from binary.toCNF(zz)
    yield from binaryEq(zz, n)

def test_commands():
    (p, q, r) = (Bool("p"), Bool("q"), Bool("r"))
    commands = [None, "bin/kissat", "bin/glueminisat -show-model", "bin/glucose -model", "bin/cadical"]
    for command in commands:
        sat = SAT(command=command)
        cnf = [ [p,q], [~p,q], [~p,~q] ]
        sat.add(*cnf)
        assert sat.nVars() == 2
        assert sat.nClauses() == 3
        assert norm(sat.find()) == ( (p,0), (q,1) )
        assert sat.stats["ncalls"] == 1
        assert sat.stats["nmodels"] == 1
        assert sat.stats["time"] < 10
        assert sat.stats["solving"] < 10
        info = sat.stats["sat"][-1]
        print(f"{command}: {info}")
        assert info["result"] == "SATISFIABLE"
        assert info["variables"] == 2
        assert info["clauses"] == 3
        if command not in ["bin/cadical", "bin/clasp"]:
            assert info["conflicts"] >= 0
            assert info["decisions"] >= 0
            assert info["propagations"] > 0
            assert info["solving"] < 10
        sat.addBlock()
        assert sat.nClauses() == 4
        assert norm(sat.find()) is None
        assert sat.stats["ncalls"] == 2
        assert sat.stats["nmodels"] == 1
        info = sat.stats["sat"][-1]
        assert info["result"] == "UNSATISFIABLE"
        assert info["clauses"] == 4

def test_exceptions():
    sat = SAT(command="XXXXX")
    with pytest.raises(CspsatException, match=r"SATソルバーの実行エラー.*"):
        sat.add(*factor())
        sat.solve()
    sat = SAT(maxClauses=10)
    with pytest.raises(CspsatException, match=r".*?より多い節が追加された.*"):
        sat.add(*factor())
    sat = SAT(limit=1)
    with pytest.raises(CspsatTimeout, match=r"SATソルバーの実行時間が.*"):
        sat.add(*factor())
        sat.solve()
    info = sat.stats["sat"][-1]
    assert info["result"] == "TIMEOUT"

def test_sat():
    (p, q) = (Bool("p"), Bool("q"))
    cnf = toCNF(["equ", p, q])
    sat = SAT()
    sols = sat.solutions(cnf, num=0)
    assert norm(sols) == norm([{p:0,q:0}, {p:1,q:1}])
    sat = SAT()
    sat.add(*cnf)
    sols = sat.solutions(num=0)
    assert norm(sols) == norm([{p:0,q:0}, {p:1,q:1}])

def test_parameters():
    (p, _q) = (Bool("p"), Bool())
    cnf = toCNF(["equ", p, _q])
    sols = solutionsSAT(cnf, num=0)
    assert norm(sols) == norm([{p:0}, {p:1}])
    sols = solutionsSAT(cnf, num=0, positiveOnly=True)
    assert norm(sols) == norm([{}, {p:1}])
    sols = solutionsSAT(cnf, num=0, includeAux=True)
    assert norm(sols) == norm([{p:0,_q:0}, {p:1,_q:1}])
    sols = solutionsSAT(cnf, num=0, includeAux=True, positiveOnly=True)
    assert norm(sols) == norm([{}, {p:1,_q:1}])

def test_save_load_sat():
    satFile = SAT._tempfileName(".sat")
    cnf = list(factor(49, 3))
    sols1 = solutionsSAT(cnf, num=0)
    saveSAT(cnf, satFile)
    cnf = loadSAT(satFile)
    sols2 = solutionsSAT(cnf, num=0)
    assert norm(sols1) == norm(sols2)
    os.remove(satFile)

    cnfFile = SAT._tempfileName(".cnf")
    cnf = list(factor(49, 3))
    saveDimacs(cnf, cnfFile)
    os.path.exists(cnfFile)
    os.remove(cnfFile)

def decodeBinary(x, sol):
    b = []
    if isinstance(x, list):
        b = [ sol[y] for y in x ]
    else:
        i = 0
        while sol.get(x(i)) != None:
            b.append(sol[x(i)])
            i += 1
    return sum([ b[i]<<i for i in range(len(b)) ])      

def test_bigint():
    n = 10
    (x, y, z) = (Bool("x"), Bool("y"), Bool("z"))
    xx = [ x(i) for i in range(n) ]
    yy = [ y(i) for i in range(n) ]
    zz = [ z(i) for i in range(n) ]
    (a1, a2) = (3, 5)
    def cnf():
        binary = Binary()
        binary.addPower(xx, 2, a=a1)
        binary.addPower(yy, 2, a=a2)
        yield from binary.toCNF(zz)
        yield xx
        yield yy
        yield zz
    for sol in solutionsSAT(cnf(), num=10):
        xv = decodeBinary(x, sol)
        yv = decodeBinary(y, sol)
        zv = decodeBinary(z, sol)
        print([xv, yv, zv])
        assert xv*xv*a1 + yv*yv*a2 == zv

