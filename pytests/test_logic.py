import pytest
from cspsat import *

def norm(f):
    if f is None or isinstance(f, (int,str,Bool,Var)):
        return f
    if isinstance(f, dict):
        f = [ (k,norm(g)) for (k,g) in f.items() ]
    else:
        f = [ norm(g) for g in iter(f) ]
    return tuple(sorted(f))

def test_bool():
    (p, _q) = (Bool("p"), Bool())
    assert str(p) == "p"
    assert str(~p) == "~p"
    assert str(p(1)) == "p(1)"
    assert str(p(1,2)) == "p(1,2)"
    assert str(_q).startswith("?")
    assert ~ ~p == p
    assert abs(p) == p
    assert abs(~p) == p
    assert not p.isAux()
    assert _q.isAux()
    assert Bool("TRUE") == TRUE
    assert Bool("FALSE") == FALSE
    assert ~TRUE == FALSE

def test_logic():
    (p, q, r) = (Bool("p"), Bool("q"), Bool("r"))
    assert variables([p, (~p,q), [~q,~r], TRUE, FALSE]) == { p, q, r }
    aa = norm(assignments({p,q}))
    assert aa == ( ((p,0),(q,0)),((p,0),(q,1)),((p,1),(q,0)),((p,1),(q,1)) )
    assert value(["not",p], {p:0}) == 1
    assert value(["not",p], {p:1}) == 0
    with pytest.raises(CspsatException, match=r"論理式の構文エラー.*"):
        value(["xxx",p,q], {p:0,q:0})
    assert isValid(["or", ["equ", p, q], ["xor", p, q]])
    assert isSat(["or", p, q])
    assert not isSat(["and", ["equ", p, q], ["xor", p, q]])
    assert isEquiv(["equ", p, q], ["xor", ~p, q])
    mm = norm(models(["or", p, q], num=0))
    assert mm == ( ((p,0),(q,1)),((p,1),(q,0)),((p,1),(q,1)) )
    def tocnf(f): return norm(toCNF(f))
    assert tocnf(["or", p, ~p]) == ()
    assert tocnf(["and", p, ~p]) == ( (~p,), (p,) )
    assert tocnf(["or", p, q]) == ( (p,q), )
    assert tocnf(["and", p, q]) == ( (p,), (q,) )
    assert norm(toDNF(["or", p, q])) == ( (p,), (q,) )

