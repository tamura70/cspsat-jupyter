"""新規関数と新規制約に対応するためのフック関数を提供．
"""

def defaultFunctionHook(function, encoder):
    """新規関数に対応するためのフックとしてデフォールトで設定されている関数．
    以下に対応している．

    * ``["div", X, n]``
    * ``["mod", X, n]``
    * ``["abs", X]``
    * ``["min", X1, ..., Xn]``
    * ``["max", X1, ..., Xn]``
    * ``["if", A, X, Y]``

    Args:
        function (list): 制約充足問題中の式.
        encoder (Encoder): Encoderインスタンス.

    Returns:
        引数の function そのまま，または function を変換した式．
    """
    from .csp import LogEncoder

    def getBound(xx):
        bounds = [ encoder.getBound(x) for x in xx ]
        lb = min(b[0] for b in bounds)
        ub = max(b[1] for b in bounds)
        return (lb, ub)
    def decomposeAbs(x):
        (lb, ub) = encoder.getBound(x)
        z = Var()
        encoder.put(["int", z, 0, max(abs(lb), abs(ub))])
        encoder.put(["imp", ["<", x, 0], ["==", z, ["-", x]]])
        encoder.put(["imp", [">=", x, 0], ["==", z, x]])
        return z
    def decomposeMin(xx):
        (lb, ub) = getBound(xx)
        z = Var()
        c1 = [ ["<=", z, x] for x in xx ]
        c2 = [ [">=", z, x] for x in xx ]
        encoder.put(["int", z, lb, ub])
        encoder.put(["and", *c1, ["or", *c2]])
        return z
    def decomposeMax(xx):
        (lb, ub) = getBound(xx)
        z = Var()
        c1 = [ [">=", z, x] for x in xx ]
        c2 = [ ["<=", z, x] for x in xx ]
        encoder.put(["int", z, lb, ub])
        encoder.put(["and", *c1, ["or", *c2]])
        return z
    def decomposeIf(c, x, y):
        (lb, ub) = getBound([x, y])
        (p, z) = (Bool(), Var())
        encoder.put(["equ", p, c])
        encoder.put(["int", z, lb, ub])
        encoder.put(["imp", p, ["==", z, x]])
        encoder.put(["imp", ~p, ["==", z, y]])
        return z
    def decomposeDivMod(x, n):
        if not isinstance(n, int) or n <= 0:
            raise CspsatException(f"divあるいはmodの第2引数が正の整数定数でない: {n}")
        (lb, ub) = encoder.getBound(x)
        (q, r) = (Var(), Var())
        encoder.put(["int", q, lb//n, ub//n])
        encoder.put(["int", r, 0, n-1])
        encoder.put(["==", x, ["+", ["*", n, q], r]])
        return (q, r)
    def decomposeBit(x, k):
        if not isinstance(encoder, LogEncoder):
            raise CspsatException("EncoderがLogEncoderでない")
        if encoder.intLb(x) != 0:
            raise CspsatException(f"変数の下限が0でない: {x}")
        return encoder.varBitK(x, k)

    match function:
        case ["div", x, n] | ["//", x, n]:
            function = decomposeDivMod(x, n)[0]
        case ["mod", x, n] | ["%", x, n]:
            function = decomposeDivMod(x, n)[1]
        case ["abs", x]:
            function = decomposeAbs(x)
        case ["min", *xx]:
            function = decomposeMin(xx)
        case ["max", *xx]:
            function = decomposeMax(xx)
        case ["if", c, x, y]:
            function = decomposeIf(c, x, y)
        case ["bit", x, k]:
            function = decomposeBit(x, k)
    return function

def defaultConstraintHook(constraint, encoder):
    """新規制約に対応するためのフックとしてデフォールトで設定されている関数．
    以下に対応している．

    * ``["alldifferent", X1, ..., Xn]``
    * ``["lexCmp", cmp, [X1,...Xn], [Y1,...,Yn]]`` (cmpは"==", "!=", "<=", "<", ">=", ">")
    * ``["mulCmp", cmp, X, Y, Z]`` (cmpは"==", "!=", "<=", "<", ">=", ">")
    * ``["powCmp", cmp, X, n, Z]`` (cmpは"==", "!=", "<=", "<", ">=", ">")
    * ``["bits", [X1,...Xn], X]``
    * ``["bit", X, i]``

    Args:
        constraint (list): 制約充足問題中の制約.
        encoder (Encoder): Encoderインスタンス.

    Returns:
        引数の constraint そのまま，または constraint を変換した制約．
    """

    from .csp import LogEncoder

    def decomposeAlldifferent(xx):
        bounds = [ encoder.getBound(x) for x in xx ]
        lb = min(b[0] for b in bounds)
        ub = max(b[1] for b in bounds)
        d = ub - lb + 1
        m = d - len(xx)
        if m < 0:
            yield FALSE
            return
        if len(xx) <= 2:
            for (x1,x2) in itertools.combinations(xx, 2):
                yield ["!=", x1, x2]
            return
        t = Bool()
        if m > 0:
            tt = [ t(j) for j in range(lb, ub+1) ]
            yield ["eqK", tt, m]
        for k in range(lb, ub+1):
            p = Bool()
            for (i,x) in enumerate(xx):
                yield ["equ", p(i), ["==", x, k]]
            pp = [ p(i) for i in range(len(xx)) ]
            if m > 0:
                pp.append(t(k))
            yield ["eqK", pp, 1]
    def _fillZero(xx, yy):
        if len(xx) < len(yy):
            xx = xx + [0] * (len(yy)-len(xx))
        elif len(xx) > len(yy):
            yy = yy + [0] * (len(xx)-len(yy))
        return (xx, yy)
    def decomposeLexEq(xx, yy):
        (xx, yy) = _fillZero(xx, yy)
        for (i,x) in enumerate(xx):
            yield ["==", x, yy[i]]
    def decomposeLexLe(xx, yy, less=False):
        (xx, yy) = _fillZero(xx, yy)
        n = len(xx)
        a = Bool()
        yield a(-1)
        for i in range(n):
            yield ["or", ~a(i-1), ["<=", ["+", xx[i], ~a(i)], yy[i]]]
        yield ~a(n-1) if less else a(n-1)
    def decomposeMulCmp(cmp, x, y, z):
        if not isinstance(encoder, LogEncoder):
            raise CspsatException("EncoderがLogEncoderでない")
        binEqu = BinaryEquation(encoder)
        binEqu.addMul(x, y)
        binEqu.add(z, a=-1)
        yield from [ ["or", *c] for c in binEqu.cmp0(cmp) ]
    def decomposePowCmp(cmp, x, n, z):
        if not isinstance(encoder, LogEncoder):
            raise CspsatException("EncoderがLogEncoderでない")
        binEqu = BinaryEquation(encoder)
        binEqu.addPower(x, n)
        binEqu.add(z, a=-1)
        yield from [ ["or", *c] for c in binEqu.cmp0(cmp) ]
    def decomposeBits(xx, x):
        if not isinstance(encoder, LogEncoder):
            raise CspsatException("EncoderがLogEncoderでない")
        if encoder.intLb(x) != 0:
            raise CspsatException(f"変数の下限が0でない: {x}")
        yield from [ ["or", *c] for c in Binary.eq(xx, encoder.getBools(x)) ]
    def decomposeBit(x, k):
        if not isinstance(encoder, LogEncoder):
            raise CspsatException("EncoderがLogEncoderでない")
        if encoder.intLb(x) != 0:
            raise CspsatException(f"変数の下限が0でない: {x}")
        yield encoder.varBitK(x, k)

    match constraint:
        case ["alldifferent", *args]:
            cs = decomposeAlldifferent(args)
            constraint = ["and", *cs]
        case ["lexCmp", "==", xx, yy]:
            cs = decomposeLexEq(xx, yy)
            constraint = ["and", *cs]
        case ["lexCmp", "!=", xx, yy]:
            cs = decomposeLexEq(xx, yy)
            constraint = ["not", ["and", *cs]]
        case ["lexCmp", "<=", xx, yy]:
            cs = decomposeLexLe(xx, yy)
            constraint = ["and", *cs]
        case ["lexCmp", "<", xx, yy]:
            cs = decomposeLexLe(xx, yy, less=True)
            constraint = ["and", *cs]
        case ["lexCmp", ">=", xx, yy]:
            cs = decomposeLexLe(yy, xx)
            constraint = ["and", *cs]
        case ["lexCmp", ">", xx, yy]:
            cs = decomposeLexLe(yy, xx, less=True)
            constraint = ["and", *cs]
        case ["mulCmp", cmp, x, y, z]:
            cs = decomposeMulCmp(cmp, x, y, z)
            constraint = ["and", *cs]
        case ["powCmp", cmp, x, n, z]:
            cs = decomposePowCmp(cmp, x, n, z)
            constraint = ["and", *cs]
        case ["bits", xx, x]:
            cs = decomposeBits(xx, x)
            constraint = ["and", *cs]
        case ["bit", x, k]:
            cs = decomposeBit(x, k)
            constraint = ["and", *cs]
    return constraint

import itertools
from .util import CspsatException, Bool, FALSE, Var, Binary, BinaryEquation
