from cspsat import *

"""ペントミノのデータ
"""
pentominos = [
    ("F", ((0,1),(0,2),(1,0),(1,1),(2,1))),
    ("I", ((0,0),(1,0),(2,0),(3,0),(4,0))),
    ("L", ((0,0),(1,0),(2,0),(3,0),(3,1))),
    ("P", ((0,0),(0,1),(1,0),(1,1),(2,0))),
    ("N", ((0,0),(0,1),(1,1),(1,2),(1,3))),
    ("T", ((0,0),(0,1),(0,2),(1,1),(2,1))),
    ("U", ((0,0),(0,2),(1,0),(1,1),(1,2))),
    ("V", ((0,0),(1,0),(2,0),(2,1),(2,2))),
    ("W", ((0,0),(1,0),(1,1),(2,1),(2,2))),
    ("X", ((0,1),(1,0),(1,1),(1,2),(2,1))),
    ("Y", ((0,2),(1,0),(1,1),(1,2),(1,3))),
    ("Z", ((0,0),(0,1),(1,1),(2,1),(2,2)))
]

"""回転・鏡像のすべてを返す．
"""
def possibleShapes(shape0):
    shapes = set()
    for shape1 in (shape0, [ (i,-j) for (i,j) in shape0 ]):
        for shape2 in (shape1, [ (-i,j) for (i,j) in shape1 ]):
            for shape3 in (shape2, [ (j,i) for (i,j) in shape2 ]):
                di = min(i for (i,j) in shape3)
                dj = min(j for (i,j) in shape3)
                shape4 = sorted([ (i-di,j-dj) for (i,j) in shape3 ])
                shapes.add(tuple(shape4))
    return sorted(list(shapes))

"""m x n 中に置いた場合に可能な座標のリストのすべてを返す．
"""
def possiblePlacements(shape0, m, n):
    for shape in possibleShapes(shape0):
        dm = max(di for (di,dj) in shape)
        dn = max(dj for (di,dj) in shape)
        for i in range(m-dm):
            for j in range(n-dn):
                yield [ (i+di,j+dj) for (di,dj) in shape ]

"""m x n 中にポリオミノを敷き詰める方法を探すCSP．
"""
def polyomino(m, n, ominos):
    t = len(ominos)
    c = Var("c")
    choices = {}
    for k in range(t):
        options = list(possiblePlacements(ominos[k][1], m, n))
        # c(k) == s : k番目のオミノをs番目の方法で置く
        yield ["int", c(k), 0, len(options)-1]
        for (s,ijs) in enumerate(options):
            for (i,j) in ijs:
                choices[(i,j)] = choices.get((i,j), [])
                choices[(i,j)].append(["==", c(k), s])
    for i in range(m):
        for j in range(n):
            pp = []
            for c in choices[(i,j)]:
                p = Bool()
                yield ["equ", p, c]
                pp.append(p)
            yield ["eqK", pp, 1]

"""ポリオミノ敷き詰めの解を表示する．
"""
def showPolyomino(sol, m, n, ominos):
    c = Var("c")
    p = [ [ "." for j in range(n) ] for i in range(m) ]
    for k in range(len(ominos)):
      options = list(possiblePlacements(ominos[k][1], m, n))
      for (i,j) in options[sol[c(k)]]:
        p[i][j] = ominos[k][0]
    for i in range(m):
      print(" ".join(p[i]))
