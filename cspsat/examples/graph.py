from itertools import *
from cspsat import *

"""McGregor graph
"""
def mcgregorGraph(n):
    def v(a, b):
        s = str(a) if a < 10 else chr(ord("a")+a-10)
        s += str(b) if b < 10 else chr(ord("a")+b-10)
        return s
    vertices = [ v(a,b) for a in range(0,n+1) for b in range(0,n) ]
    edges = []
    def e(u, v):
        edges.append((u, v))
    for a in range(0,n+1):
        for b in range(0,n):
            if a == 1 and b == 0:
                for i in range(int(n/2),n):
                    e(v(a, b), v(n, i))
            else:
                if a == 0 and b == 0:
                    e(v(a, b), v(1, 0))
                    for i in range(1,int(n/2)+1):
                        e(v(a, b), v(n, i))
                if a < n and b < n-1:
                    e(v(a, b), v(a+1, b+1))
                if a == 0:
                    e(v(a, b), v(n, n-1))
                if a != b and a < n:
                    e(v(a, b), v(a+1, b))
                if a != b+1 and b < n-1:
                    e(v(a, b), v(a, b+1))
                if a == b and b < n-1:
                    e(v(a, b), v(n-a, 0))
                if a == b and a > 0:
                    e(v(a, b), v(n+1-a, 0))
                if b == n-1 and a > 0 and a < b:
                    e(v(a, b), v(n-a, n-a-1))
                if b == n-1 and a > 0 and a < n:
                    e(v(a, b), v(n+1-a, n-a))
    return (vertices, edges)

"""Complete graph K_n
"""
def completeGraph(n):
    vertices = [ i for i in range(n) ]
    edges = list(combinations(vertices, 2))
    return (vertices, edges)

"""Complete bipartite graph K_{m,n}
"""
def completeBipartiteGraph(m, n):
    vertices = [ i for i in range(m+n) ]
    edges = [ (i,m+j) for i in range(m) for j in range(n) ]
    return (vertices, edges)

"""Grid graph
"""
def gridGraph(m, n):
    vertices = [ (i,j) for i in range(m) for j in range(n) ]
    edges = []
    for (i,j) in vertices:
        for (k,l) in [(i,j+1), (i+1,j)]:
            if k in range(m) and l in range(n):
                edges.append(((i,j), (k,l)))
    return (vertices, edges)

"""Cycle graph C_n
"""
def cycleGraph(n):
    vertices = [ i for i in range(n) ]
    edges = [ (i,i+1) for i in range(n-1) ] + [ (n-1,0) ]
    return (vertices, edges)

"""Star graph S_n
"""
def starGraph(n):
    return completeBipartiteGraph(1, n-1)

"""Wheel graph W_n
"""
def wheelGraph(n):
    (vertices, edges) = starGraph(n)
    edges += [ (i,i+1) for i in range(1,n-1) ] + [ (n-1,1) ]
    return (vertices, edges)

"""Queen graph Q_n
"""
def queenGraph(n):
    vertices = [ (i,j) for i in range(n) for j in range(n) ]
    edges = []
    for ((i,j),(k,l)) in combinations(vertices, 2):
        if i == k or j == l or i+j == k+l or i-j == k-l:
            edges.append(((i,j), (k,l)))
    return (vertices, edges)

"""Knight graph
"""
def knightGraph(n):
    vertices = [ (i,j) for i in range(n) for j in range(n) ]
    edges = []
    for (i,j) in vertices:
        for (k,l) in [(i+1,j-2), (i+1,j+2), (i+2,j-1), (i+2,j+1)]:
            if k in range(n) and l in range(n):
                edges.append(((i,j), (k,l)))
    return (vertices, edges)

"""Super queen graph
"""
def superQueenGraph(n):
    vertices = [ (i,j) for i in range(n) for j in range(n) ]
    edges = []
    for ((i,j),(k,l)) in combinations(vertices, 2):
        if i == k or j == l or i+j == k+l or i-j == k-l:
            edges.append(((i,j), (k,l)))
        elif (k,l) in [(i+1,j-2), (i+1,j+2), (i+2,j-1), (i+2,j+1)]:
            edges.append(((i,j), (k,l)))
    return (vertices, edges)

"""King graph
"""
def kingGraph(n):
    vertices = [ (i,j) for i in range(n) for j in range(n) ]
    edges = []
    for (i,j) in vertices:
        for (k,l) in [(i,j+1), (i+1,j-1), (i+1,j), (i+1,j+1)]:
            if k in range(n) and l in range(n):
                edges.append(((i,j), (k,l)))
    return (vertices, edges)

"""Sudoku graph
"""
def sudokuGraph(n):
    m = math.floor(math.sqrt(n))
    assert n == m*m
    vertices = [ (i,j) for i in range(n) for j in range(n) ]
    edges = []
    for ((i,j),(k,l)) in combinations(vertices, 2):
        (a,b) = (math.floor(i/m),math.floor(j/m))
        (c,d) = (math.floor(k/m),math.floor(l/m))
        if i == k or j == l or (a == c and b == d):
            edges.append(((i,j), (k,l)))
    return (vertices, edges)

"""Graph (vertex) coloring CNF
"""
def vertexColoringCNF(vertices, edges, k, x=Bool("x")):
    for v in vertices:
        yield from eq1([ x(v,c) for c in range(1,k+1) ])
    for (u,v) in edges:
        for c in range(1,k+1):
            yield [ ~x(u,c), ~x(v,c) ]

"""Graph (vertex) coloring CSP
"""
def vertexColoring(vertices, edges, k, x=Var("x")):
    for v in vertices:
        yield ["int", x(v), 1, k]
    for (u,v) in edges:
        yield ["!=", x(u), x(v)]

"""Graph edge coloring CNF
"""
def edgeColoringCNF(vertices, edges, k, x=Bool("x")):
    for e in edges:
        yield from eq1([ x(e,c) for c in range(1,k+1) ])
    for (e1,e2) in combinations(edges, 2):
        if any(u == v for u in e1 for v in e2):
            for c in range(1,k+1):
                yield [ ~x(e1,c), ~x(e2,c) ]

"""Graph edge coloring CSP
"""
def edgeColoring(vertices, edges, k, x=Var("x")):
    for e in edges:
        yield ["int", x(e), 1, k]
    for (e1,e2) in combinations(edges, 2):
        if any(u == v for u in e1 for v in e2):
            yield ["!=", x(e1), x(e2)]

"""Hamiltonian cycle CSP
"""
def hamiltonianCycle(vertices, edges, e=Bool("e"), a=Bool("a"), x=Var("x")):
    def adj(v):
        return [ edge[1-i] for edge in edges for i in [0,1] if edge[i] == v ]
    for (u,v) in edges:
        yield ["==", e(u,v), ["+", a(u,v), a(v,u)]]
    for v in vertices:
        yield ["==", ["+", *[ a(v,u) for u in adj(v) ]], 1]
        yield ["==", ["+", *[ a(u,v) for u in adj(v) ]], 1]
    s = vertices[0]
    yield ["int", x(s), 0, 0]
    for v in vertices[1:]:
        yield ["int", x(v), 1, len(vertices)-1]
    for u in vertices:
        for v in [ v for v in adj(u) if v != s ]:
            yield ["imp", a(u,v), ["==", x(v), ["+", x(u), 1]]]

"""Hamiltonian path CSP
"""
def hamiltonianPath(vertices, edges, e=Bool("e"), a=Bool("a"), x=Var("x")):
    s = None
    edges1 = edges + [ (s,v) for v in vertices ]
    vertices1 = [s] + vertices
    yield from hamiltonianCycle(vertices1, edges1, e=e, a=a, x=x)

"""Single cycle CSP
"""
def singleCycle(vertices, edges, minLen=None, maxLen=None, e=Bool("e"), x=Var("x")):
    (a, d, r) = (Bool(), Bool(), Bool())
    def adj(v):
        return [ edge[1-i] for edge in edges for i in [0,1] if edge[i] == v ]
    for (u,v) in edges:
        yield ["==", e(u,v), ["+", a(u,v), a(v,u)]]
    for v in vertices:
        yield ["==", ["+", *[ a(v,u) for u in adj(v) ]], d(v)]
        yield ["==", ["+", *[ a(u,v) for u in adj(v) ]], d(v)]
    yield ["eqK", [ r(v) for v in vertices ], 1]
    for v in vertices:
        yield ["int", x(v), 0, len(vertices)]
        yield ["equ", d(v), [">", x(v), 0]]
    for v in vertices:
        for u in adj(v):
            yield ["imp", ["and", r(v), a(v,u)], ["==", x(u), 1]]
            yield ["imp", ["and", ~r(v), a(v,u)], ["==", ["+", x(v), 1], x(u)]]
    for v in vertices:
        if minLen:
            yield ["imp", r(v), [">=", x(v), minLen]]
        if maxLen:
            yield ["imp", r(v), ["<=", x(v), maxLen]]
    if minLen:
        yield ["geK", [ d(v) for v in vertices ], minLen]
    if maxLen:
        yield ["leK", [ d(v) for v in vertices ], maxLen]

"""Single path CSP
"""
def singlePath(vertices, edges, minLen=None, maxLen=None, e=Bool("e"), x=Var("x")):
    (a, d, r) = (Bool(), Bool(), Bool())
    maxLen = maxLen or len(vertices)-1
    s = None
    edges1 = edges + [ (s,v) for v in vertices ]
    vertices1 = [s] + vertices
    yield r(s)
    minLen1 = minLen+2 if minLen else None
    maxLen1 = maxLen+2 if maxLen else None
    yield from singleCycle(vertices1, edges1, minLen=minLen1, maxLen=maxLen1, e=e, a=a, d=d, r=r, x=x)

"""Digraph Hamiltonian cycle CSP
"""
def digraphHamiltonianCycle(vertices, arcs, a=Bool("a"), x=Var("x")):
    def inAdj(v): return [ arc[0] for arc in arcs if arc[1] == v ]
    def outAdj(v): return [ arc[1] for arc in arcs if arc[0] == v ]
    for v in vertices:
        yield ["==", ["+", *[ a(u,v) for u in inAdj(v) ]], 1]
        yield ["==", ["+", *[ a(v,u) for u in outAdj(v) ]], 1]
    s = vertices[0]
    yield ["int", x(s), 0, 0]
    for v in vertices[1:]:
        yield ["int", x(v), 1, len(vertices)-1]
    for u in vertices:
        for v in [ v for v in outAdj(u) if v != s ]:
            yield ["imp", a(u,v), ["==", x(v), ["+", x(u), 1]]]

"""Digraph Hamiltonian path CSP
"""
def digraphHamiltonianPath(vertices, arcs, a=Bool("a"), x=Var("x")):
    s = None
    arcs1 = arcs + [ (s,v) for v in vertices ] + [ (v,s) for v in vertices ]
    vertices1 = [s] + vertices
    yield from digraphHamiltonianCycle(vertices1, arcs1, a=a, x=x)
