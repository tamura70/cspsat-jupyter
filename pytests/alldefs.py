from cspsat import *
from cspsat.examples.graph import *

# BEGIN notebook/04-prop-logic.org

def rivestCNF():
    (a, b, c, d) = (Bool("a"), Bool("b"), Bool("c"), Bool("d"))
    f1 = ["imp", ["and",~a,~b], ~c]
    f2 = ["imp", d, ["or",b,c]]
    f3 = ["not", ["and",~a,~c,~d]]
    f4 = ["imp", ["and",~d,a], b]
    f5 = ["imp", ["and",a,b], c]
    f6 = ["imp", ~d, ["or",~b,~c]]
    f7 = ["not", ["and",a,c,d]]
    f8 = ["imp", ["and",b,d], a]
    return toCNF(["and", f1, f2, f3, f4, f5, f6, f7, f8])

# END notebook/04-prop-logic.org

# BEGIN notebook/05-sat-examples.org

def liarPuzzle():
    yield [~A, ~B]
    yield [A, B]
    yield [A, ~B]

def liarPuzzle2():
    for clause in liarPuzzle():
        yield clause
    yield [~A, B]

def liarPuzzle2():
    yield from liarPuzzle()
    yield [~A, B]

def grouping():
    (A, B, C, D) = (Bool("A"), Bool("B"), Bool("C"), Bool("D"))
    yield from toCNF(["not", ["equ", A, B]])
    yield from toCNF(["not", ["equ", B, C]])
    yield from toCNF(["not", ["equ", C, D]])
    yield from toCNF(["not", ["equ", D, A]])

def waerden3(n, x=Bool("x")):
    for d in range(1,n):
        for i in range(1,n+1):
            if i+2*d <= n:
                yield [ x(i), x(i+d), x(i+2*d) ]
                yield [ ~x(i), ~x(i+d), ~x(i+2*d) ]

def waerden(k, n, x=Bool("x")):
    for d in range(1,n):
        for i in range(1,n+1):
            if i+(k-1)*d <= n:
                yield [ x(i+j*d) for j in range(k) ]
                yield [ ~x(i+j*d) for j in range(k) ]

def solveWaerden(k, n, num=1, x=Bool("x")):
    for sol in solutionsSAT(waerden(k, n), num=num):
        print([ sol[x(i)] for i in range(1,n+1) ])

def lightsOut1d(lights, p=Bool("p")):
    n = len(lights)
    yield from toCNF(["and", ~p(0), ~p(n+1)])
    for i in range(1, n+1):
        if lights[i-1]:
            yield from toCNF(["xor", p(i-1), p(i), p(i+1)])
        else:
            yield from toCNF(["xor", ~p(i-1), p(i), p(i+1)])

def lightsOut2d(lights, p=Bool("p")):
    (m, n) = (len(lights), len(lights[0]))
    for i in [0,m+1]:
        yield from [ [~p(i,j)] for j in range(n+2) ]
    for j in [0,n+1]:
        yield from [ [~p(i,j)] for i in range(1,m+1) ]
    for i in range(1, m+1):
        for j in range(1, n+1):
            f = ["xor", p(i,j), p(i-1,j), p(i+1,j), p(i,j-1), p(i,j+1)]
            if lights[i-1][j-1]:
                yield from toCNF(f)
            else:
                yield from toCNF(["not", f])

def exactOne(n, x=Bool("x")):
    for i in range(1,n+1):
        yield from eq1([ x(i,j) for j in range(1,n+1) ])
    for j in range(1,n+1):
        yield from eq1([ x(i,j) for i in range(1,n+1) ])

def pigeonHolePrinciple(n, x=Bool("x")):
    for i in range(1,n+2):
        yield from ge1([ x(i,j) for j in range(1,n+1) ])
    for j in range(1,n+1):
        yield from le1([ x(i,j) for i in range(1,n+2) ])

def logicPuzzle1_0(x=Bool("x")):
    I = [ "論", "理", "学" ]
    J = [ "長男", "次男", "三男" ]
    for i in I:
        yield from eq1([ x(i,j) for j in J ])
    for j in J:
        yield from eq1([ x(i,j) for i in I ])

def logicPuzzle1_1():
    yield from logicPuzzle1_0(x=Bool("x"))
    yield [ ~x("学","長男") ]
    yield [ ~x("論","長男"), x("理","次男") ]
    yield [ ~x("論","次男"), x("理","三男") ]

def logicPuzzle1():
    yield from logicPuzzle1_0(x=Bool("x"))
    yield [ ~x("学","長男") ]
    yield [ ~x("論","長男"), x("理","次男") ]
    yield [ ~x("論","次男"), x("理","三男") ]
    yield [ ~x("論","三男") ]

def logicPuzzle1(x=Bool("x")):
    yield from logicPuzzle1_0()
    yield [ ~x("学","長男") ]
    yield [ ~x("論","長男"), ~x("理","三男") ]
    yield [ ~x("論","次男"), ~x("理","長男") ]
    yield [ ~x("論","三男") ]

def logicPuzzle2(x=Bool("x")):
    I = [ "論", "理", "学" ]
    J = [ "袋", "箱", "缶" ]
    K = [ "赤", "緑", "青" ]
    for i in I:
        yield from eq1([ x(i,j,k) for j in J for k in K ])
    for j in J:
        yield from eq1([ x(i,j,k) for i in I for k in K ])
    for k in K:
        yield from eq1([ x(i,j,k) for i in I for j in J ])
    # 論は袋を買いました
    yield [ x("論","袋",k) for k in K ]
    # 理が買ったのは箱ではありません
    for k in K:
        yield [ ~x("理","箱",k) ]
    # 袋は青色です
    yield [ x(i,"袋","青") for i in I ]
    # 学が買ったものは赤色ではありません
    for j in J:
        yield [ ~x("学",j,"赤") ]

def queens(n, x=Bool("x")):
    for i in range(n):
        yield from eq1([ x(i,j) for j in range(n) ])
    for j in range(n):
        yield from eq1([ x(i,j) for i in range(n) ])
    for a in range(0, 2*n-1):
        yield from le1([ x(i,a-i) for i in range(n) if a-i in range(n) ])
    for b in range(-n+1, n):
        yield from le1([ x(i,i-b) for i in range(n) if i-b in range(n) ])

def squeens(n, x=Bool("x")):
    yield from queens(n)
    for i in range(n):
        for j in range(n):
            yield [ ~x(i,j), x(n-1-i,n-1-j) ]

def queens0(n):
    def q(i, qs, qs1, qs2):
        if i == n:
            yield qs
        else:
            for j in range(0, n):
                if j not in qs and i+j not in qs1 and i-j not in qs2:
                    yield from q(i+1, qs+[j], qs1+[i+j], qs2+[i-j])
    yield from q(0, [], [], [])

def graphColoring(vertices, edges, k, x=Bool("x")):
    for v in vertices:
        yield from eq1([ x(v,c) for c in range(1,k+1) ])
    for (u,v) in edges:
        for c in range(1,k+1):
            yield [ ~x(u,c), ~x(v,c) ]

def queenGraph(n):
    vertices = [ (i,j) for i in range(n) for j in range(n) ]
    edges = []
    for ((i,j),(k,l)) in combinations(vertices, 2):
            if i == k or j == l or i+j == k+l or i-j == k-l:
                    edges.append(((i,j), (k,l)))
    return (vertices, edges)

def radioColoring(vertices, edges, k, d=1, x=Bool("x")):
    for v in vertices:
        yield from eq1([ x(v,c) for c in range(1,k+1) ])
    for (u,v) in edges:
        for c1 in range(1,k+1):
            for c2 in range(1,k+1):
                if -d < c1-c2 < d:
                    yield [ ~x(u,c1), ~x(v,c2) ]

def m32bchain(xx, yy, zz):
    (a, b, c, t) = (Bool(), Bool(), Bool(), Bool())
    for i in range(3):
        yield ["equ", a(i), ["and", xx[i], yy[0]]]
    for i in range(3):
        yield ["equ", b(i), ["and", xx[i], yy[1]]]
    yield ["equ", zz[0], a(0)]
    yield ["equ", zz[1], ["xor", a(1), b(0)]]
    yield ["equ", c(0), ["and", a(1), b(0)]]
    yield ["equ", t(1), ["xor", a(2), b(1)]]
    yield ["equ", t(2), ["and", a(2), b(1)]]
    yield ["equ", zz[2], ["xor", t(1), c(0)]]
    yield ["equ", t(3), ["and", t(1), c(0)]]
    yield ["equ", c(1), ["or", t(2), t(3)]]
    yield ["equ", zz[3], ["xor", b(2), c(1)]]
    yield ["equ", c(2), ["and", b(2), c(1)]]
    yield ["equ", zz[4], c(2)]

def factor(n, m):
    (x, y, z) = (Bool("x"), Bool("y"), Bool("z"))
    xx = [ x(i) for i in range(m) ]
    yy = [ y(i) for i in range(m) ]
    zz = [ z(i) for i in range(2*m) ]
    binary = Binary()
    binary.addMul(xx, yy)
    cnf = [*binary.toCNF(zz), *Binary.eqK(zz, n)]
    for sol in solutionsSAT(cnf):
        xv = sum([ sol[x] << i for (i,x) in enumerate(xx) ])
        yv = sum([ sol[y] << i for (i,y) in enumerate(yy) ])
        zv = sum([ sol[z] << i for (i,z) in enumerate(zz) ])
        print([xv, yv, zv])

def palindrome(xx, n):
    for i in range(int(n/2)):
        yield from toCNF(["equ", xx[i], xx[n-i-1]])
    for i in range(n,len(xx)):
        yield [~xx[i]]
    yield [xx[0]]

def binaryPalindromicSquare(n, x=Bool("x"), z=Bool("z")):
    xx = [ x(i) for i in range((n+1)//2) ]
    zz = [ z(i) for i in range(n) ]
    binary = Binary()
    binary.addPower(xx, 2)
    cnf = [*binary.toCNF(zz), *palindrome(zz, n)]
    if n >= 4:
        cnf.extend([ [~zz[1]], [~zz[2]] ])
    for sol in solutionsSAT(cnf, num=0):
        zv = sum([ sol[z] << i for (i,z) in enumerate(zz) ])
        print(f"{n} bits: {zv}")

# END notebook/05-sat-examples.org

# BEGIN notebook/06-csp.org

def magicSequence(n, x=Bool("x")):
    for i in range(n):
        yield ["eqK", [ x(i,j) for j in range(n) ], 1]
        for j in range(n):
            yield ["equ", x(i,j), ["eqK", [ x(k,i) for k in range(n) ], j]]

def composite(m, n=Var("n")):
    yield ["int", n, 2, m]
    cs = [ ["and", ["<", k, n], ["==", ["mod", n, k], 0]] for k in range(2, m) ]
    yield ["or", *cs]

def prime(m, n=Var("n")):
    yield ["int", n, 2, m]
    for k in range(2, m+1):
        yield ["imp", ["<", k, n], [">", ["mod", n, k], 0]]

# END notebook/06-csp.org

# BEGIN notebook/07-csp-examples.org

def magicSquare3x3(x=Var("x")):
    for i in range(3):
        for j in range(3):
            yield ["int", x(i,j), 1, 9]
    xx = [ x(i,j) for i in range(3) for j in range(3) ]
    yield ["alldifferent", *xx]
    for i in range(3):
        yield ["==", ["+", x(i,0), x(i,1), x(i,2)], 15]
    for j in range(3):
        yield ["==", ["+", x(0,j), x(1,j), x(2,j)], 15]
    yield ["==", ["+", x(0,0), x(1,1), x(2,2)], 15]
    yield ["==", ["+", x(0,2), x(1,1), x(2,0)], 15]

def magicSquare(n, x=Var("x")):
    s = n*(n**2+1)//2
    xx = [ x(i,j) for i in range(n) for j in range(n) ]
    for v in xx:
        yield ["int", v, 1, n*n]
    yield ["alldifferent", *xx]
    for i in range(n):
        yield ["==", ["+", *[ x(i,j) for j in range(n) ]], s]
    for j in range(n):
        yield ["==", ["+", *[ x(i,j) for i in range(n) ]], s]
    yield ["==", ["+", *[ x(i,i) for i in range(n) ]], s]
    yield ["==", ["+", *[ x(i,n-i-1) for i in range(n) ]], s]
    yield ["<", x(0,0), x(0,n-1)]
    yield ["<", x(0,0), x(n-1,0)]
    yield ["<", x(0,0), x(n-1,n-1)]
    yield ["<", x(0,n-1), x(n-1,0)]

def pandiagonalMagicSquare(n, x=Var("x")):
    s = n*(n**2+1)//2
    xx = [ x(i,j) for i in range(n) for j in range(n) ]
    for v in xx:
        yield ["int", v, 1, n*n]
    yield ["alldifferent", *xx]
    for i in range(n):
        yield ["==", ["+", *[ x(i,j) for j in range(n) ]], s]
    for j in range(n):
        yield ["==", ["+", *[ x(i,j) for i in range(n) ]], s]
    for a in range(n):
        yield ["==", ["+", *[ x(i,(a-i)%n) for i in range(n) ]], s]
    for b in range(n):
        yield ["==", ["+", *[ x(i,(i-b)%n) for i in range(n) ]], s]
    yield ["<", x(0,0), x(0,n-1)]
    yield ["<", x(0,0), x(n-1,0)]
    yield ["<", x(0,0), x(n-1,n-1)]
    yield ["<", x(0,n-1), x(n-1,0)]

def alldiff(xx):
    for (i,j) in combinations(range(len(xx)), 2):
        yield ["!=", xx[i], xx[j]]

def alldiff2(xx):
    yield from alldiff(xx)
    for k in range(1,10):
        yield ["or", *[ ["==", x, k] for x in xx ]]

def sudoku(clues):
    x = Var("x")
    for (i,j) in product(range(9), range(9)):
        yield ["int", x(i,j), 1, 9]
    for i in range(9):
        yield ["alldifferent", *[ x(i,j) for j in range(9) ]]
    for j in range(9):
        yield ["alldifferent", *[ x(i,j) for i in range(9) ]]
    for (i,j) in product(range(0,9,3), range(0,9,3)):
        yield ["alldifferent", *[ x(i+a,j+b) for a in range(3) for b in range(3) ]]
    for (i,j) in product(range(9), range(9)):
        if clues[i][j] > 0:
            yield ["==", x(i,j), clues[i][j]]

def giantSudoku(clues, n=9, m=3, x=Var("x")):
    for (i,j) in product(range(n), range(n)):
        yield ["int", x(i,j), 1, n]
    for i in range(n):
        yield ["alldifferent", *[ x(i,j) for j in range(n) ]]
    for j in range(n):
        yield ["alldifferent", *[ x(i,j) for i in range(n) ]]
    for (i,j) in product(range(0,n,m), range(0,n,m)):
        yield ["alldifferent", *[ x(i+a,j+b) for a in range(m) for b in range(m) ]]
    for (i,j) in product(range(n), range(n)):
        if clues[i][j] > 0:
            yield ["==", x(i,j), clues[i][j]]

def uniqSol(csp):
    sols = list(solutionsCSP(csp, num=2))
    return len(sols) == 1

def initialClues(n=9, m=3):
    (row, nums) = ([], [ i+1 for i in range(n) ])
    while nums:
        k = random.randrange(0, len(nums))
        row.append(nums.pop(k))
    clues = [row] + [[0] * n] * (n-1)
    [sol] = solutionsCSP(giantSudoku(clues, n=n, m=m))
    x = Var("x")
    return [ [ sol[x(i,j)] for j in range(n) ] for i in range(n) ]

def sendMoreMoney():
    vs = [S,E,N,D,M,O,R,Y] = [ Var(v) for v in "SENDMORY" ]
    for v in vs:
        yield ["int", v, 0, 9]
    yield ["alldifferent", *vs]
    yield ["and", [">", S, 0], [">", M, 0]]
    send = ["+", ["*",1000,S], ["*",100,E], ["*",10,N], D]
    more = ["+", ["*",1000,M], ["*",100,O], ["*",10,R], E]
    money = ["+", ["*",10000,M], ["*",1000,O], ["*",100,N], ["*",10,E], Y]
    yield ["==", ["+", send, more], money]

def copris1():
    def word(xx):
        return [ ["*",xx[len(xx)-i-1],10**i] for i in range(len(xx)) ]
    vs = [C,O,P,R,I,S,A,L,V,E] = [ Var(v) for v in "COPRISALVE" ]
    for v in vs:
        yield ["int", v, 0, 9]
    yield ["alldifferent", *vs]
    yield ["and", [">", C, 0], [">", S, 0]]
    copris = ["+", *word([C,O,P,R,I,S])]
    scala = ["+", *word([S,C,A,L,A])]
    csp = ["+", *word([C,S,P])]
    solver = ["+", *word([S,O,L,V,E,R])]
    yield ["==", ["+", copris, scala, csp], solver]

def copris2():
    vs = [C,O,P,R,I,S,A,L,V,E] = [ Var(v) for v in "COPRISALVE" ]
    for v in vs:
        yield ["int", v, 0, 9]
    yield ["alldifferent", *vs]
    yield [">", C, 0]
    yield [">", S, 0]
    c = Var("c")
    for i in range(1,6): 
        yield ["int", c(i), 0, 2]
    yield ["==", ["+", S, A, P      ], ["+", R, ["*", 10, c(1)]]]
    yield ["==", ["+", I, L, S, c(1)], ["+", E, ["*", 10, c(2)]]]
    yield ["==", ["+", R, A, C, c(2)], ["+", V, ["*", 10, c(3)]]]
    yield ["==", ["+", P, C,    c(3)], ["+", L, ["*", 10, c(4)]]]
    yield ["==", ["+", O, S,    c(4)], ["+", O, ["*", 10, c(5)]]]
    yield ["==", ["+", C,       c(5)],       S ]

def alphametic(words, x=Var("x"), c=Var("c")):
    letters = set(chain(*words))
    for l in letters:
        yield ["int", x(l), 0, 9]
    yield ["alldifferent", *[ x(l) for l in letters ]]
    yield from [ [">", x(w[0]), 0] for w in words ]
    cols = list(zip_longest(*map(lambda w: reversed(w), words)))
    n = len(cols)
    for i in range(n+1):
        yield ["int", c(i), 0, len(words)-2]
    yield from [ ["==", c(0), 0], ["==", c(n), 0] ]
    for i in range(n):
        xx = [ x(l) for l in cols[i] if l ]
        yield ["==", ["+", *xx[:-1], c(i)], ["+", xx[-1], ["*", 10, c(i+1)]]]

def taxicabNumber(m, n=Var("n"), a=Var("a"), c=Var("c")):
    yield ["int", n, 0, m]
    for i in range(4):
        yield ["int", a(i), 0, m]
        yield ["int", c(i), 0, m]
        yield ["powCmp", "==", a(i), 3, c(i)]
    yield ["==", n, ["+", c(0), c(1)]]
    yield ["==", n, ["+", c(2), c(3)]]
    yield ["<", a(0), a(1)]
    yield ["<", a(2), a(3)]
    yield ["<", a(0), a(2)]

def eulersConjecture(m=150, x=Var("x"), y=Var("y")):
    for i in range(5):
        yield ["int", x(i), 0, m]
        yield [">", x(i), 0]
        yield ["int", y(i), 0, m**5]
        yield ["powCmp", "==", x(i), 5, y(i)]
    yield ["==", ["+", y(0), y(1), y(2), y(3)], y(4)]

def collatz(m, steps, n=Var("n")):
    for i in range(steps+1):
        yield ["int", n(i), 1, m]
    for i in range(steps):
        yield ["==", n(i+1), ["if", ["==", ["mod", n(i), 2], 0], ["div", n(i), 2], ["+", ["*", 3, n(i)], 1]]]
    yield ["or", *[ ["==", n(0), n(t)] for t in range(1,steps+1) ]]

def collatzX(a, b, m, steps, n=Var("n")):
    for i in range(steps+1):
        yield ["int", n(i), 2, m]
    for i in range(steps):
        yield ["==", n(i+1), ["if", ["==", ["mod", n(i), 2], 0], ["div", n(i), 2], ["+", ["*", a, n(i)], b]]]
    yield ["or", *[ ["==", n(0), n(t)] for t in range(1,steps+1) ]]

def queenGraphColoring(n, k, q=Var("q")):
    def U(a): return [ (i,a-i) for i in range(n) if a-i in range(n) ]
    def D(b): return [ (i,i-b) for i in range(n) if i-b in range(n) ]
    for (i,j) in product(range(n), range(n)):
        yield ["int", q(i,j), 1, k]
    for i in range(n):
        yield ["alldifferent", *[ q(i,j) for j in range(n) ]]
    for j in range(n):
        yield ["alldifferent", *[ q(i,j) for i in range(n) ]]
    for a in range(0, 2*n-1):
        yield ["alldifferent", *[ q(i,j) for (i,j) in U(a) ]]
    for b in range(-n+1, n):
        yield ["alldifferent", *[ q(i,j) for (i,j) in D(b) ]]
    for j in range(n):
        yield ["==", q(0,j), j+1]

def latinSquare(n, x=Var("x")):
    for i in range(n):
        for j in range(n):
            yield ["int", x(i,j), 1, n]
    for i in range(n):
        yield ["alldifferent", *[ x(i,j) for j in range(n) ]]
    for j in range(n):
        yield ["alldifferent", *[ x(i,j) for i in range(n) ]]
    for j in range(n):
        yield ["==", x(0,j), j+1]

def eulerSquare(n, x=Var("x"), y=Var("y")):
    yield from latinSquare(n, x)
    yield from latinSquare(n, y)
    for i in range(1,n):
        yield ["==", x(i,0), i+1]
    zz = [ ["+", ["*", n, x(i,j)], y(i,j), -n] for i in range(n) for j in range(n) ]
    yield ["alldifferent", *zz]

def gracefulLabeling(vertices, edges, x=Var("x"), y=Var("y")):
    (n, m) = (len(vertices), len(edges))
    for v in vertices:
        yield ["int", x(v), 0, m]
    yield ["alldifferent", *[ x(v) for v in vertices ]]
    for (u,v) in edges:
        yield ["int", y(u,v), 1, m]
        yield ["==", y(u,v), ["abs", ["-", x(u), x(v)]]]
    yield ["alldifferent", *[ y(u,v) for (u,v) in edges ]]

def nqueens(n, q=Var("q")):
    for i in range(n):
        yield ["int", q(i), 0, n-1]
    yield ["alldifferent", *[ q(i) for i in range(n) ]]
    yield ["alldifferent", *[ ["+", q(i), i] for i in range(n) ]]
    yield ["alldifferent", *[ ["-", q(i), i] for i in range(n) ]]

def allIntervalSeries(n, x=Var("x"), d=Var("d")):
    for i in range(n):
        yield ["int", x(i), 0, n-1]
    yield ["alldifferent", *[ x(i) for i in range(n) ]]
    for i in range(n-1):
        yield ["int", d(i), 1, n-1]
        yield ["==", d(i), ["mod", ["-", x(i+1), x(i)], n]]
    yield ["alldifferent", *[ d(i) for i in range(n-1) ]]

def queenDomination1(n, s, q=Bool("q")):
    def attacked(i, j, k, l):
        return i == k or j == l or i+j == k+l or i-j == k-l
    for (i,j) in product(range(n), range(n)):
        qq = [ q(k,l) for k in range(n) for l in range(n) if attacked(i,j,k,l) ]
        yield ["or", *qq]
    yield ["eqK", [ q(i,j) for i in range(n) for j in range(n) ], s]

def queenDomination(n, s, q=Bool("q")):
    def U(a): return [ (i,a-i) for i in range(n) if a-i in range(n) ]
    def D(b): return [ (i,i-b) for i in range(n) if i-b in range(n) ]
    (r, c, u, d) = (Bool(), Bool(), Bool(), Bool())
    for i in range(n):
        yield ["equ", r(i), ["or", *[ q(i,j) for j in range(n)]]]
    for j in range(n):
        yield ["equ", c(j), ["or", *[ q(i,j) for i in range(n)]]]
    for a in range(0, 2*n-1):
        yield ["equ", u(a), ["or", *[ q(i,j) for (i,j) in U(a)]]]
    for b in range(-n+1, n):
        yield ["equ", d(b), ["or", *[ q(i,j) for (i,j) in D(b)]]]
    for (i,j) in product(range(n), range(n)):
        yield ["or", r(i), c(j), u(i+j), d(i-j)]
    yield ["==", ["+", *[ q(i,j) for i in range(n) for j in range(n) ]], s]
    yield ["<=", ["+", *[ r(i) for i in range(n) ]], s]
    yield ["<=", ["+", *[ c(j) for j in range(n) ]], s]
    yield ["<=", ["+", *[ u(a) for a in range(0, 2*n-1) ]], s]
    yield ["<=", ["+", *[ d(b) for b in range(-n+1, n) ]], s]

def queenDominationOpt(n, s=Var("s"), q=Bool("q")):
    def U(a): return [ (i,a-i) for i in range(n) if a-i in range(n) ]
    def D(b): return [ (i,i-b) for i in range(n) if i-b in range(n) ]
    (r, c, u, d) = (Bool(), Bool(), Bool(), Bool())
    for i in range(n):
        yield ["equ", r(i), ["or", *[ q(i,j) for j in range(n)]]]
    for j in range(n):
        yield ["equ", c(j), ["or", *[ q(i,j) for i in range(n)]]]
    for a in range(0, 2*n-1):
        yield ["equ", u(a), ["or", *[ q(i,j) for (i,j) in U(a)]]]
    for b in range(-n+1, n):
        yield ["equ", d(b), ["or", *[ q(i,j) for (i,j) in D(b)]]]
    for (i,j) in product(range(n), range(n)):
        yield ["or", r(i), c(j), u(i+j), d(i-j)]
    yield ["int", s, 1, n]
    yield ["==", ["+", *[ q(i,j) for i in range(n) for j in range(n) ]], s]
    yield ["<=", ["+", *[ r(i) for i in range(n) ]], s]
    yield ["<=", ["+", *[ c(j) for j in range(n) ]], s]
    yield ["<=", ["+", *[ u(a) for a in range(0, 2*n-1) ]], s]
    yield ["<=", ["+", *[ d(b) for b in range(-n+1, n) ]], s]
    yield ["minimize", s]

def qidp(n, s, q=Bool("q"), r=Bool("r"), c=Bool("c"), u=Bool("u"), d=Bool("d")):
    for i in range(n):
        yield ["==", r(i), ["+", *[ q(i,j) for j in range(n)]]]
    for j in range(n):
        yield ["==", c(j), ["+", *[ q(i,j) for i in range(n)]]]
    for a in range(0, 2*n-1):
        qq = [ q(i,a-i) for i in range(n) if a-i in range(n) ]
        yield ["==", u(a), ["+", *qq]]
    for b in range(-n+1, n):
        qq = [ q(i,i-b) for i in range(n) if i-b in range(n) ]
        yield ["==", d(b), ["+", *qq]]
    for (i,j) in product(range(n), range(n)):
        yield ["or", r(i), c(j), u(i+j), d(i-j)]
    yield ["eqK", [ r(i) for i in range(n) ], s]
    yield ["eqK", [ c(j) for j in range(n) ], s]
    yield ["eqK", [ u(a) for a in range(0, 2*n-1) ], s]
    yield ["eqK", [ d(b) for b in range(-n+1, n) ], s]

def dominatingSet(vertices, edges, k=None, x=Bool("x")):
    def adj(v):
        return [ e[1-i] for e in edges for i in [0,1] if e[i] == v ]
    for v in vertices:
        yield ["or", x(v), *[ x(u) for u in adj(v) ]]
    if k:
        yield ["leK", [ x(v) for v in vertices ], k]

def independentSet(vertices, edges, k=None, x=Bool("x")):
    for (u,v) in edges:
        yield ["leK", [x(u), x(v)], 1]
    if k:
        yield ["geK", [ x(v) for v in vertices ], k]

def squarePacking(s, size, x=Var("x"), y=Var("y")):
    n = len(s)
    for i in range(n):
        yield ["int", x(i), 0, size-s[i]]
        yield ["int", y(i), 0, size-s[i]]
    for (i,j) in combinations(range(n), 2):
        yield ["or",
                      ["<=", ["+", x(i), s[i]], x(j)],
                      ["<=", ["+", x(j), s[j]], x(i)],
                      ["<=", ["+", y(i), s[i]], y(j)],
                      ["<=", ["+", y(j), s[j]], y(i)]
        ]
    yield ["<=", x(0), ["-", size, ["+", x(0), s[0]]]]
    yield ["<=", y(0), ["-", size, ["+", y(0), s[0]]]]
    yield [">=", x(0), y(0)]

def showSquarePacking(sol, s, size, x=Var("x"), y=Var("y")):
    p = [ [ "." for b in range(size) ] for a in range(size) ]
    for i in range(len(s)):
        (a, b) = (sol[y(i)], sol[x(i)])
        for (da,db) in product(range(s[i]), range(s[i])):
            p[a+da][b+db] = chr(ord("A")+i) if i < 26 else chr(ord("a")+i-26)
    for a in reversed(range(size)):
        print("".join(p[a]))

def squarePackingOpt(n, x=Var("x"), y=Var("y"), size=Var("size")):
    sizeLb = math.ceil(math.sqrt(n*(n+1)*(2*n+1)/6))
    sizeUb = sizeLb + 2
    yield from squarePacking(range(1,n+1), sizeUb)
    yield ["int", size, sizeLb, sizeUb]
    for i in range(n):
        yield ["<=", ["+", x(i), i+1], size]
        yield ["<=", ["+", y(i), i+1], size]
    yield ["minimize", size]

def quadFree(n, k, x=Bool("x")):
    for (i,i1) in combinations(range(n), 2):
        for (j,j1) in combinations(range(n), 2):
            yield ["leK", [x(i,j), x(i,j1), x(i1,j), x(i1,j1)], 3]
    xx = [ x(i,j) for i in range(n) for j in range(n) ]
    yield ["geK", xx, k]

def quadFreeSb(n, k, x=Bool("x")):
    yield from quadFree(n, k)
    for i in range(n-1):
        yield ["lexCmp", "<=", [ x(i,j) for j in range(n) ], [ x(i+1,j) for j in range(n) ]]
    for j in range(n-1):
        yield ["lexCmp", "<=", [ x(i,j) for i in range(n) ], [ x(i,j+1) for i in range(n) ]]

def rbibd1(v, b, r, k, x=Bool("x")):
    s = b//r
    for (i,d) in product(range(v), range(r)):
        yield ["eqK", [ x(i,s*d+j) for j in range(s) ], 1]
    for j in range(b):
        yield ["eqK", [ x(i,j) for i in range(v) ], k]
    for (i, i1) in combinations(range(v), 2):
        for (j, j1) in combinations(range(b), 2):
            yield ["leK", [x(i,j), x(i,j1), x(i1,j), x(i1,j1)], 3]

def rbibd1sb(v, b, r, k, x=Bool("x")):
    yield from rbibd1(v, b, r, k, x=x)
    s = b//r
    for j in range(s):
        yield ["and", *[ x(k*j+i,j) for i in range(k) ]]
    for i in range(k):
        yield ["and", *[ x(i,s*d+i) for d in range(1,r) ]]

def cyclicDiffSet1(v, k, x=Bool("x")):
    yield ["eqK", [ x(i) for i in range(v) ], k]
    for d in range(1,v//2+1):
        yield ["or", *[ ["and", x(i), x((i+d)%v)] for i in range(v) ]]
    yield x(0)
    yield x(1)

def golombRuler(m, n, x=Bool("x")):
    yield ["eqK", [ x(i) for i in range(n+1) ], m]
    yield ["and", x(0), x(n)]
    for (i,j) in combinations(range(n+1), 2):
        for d in range(1,n):
            if i+d <= j and j+d <= n:
                yield ["or", ~x(i), ~x(i+d), ~x(j), ~x(j+d)]

def sparsebRuler(m, n, x=Bool("x")):
    yield ["eqK", [ x(i) for i in range(n+1) ], m]
    yield ["and", x(0), x(n)]
    for d in range(1,n+1):
        yield ["or", *[ ["and", x(i), x(i+d)] for i in range(n) if i+d <= n ]]
    yield x(1)

def nonogram(rows, cols, x=Bool("x")):
    (m, n) = (len(rows), len(cols))
    (h, v) = (Var(), Var())
    for i in range(m):
        clues = rows[i]
        for k in range(len(clues)):
            yield ["int", h(i,k), 0, n-clues[k]]
        for k in range(len(clues)-1):
            yield ["<", ["+", h(i,k), clues[k]], h(i,k+1)]
    for j in range(n):
        clues = cols[j]
        for k in range(len(clues)):
            yield ["int", v(j,k), 0, m-clues[k]]
        for k in range(len(clues)-1):
            yield ["<", ["+", v(j,k), clues[k]], v(j,k+1)]
    for (i,j) in product(range(m), range(n)):
        (clues1, clues2) = (rows[i], cols[j])
        cs1 = [ ["and", ["<=", h(i,k), j], ["<", j, ["+", h(i,k), clues1[k]]]] for k in range(len(clues1)) ]
        yield ["equ", x(i,j), ["or", *cs1]]
        cs2 = [ ["and", ["<=", v(j,k), i], ["<", i, ["+", v(j,k), clues2[k]]]] for k in range(len(clues2)) ]
        yield ["equ", x(i,j), ["or", *cs2]]

def nfa(sigma, delta, fins, xx):
    (m, n) = (len(delta), len(xx))
    s = Var()
    for i in range(n+1):
        yield ["int", s(i), 0, m-1]
    yield ["==", s(0), 0]
    yield ["or", *[ ["==", s(n), q] for q in fins ]]
    for (i,q,a) in product(range(n), range(m), sigma):
        c1 = ["and", ["==", s(i), q], ["==", xx[i], a]]
        c2 = [ ["==", s(i+1), q1] for q1 in delta[q].get(a, []) ]
        yield ["imp", c1, ["or", *c2]]

def pattern13(x=Var("x")):
    sigma = [0, 1]
    delta = [ {0:{0},1:{1}}, {0:{2}}, {0:{2},1:{3}}, {1:{4}}, {1:{5}}, {0:{5}} ]
    fins = [5]
    n = 7
    for i in range(n):
        yield ["int", x(i), 0, 1]
    yield from nfa(sigma, delta, fins, [ x(i) for i in range(n) ])

def nfaByPattern(pattern, xx):
    if pattern == [1]:
        delta = [ {0:{0},1:{1}}, {0:{1}} ]
        yield from nfa([0,1], delta, [1], xx)
    elif pattern == [3]:
        delta = [ {0:{0},1:{1}}, {1:{2}}, {1:{3}}, {0:{3}} ]
        yield from nfa([0,1], delta, [3], xx)
    elif pattern == [1,3]:
        delta = [ {0:{0},1:{1}}, {0:{2}}, {0:{2},1:{3}}, {1:{4}}, {1:{5}}, {0:{5}} ]
        yield from nfa([0,1], delta, [5], xx)
    else:
        raise Exception(f"Unknown NFA pattern: {pattern}")

def nonogramNFA(rows, cols, x=Var("x")):
    (m, n) = (len(rows), len(cols))
    for (i,j) in product(range(m), range(n)):
        yield ["int", x(i,j), 0, 1]
    for i in range(m):
        yield from nfaByPattern(rows[i], [ x(i,j) for j in range(n) ])
    for j in range(n):
        yield from nfaByPattern(cols[j], [ x(i,j) for i in range(n) ])

def knightGraph(n):
    vertices = [ (i,j) for i in range(n) for j in range(n) ]
    edges = []
    for (i,j) in vertices:
        for (k,l) in [(i+1,j-2), (i+1,j+2), (i+2,j-1), (i+2,j+1)]:
            if k in range(n) and l in range(n):
                edges.append(((i,j), (k,l)))
    return (vertices, edges)

def hamiltonianCycle(vertices, edges, x=Var("x"), a=Bool("a")):
    def adj(v):
        return [ e[1-i] for e in edges for i in [0,1] if e[i] == v ]
    for (u,v) in edges:
        yield ["<=", ["+", a(u,v), a(v,u)], 1]
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

def symmetricKnightTour(n, x=Var("x"), a=Bool("a")):
    (vertices, edges) = knightGraph(n)
    yield from hamiltonianCycle(vertices, edges, x=x, a=a)
    for (u,v) in edges:
        ((i1,j1), (i2,j2)) = (u, v)
        (u2, v2) = ((n-1-i1,n-1-j1), (n-1-i2,n-1-j2))
        if (u,v) < (u2,v2):
            yield ["equ", a(u,v), a(u2,v2)]
            yield ["equ", a(v,u), a(v2,u2)]

def hamiltonianPath(vertices, edges, x=Var("x"), a=Bool("a")):
    s = None
    edges = edges + [ (s,v) for v in vertices ]
    vertices = [s] + vertices
    yield from hamiltonianCycle(vertices, edges, x=x, a=a)

def uncrossedKnightPath(n, minLen, maxLen=None, e=Bool("e"), x=Var("x")):
    def side(w, line):
        (x,y) = w
        ((x1,y1),(x2,y2)) = line
        s = (x1-x2)*(y-y1) + (y1-y2)*(x1-x)
        return -1 if s < 0 else 0 if s == 0 else 1
    def crossing(e1, e2):
        return side(e1[0], e2)*side(e1[1], e2) < 0 and side(e2[0], e1)*side(e2[1], e1) < 0
    (vertices, edges) = knightGraph(n)
    yield from singlePath(vertices, edges, minLen=minLen, maxLen=maxLen, e=e, x=x)
    for (e1,e2) in combinations(edges, 2):
        if crossing(e1, e2):
            ((u1,v1), (u2,v2)) = (e1, e2)
            yield ["<=", ["+", e(u1,v1), e(u2,v2)], 1]

def uncrossedKnightTour(n, minLen, maxLen=None, e=Bool("e"), x=Var("x")):
    def side(w, line):
        (x,y) = w
        ((x1,y1),(x2,y2)) = line
        s = (x1-x2)*(y-y1) + (y1-y2)*(x1-x)
        return -1 if s < 0 else 0 if s == 0 else 1
    def crossing(e1, e2):
        return side(e1[0], e2)*side(e1[1], e2) < 0 and side(e2[0], e1)*side(e2[1], e1) < 0
    (vertices, edges) = knightGraph(n)
    yield from singleCycle(vertices, edges, minLen=minLen, maxLen=maxLen, e=e, x=x)
    for (e1,e2) in combinations(edges, 2):
        if crossing(e1, e2):
            ((u1,v1), (u2,v2)) = (e1, e2)
            yield ["<=", ["+", e(u1,v1), e(u2,v2)], 1]

def goishiGraph(board):
    (m, n) = (len(board), len(board[0]))
    vertices = [ (i,j) for i in range(m) for j in range(n) if board[i][j] != "." ]
    edges = []
    for (i,j) in vertices:
        edges.extend([ ((i,j),(k,j)) for k in range(i+1,m) if (k,j) in vertices ])
        edges.extend([ ((i,j),(i,l)) for l in range(j+1,n) if (i,l) in vertices ])
    return (vertices, edges)

def between(w, u, v):
    if w[0] == u[0] == v[0]:
        return min(u[1],v[1]) < w[1] < max(u[1],v[1])
    if w[1] == u[1] == v[1]:
        return min(u[0],v[0]) < w[0] < max(u[0],v[0])
    return False

def goishiHiroi(board, a=Bool("a"), x=Var("x")):
    (vertices, edges) = goishiGraph(board)
    yield from hamiltonianPath(vertices, edges, x=x, a=a)
    for v in vertices:
        us1 = [ u for u in vertices if u[0] == v[0] and u[1] < v[1] ]
        us2 = [ u for u in vertices if u[0] == v[0] and u[1] > v[1] ]
        us3 = [ u for u in vertices if u[0] < v[0] and u[1] == v[1] ]
        us4 = [ u for u in vertices if u[0] > v[0] and u[1] == v[1] ]
        for us in [us1,us2,us3,us4]:
            for [u1,u2] in permutations(us, 2):
                yield ["or", ~a(u1,v), ~a(v,u2)]
    for (u,v) in edges:
        ws = [ w for w in vertices if between(w, u, v) ]
        cs = [ ["and", ["<", x(w), x(u)], ["<", x(w), x(v)]] for w in ws ]
        yield ["imp", ["or", a(u,v), a(v,u)], ["and", *cs]]

def ginDigraph(m, n):
    vertices = [ (i,j) for i in range(m) for j in range(n) ]
    arcs = []
    for (i,j) in vertices:
        for (k,l) in [(i-1,j-1), (i-1,j), (i-1,j+1), (i+1,j-1), (i+1,j+1)]:
            if k in range(m) and l in range(n):
                arcs.append(((i,j),(k,l)))
    return (vertices, arcs)

def gridGraph(m, n):
    vertices = [ (i,j) for i in range(m) for j in range(n) ]
    edges = []
    for (i,j) in vertices:
        for (k,l) in [(i,j+1), (i+1,j)]:
            if k in range(m) and l in range(n):
                edges.append(((i,j), (k,l)))
    return (vertices, edges)

def numberlink(m, n, links, a=Bool("a"), d=Bool("d"), x=Var("x")):
    (vertices, edges) = gridGraph(m, n)
    def adj(v):
        return [ e[1-i] for e in edges for i in [0,1] if e[i] == v ]
    num = { v: h for h in links for v in links[h] }
    for v in vertices:
        yield ["int", x(v), 1, max(links.keys())]
        h = num.get(v, 0)
        if h:
            yield ["==", x(v), h]
            (inDeg, outDeg) = (0, 1) if v == links[h][0] else (1, 0)
        else:
            (inDeg, outDeg) = (d(v), d(v))
        yield ["==", ["+", *[ a(u,v) for u in adj(v) ]], inDeg]
        yield ["==", ["+", *[ a(v,u) for u in adj(v) ]], outDeg]
    for (u,v) in edges:
        yield ["imp", ["or", a(u,v), a(v,u)], ["==", x(u), x(v)]]

def singleCycle(vertices, edges, minLen=None, maxLen=None, e=Bool("e"), a=Bool("a"), d=Bool("d"), r=Bool("r"), x=Var("x")):
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

def slitherlink(clues, e=Bool("e")):
    def surrounds(i, j):
        es = [ ((i,j),(i,j+1)), ((i,j),(i+1,j)), ((i,j+1),(i+1,j+1)), ((i+1,j),(i+1,j+1)) ]
        return [ e(u,v) for (u,v) in es ]
    (m, n) = (len(clues), len(clues[0]))
    (vertices, edges) = gridGraph(m+1, n+1)
    yield from singleCycle(vertices, edges, e=e)
    for (i,j) in product(range(m), range(n)):
        h = clues[i][j]
        if h <= 4:
            yield ["eqK", surrounds(i, j), h]

def showSlitherlink(clues, sol, e=Bool("e")):
    (m, n) = (len(clues), len(clues[0]))
    for i in range(m+1):
        line = "+"
        for j in range(n):
            line += "---+" if sol[e((i,j),(i,j+1))] else "   +"
        print(line)
        if i == m:
            continue
        line = ""
        for j in range(n+1):
            line += "|" if sol[e((i,j),(i+1,j))] else "."
            if j == n:
                continue
            h = clues[i][j]
            line += f" {h} " if h <= 4 else "   "
        print(line)

def uniqSol(csp):
    sols = list(solutionsCSP(csp, num=2))
    return len(sols) == 1

def life(steps, m, n, x=Bool("x")):
    def surrounds(i, j):
        kls = [ kl for kl in product([i-1,i,i+1], [j-1,j,j+1]) if kl != (i,j) ]
        return [ (k,l) for (k,l) in kls if 0 <= k < m and 0 <= l < n ]
    def state(t):
        yield from [ ~x(t,i,j) for i in [0,m-1] for j in range(n) ]
        yield from [ ~x(t,i,j) for i in range(m) for j in [0,n-1] ]
    def transition(t):
        for (i,j) in product(range(m), range(n)):
            s = Var()
            yield ["int", s, 0, 8]
            yield ["==", s, ["+", *[ x(t,k,l) for (k,l) in surrounds(i, j) ]]]
            yield ["equ", x(t+1,i,j), ["or", ["==", s, 3], ["and", x(t,i,j), ["==", s, 2]]]]

def lifePattern(t, pattern, x=Bool("x")):
    for (i,row) in enumerate(pattern):
        for (j,p) in enumerate(row):
            yield ~x(t,i,j) if p == "." else x(t,i,j)

def showLife(sol, steps, m, n, x=Bool("x")):
    for t in range(steps+1):
        c = 0
        for i in range(m):
            ss = [ sol[x(t,i,j)] for j in range(n) ]
            c += sum(ss)
            print("".join([ "#" if s else "." for s in ss ]))
        print(f"Step {t}: {c} cells")
        print()

def lifeLim(m, n, limits=[], x=Bool("x")):
    for (t,(lb,ub)) in enumerate(limits):
        xx = [ x(t,i,j) for i in range(m) for j in range(n) ]
        if isinstance(lb, int):
            yield ["geK", xx, lb]
        if isinstance(ub, int):
            yield ["leK", xx, ub]

def lifeMove(m, n, t1, t2, di, dj, x=Bool("x")):
    for (i,j) in product(range(m), range(n)):
        if i+di in range(m) and j+dj in range(n):
            yield ["equ", x(t2,i+di,j+dj), x(t1,i,j)]
        else:
            yield ~x(t1,i,j)

def isWall(i, j, board): return board[i][j] == "#"

def isBox(i, j, board): return board[i][j] in "$*"

def sokoban(board, steps, p=Bool("p"), b=Bool("b"), g=Bool("g"), di=Var("di"), dj=Var("dj")):
    (m, n) = (len(board), len(board[0]))
    floor = { (i,j) for i in range(m) for j in range(n) if not isWall(i, j, board) }
    boxes = [ (i,j) for (i,j) in floor if isBox(i, j, board) ]
    goals = [ (i,j) for (i,j) in floor if isGoal(i, j, board) ]
    def state(t):
        yield ["eqK", [ p(t,i,j) for (i,j) in floor ], 1]
        yield ["eqK", [ b(t,i,j) for (i,j) in floor ], len(boxes)]
        yield ["equ", g(t), ["and", *[ b(t,i,j) for (i,j) in goals ]]]
    def initial(t):
        for (i,j) in floor:
            if isPlayer(i, j, board):
                yield p(t,i,j)
            elif isBox(i, j, board):
                yield b(t,i,j)
    def d0(i, j):
        return [ (k,l) for (k,l) in [(0,0),(-1,0),(1,0),(0,-1),(0,1)] if (i-k,j-l) in floor ]
    def d1(i, j):
        return [ (k,l) for (k,l) in [(-1,0),(1,0),(0,-1),(0,1)] if (i-k,j-l) in floor and (i-2*k,j-2*l) in floor ]
    def transition(t):
        yield ["int", di(t), -1, 1]
        yield ["int", dj(t), -1, 1]
        yield ["or", ["==", di(t), 0], ["==", dj(t), 0]]
        yield ["imp", g(t), ["and", ["==", di(t), 0], ["==", dj(t), 0]]]
        for (i,j) in floor:
            c = ["or"]
            for (k,l) in d0(i, j):
                c.append(["and", ["==", di(t), k], ["==", dj(t), l], p(t,i-k,j-l)])
            yield ["equ", p(t+1,i,j), c]
        for (i,j) in floor:
            c = ["or"]
            for (k,l) in d1(i, j):
                c.append(["and", b(t,i-k,j-l), p(t,i-2*k,j-2*l), p(t+1,i-k,j-l)])
            yield ["equ", b(t+1,i,j), ["or", ["and", ~b(t,i,j), c], ["and", b(t,i,j), ~p(t+1,i,j)]]]

def showSolution(sol, t, board, p=Bool("p"), b=Bool("b"), g=Bool("g")):
    (m, n) = (len(board), len(board[0]))
    floor = { (i,j) for i in range(m) for j in range(n) if not isWall(i, j, board) }
    player = [ (i,j) for (i,j) in floor if sol[p(t,i,j)] ][0]
    boxes = [ (i,j) for (i,j) in floor if sol[b(t,i,j)] ]
    goal = sol[g(t)]
    print(f"Step = {t}")
    print(f"goal = {goal}, player = {player}, boxes = {boxes}")
    for i in range(len(board)):
        s = []
        for j in range(len(board[i])):
            if isWall(i, j, board):
                c = "#"
            elif sol[p(t,i,j)]:
                c = "+" if isGoal(i, j, board) else "@"
            elif sol[b(t,i,j)]:
                c = "*" if isGoal(i, j, board) else "$"
            else:
                c = "." if isGoal(i, j, board) else " "
            s.append(c)
        print("".join(s))
    print()

def sokobanP(board, steps):
    (m, n) = (len(board), len(board[0]))
    floor = { (i,j) for i in range(m) for j in range(n) if not isWall(i, j, board) }
    edges = [ ((i,j),(i+di,j+dj)) for (i,j) in floor for (di,dj) in [(1,0),(0,1)] if (i+di,j+dj) in floor ]
    boxes = [ (i,j) for (i,j) in floor if isBox(i, j, board) ]
    goals = [ (i,j) for (i,j) in floor if isGoal(i, j, board) ]
    (p, r, b, g) = (Bool("p"), Bool("r"), Bool("b"), Bool("g"))
    (di, dj) = (Var("di"), Var("dj"))
    def state(t):
        yield ["eqK", [ p(t,i,j) for (i,j) in floor ], 1]
        yield ["eqK", [ b(t,i,j) for (i,j) in floor ], len(boxes)]
        yield ["equ", g(t), ["and", *[ b(t,i,j) for (i,j) in goals ]]]
    def initial(t):
        for (i,j) in floor:
            if isPlayer(i, j, board):
                yield p(t,i,j)
            elif isBox(i, j, board):
                yield b(t,i,j)
    def d0(i, j):
        return [ (k,l) for (k,l) in [(0,0),(-1,0),(1,0),(0,-1),(0,1)] if (i-k,j-l) in floor ]
    def d1(i, j):
        return [ (k,l) for (k,l) in [(-1,0),(1,0),(0,-1),(0,1)] if (i-k,j-l) in floor and (i-2*k,j-2*l) in floor ]
    def reachability(t):
        (a, inD, outD) = (Bool(), Bool(), Bool())
        for (u,v) in edges:
            yield ["leK", [a(t,u,v), a(t,v,u)], 1]
        for v in floor:
            (i,j) = v
            us = [ e[1-i] for e in edges for i in [0,1] if e[i] == v ]
            yield ["==", ["+", *[ a(t,u,v) for u in us ]], inD(t,i,j)]
            yield ["==", ["+", *[ a(t,v,u) for u in us ]], outD(t,i,j)]
            yield ["imp", p(t,i,j), ["or", r(t,i,j), ["and", ~inD(t,i,j), outD(t,i,j)]]]
            yield ["imp", r(t,i,j), ["or", p(t,i,j), ["and", inD(t,i,j), ~outD(t,i,j)]]]
            yield ["imp", ["and", ~p(t,i,j), ~r(t,i,j)], ["==", inD(t,i,j), outD(t,i,j)]]
            yield ["imp", b(t,i,j), ["and", ~inD(t,i,j), ~outD(t,i,j)]]
    def transition(t):
        yield ["int", di(t), -1, 1]
        yield ["int", dj(t), -1, 1]
        yield ["or", ["==", di(t), 0], ["==", dj(t), 0]]
        yield ["imp", g(t), ["and", ["==", di(t), 0], ["==", dj(t), 0]]]
        yield from reachability(t)
        for (i,j) in floor:
            c = ["or"]
            for (k,l) in d0(i, j):
                c.append(["and", ["==", di(t), k], ["==", dj(t), l], r(t,i-k,j-l)])
            yield ["equ", p(t+1,i,j), c]
        for (i,j) in floor:
            c = ["or"]
            for (k,l) in d1(i, j):
                c.append(["and", b(t,i-k,j-l), r(t,i-2*k,j-2*l), p(t+1,i-k,j-l)])
            yield ["equ", b(t+1,i,j), ["or", ["and", ~b(t,i,j), c], ["and", b(t,i,j), ~p(t+1,i,j)]]]

def pegSolitaire(board0, board1, steps, x=Bool("x"), p=Bool("p")):
    (m, n) = (len(board0), len(board0[0]))
    vertices = [ (i,j) for i in range(m) for j in range(n) if board0[i][j] != " " ]
    dirs = [(-1,0), (0,-1), (0,1), (1,0)]
    def adj2(u):
        (i,j) = u
        return [ ((i+di,j+dj),(i+2*di,j+2*dj)) for (di,dj) in dirs if (i+di,j+dj) in vertices and (i+2*di,j+2*dj) in vertices ]
    def setBoard(t, board):
        for u in vertices:
            (i,j) = u
            if board[i][j] == ".":
                yield ~x(t,u)
            elif board[i][j] == "#":
                yield x(t,u)
    def transition(t):
        yield ["eqK", [ p(t,u,w) for u in vertices for (_,w) in adj2(u) ], 1]
        for u in vertices:
            for (v,w) in adj2(u):
                yield ["imp", p(t,u,w), ["and", x(t,u), x(t,v), ~x(t,w)]]
        for u in vertices:
            f = [ p(t,v1,v3) for v1 in vertices for (v2,v3) in adj2(v1) if u in [v1,v2,v3] ]
            yield ["equ", x(t+1,u), ["xor", x(t,u), ["or", *f]]]

# END notebook/07-csp-examples.org

