from itertools import combinations, permutations
from cspsat import *

def chr2int(c):
    return ord(c) - ord("A")

def int2chr(i):
    return chr(ord("A") + i)

def str2ints(s):
    return [ chr2int(c) if isinstance(c, str) else c for c in s ]

def ints2str(ii):
    return "".join([ int2chr(i) for i in ii ])

enigmaRotors = {
    1: str2ints("EKMFLGDQVZNTOWYHXUSPAIBRCJ"),
    2: str2ints("AJDKSIRUXBLHWTMCQGZNPYFVOE"),
    3: str2ints("BDFHJLCPRTXVZNYEIWGAKMUSQO"),
    4: str2ints("ESOVPZJAYQUIRHXLNFTGKDCMWB"),
    5: str2ints("VZBRGITYUPSDNHLXAWMJQOFECK")
}

enigmaRotorTurnovers = {
    1: chr2int("R"),
    2: chr2int("F"),
    3: chr2int("W"),
    4: chr2int("K"),
    5: chr2int("A")
}

enigmaReflector = str2ints("YRUHQSLDPXNGOKMIEBFZCWVJAT") # B reflector

class Enigma:

    def __init__(self, verbose=0):
        self.nChars = 26
        self.k = 3
        self.l = 3
        self.r = Var("r")
        self.p = Var("p")
        self.s = Var("s")
        self.t = Var("t")
        self.w = Var("w")
        self.verbose = verbose

    def params(self):
        return (self.nChars, self.k, self.l, self.r, self.p, self.s, self.t, self.w)

    def enigmaDef(self, useW=True, maxSteps=None):
        (n, k, l, r, p, s, t, w) = self.params()
        for j in range(3):
            yield ["int", r(j), 1, k]
        yield ["alldifferent", *[ r(j) for j in range(3) ]]
        for j in range(3):
            yield ["int", p(j), 0, n-1]
        for a in range(n):
            yield ["int", s(a), 0, n-1]
        for (a,b) in combinations(range(n), 2):
            yield ["equ", ["==", s(a), b], ["==", s(b), a]]
        q = Bool()
        for a in range(n):
            yield ["equ", q(a), ["!=", s(a), a]]
        yield ["leK", [ q(a) for a in range(n) ], 2*l]
        yield ["int", t, 1, maxSteps or n]
        if useW:
            yield from self.enigmaDefW()

    def enigmaDefW(self):
        (n, k, l, r, p, s, t, w) = self.params()
        for o in [0,1]:
            for a in range(n):
                yield ["int", w(o,a), 0, n-1]
            ww = {}
            for (j0,j1) in permutations(range(1,k+1), 2):
                r0 = enigmaRotors[j0]
                r0x = dict((b,a) for (a,b) in enumerate(r0))
                r1 = enigmaRotors[j1]
                r1x = dict((b,a) for (a,b) in enumerate(r1))
                for p0 in range(n):
                    for p1 in range(n):
                        for a in range(n):
                            b = (r1[(a+p1+o)%n]-p1-o)%n
                            b = (r0[(b+p0)%n]-p0)%n
                            b = enigmaReflector[b]
                            b = (r0x[(b+p0)%n]-p0)%n
                            b = (r1x[(b+p1+o)%n]-p1-o)%n
                            if (a,b) not in ww:
                                ww[(a,b)] = []
                            ww[(a,b)].append(["and", ["==", r(0), j0], ["==", r(1), j1], ["==", p(0), p0], ["==", p(1), p1]])
            for (a,b) in ww.keys():
                yield ["equ", ["==", w(o,a), b], ["or", *ww[(a,b)]]]

    def plugboard(self, x, z):
        (n, k, l, r, p, s, t, w) = self.params()
        for a in range(n):
            yield ["equ", ["==", x, a], ["==", z, s(a)]]

    def reflector(self, x, z):
        (n, k, l, r, p, s, t, w) = self.params()
        for a in range(n):
            yield ["equ", ["==", x, a], ["==", z, enigmaReflector[a]]]

    def rotor(self, i, j, x, z):
        (n, k, l, r, p, s, t, w) = self.params()
        if j == 2:
            dp = i
        elif j == 1:
            dp = ["if", [">=", i, t], 1, 0]
        else:
            dp = 0
        y = Var()
        yield ["int", y(0), 0, n-1]
        yield ["int", y(1), 0, n-1]
        yield ["==", y(0), ["mod", ["+", x, p(j), dp], n]]
        yield ["==", y(1), ["mod", ["+", z, p(j), dp], n]]
        for d in enigmaRotors.keys():
            for a in range(n):
                yield ["imp", ["==", r(j), d], ["equ", ["==", y(0), a], ["==", y(1), enigmaRotors[d][a]]]]

    def encipher(self, i, x, z, useW=True):
        (n, k, l, r, p, s, t, w) = self.params()
        y = Var()
        if useW:
            for j in range(4):
                yield ["int", y(j), 0, n-1]
            yield from self.plugboard(x, y(0))
            yield from self.rotor(i, 2, y(0), y(1))
            for a in range(n):
                yield ["imp", ["<", i, t], ["equ", ["==", y(1), a], ["==", y(2), w(0,a)]]]
                yield ["imp", ["<", i, t], ["equ", ["==", y(1), w(0,a)], ["==", y(2), a]]]
                yield ["imp", [">=", i, t], ["equ", ["==", y(1), a], ["==", y(2), w(1,a)]]]
                yield ["imp", [">=", i, t], ["equ", ["==", y(1), w(1,a)], ["==", y(2), a]]]
            yield from self.rotor(i, 2, y(3), y(2))
            yield from self.plugboard(y(3), z)
        else:
            for j in range(8):
                yield ["int", y(j), 0, n-1]
            yield from self.plugboard(x, y(0))
            yield from self.rotor(i, 2, y(0), y(1))
            yield from self.rotor(i, 1, y(1), y(2))
            yield from self.rotor(i, 0, y(2), y(3))
            yield from self.reflector(y(3), y(4))
            yield from self.rotor(i, 0, y(5), y(4))
            yield from self.rotor(i, 1, y(6), y(5))
            yield from self.rotor(i, 2, y(7), y(6))
            yield from self.plugboard(y(7), z)

    def showSol(self, sol):
        (n, k, l, r, p, s, t, w) = self.params()
        rs = [ sol[r(j)] for j in range(3) ]
        ps = ints2str([ sol[p(j)] for j in range(3) ])
        ss = []
        for a in range(n):
            if sol[s(a)] > a:
                ss.append(ints2str([a, sol[s(a)]]))
        turn = sol[t]
        pos2 = (enigmaRotorTurnovers[sol[r(2)]]-turn)%n
        pos = ints2str([sol[p(0)], sol[p(1)], pos2])
        ring2 = (pos2-sol[p(2)])%n
        ring = ints2str([0, 0, ring2])
        print(f"Rotors: {rs}")
        print(f"Ring Positions: {ring}")
        print(f"Start Positions: {pos}")
        print(f"Plugboard: {ss}")
        print(f"Turnover: {turn} step")

    def enigmaSim(self, rotors, rings, positions, plugs, xx, z=Var("z"), useW=True):
        (n, k, l, r, p, s, t, w) = self.params()
        yield from self.enigmaDef(useW=useW)
        for j in range(3):
            yield ["==", r(j), rotors[j]]
        for j in range(3):
            yield ["==", p(j), (chr2int(positions[j])-chr2int(rings[j]))%n]
        pp = dict((chr2int(cs[k]),chr2int(cs[1-k])) for cs in plugs for k in [0,1])
        for a in range(n):
            yield ["==", s(a), pp[a] if a in pp else a]
        turnStep = (enigmaRotorTurnovers[rotors[2]]-chr2int(positions[2]))%n
        yield ["==", t, turnStep]
        for (i,x) in enumerate(xx):
            yield ["int", z(i), 0, n-1]
            yield from self.encipher(i+1, chr2int(x), z(i), useW=useW)

    def enigmaBreak(self, plain, cipher, useW=True):
        (n, k, l, r, p, s, t, w) = self.params()
        xx = str2ints(plain)
        zz = str2ints(cipher)
        yield from self.enigmaDef(useW=useW, maxSteps=len(xx)+1)
        for i in range(len(plain)):
            yield from self.encipher(i+1, xx[i], zz[i], useW=useW)
