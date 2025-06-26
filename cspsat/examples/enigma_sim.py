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

enigmaReflectors = {
    1: str2ints("YRUHQSLDPXNGOKMIEBFZCWVJAT") # B reflector
}

class EnigmaSim:

    def __init__(self, verbose=0):
        self.nChars = 26
        self.nRotors = 3
        self.reflector = 1
        self.rotors = [ j+1 for j in range(self.nRotors) ]
        self.ringPositions = [0] * self.nRotors
        self.startPositions = [0] * self.nRotors
        self.plugboard = dict((a,a) for a in range(self.nChars))
        self.verbose = verbose

    def setRotors(self, rotors):
        self.rotors = rotors

    def setRingPositions(self, ringPositions):
        self.ringPositions = str2ints(ringPositions)

    def setStartPositions(self, startPositions):
        self.startPositions = str2ints(startPositions)

    def setPlugboard(self, plugConnections):
        self.plugboard = dict((a,a) for a in range(self.nChars))
        for [a1,a2] in [ str2ints(s) for s in plugConnections ]:
            self.plugboard[a1] = a2
            self.plugboard[a2] = a1

    def applyPlugboard(self, x):
        z = self.plugboard[x]
        self.trace.append(z)
        return z

    def applyReflector(self, x):
        perm = enigmaReflectors[self.reflector]
        z = perm[x]
        self.trace.append(z)
        return z

    def getPositions(self, step):
        pos = [0] * self.nRotors
        for j in range(self.nRotors):
            p = enigmaRotorTurnovers[self.rotors[j]]
            pos[j] = (self.startPositions[j] - p) % self.nChars
        n = 0
        for j in range(self.nRotors):
            m = self.nChars - (1 if j == self.nRotors-2 else 0)
            n = m*n + pos[j]
        n += step
        for j in reversed(range(self.nRotors)):
            m = self.nChars - (1 if j == self.nRotors-2 else 0)
            pos[j] = n % m
            n //= m
        if pos[-1] == 0 and pos[-2] == 0:
            pos[-2] -= 1
            pos[-3] -= 1
        for j in range(self.nRotors):
            p = enigmaRotorTurnovers[self.rotors[j]]
            pos[j] = (pos[j] + p) % self.nChars
        return pos

    def applyRotor(self, j, step, x, inverse=False):
        perm = enigmaRotors[self.rotors[j]]
        if inverse:
            perm = dict((b,a) for (a,b) in enumerate(perm))
        pos = self.getPositions(step)
        d = pos[j] - self.ringPositions[j]
        x = (x + d) % self.nChars
        z = perm[x]
        z = (z - d) % self.nChars
        self.trace.append(z)
        return z

    def encipher(self, step, x):
        self.trace = [x]
        x = self.applyPlugboard(x)
        for j in reversed(range(self.nRotors)):
            x = self.applyRotor(j, step, x)
        x = self.applyReflector(x)
        for j in range(self.nRotors):
            x = self.applyRotor(j, step, x, inverse=True)
        x = self.applyPlugboard(x)
        if self.verbose >= 1:
            print(f"Positions: {ints2str(self.getPositions(step))}")
        if self.verbose >= 2:
            t = " -> ".join([ int2chr(a) for a in self.trace ])
            print(f"Translation: {t}")
        return x

    def encipherStr(self, plain, i=1):
        cipher = ""
        for (di,c) in enumerate(plain):
            c1 = self.encipher(i+di, chr2int(c))
            cipher += int2chr(c1)
        return cipher
