"""DPLLソルバーのモジュール．

以下のクラスからなる．

* DPLLクラス: 命題変数のクラス

Note:
    本プログラムは学習用の目的で作成されている．
    実用上の問題への適用は想定していない．
"""

class DPLL():
    """DPLLソルバーのクラス．

    Args:
        verbose (int, optional): 正ならDPLLソルバーの情報を表示する (デフォルト値は0)．
        interactive (bool, optional): 真なら会話的に実行する (デフォルト値はFalse)．
    """
    def __init__(self, verbose=0, interactive=False):
        self.verbose = verbose
        self.interactive = interactive
        self.vname2dimacslit = {}
        self.vName = []
        self.watches = []
        self.vAssigns = []
        self.vLevel = []
        self.vReason = []
        self.clauses = []
        self.trail_lim = []
        self.trail = []
        self.propQueue = []
        self.isSat = None
        self.stat = {}

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        return

    def nVars(self):
        """命題変数の個数を返す．

        Returns:
            命題変数の個数．
        """
        return len(self.vAssigns)

    def nClauses(self):
        """節の個数を返す．

        Returns:
            節の個数．
        """
        return len(self.clauses)

    def toLit(self, v, negative=0):
        """変数番号vに対するリテラル番号を返す．

        Args:
            v (int): 変数番号．
            negative (int, optional): 1なら否定 (デフォールト値は0)．

        Returns:
            リテラル番号．
        """
        return (v << 1) + negative

    def toVar(self, lit):
        """リテラルlitに対する変数と符号のタプルを返す．

        Args:
            lit (int): リテラル．

        Returns:
            変数と符号のタプル．
        """
        return (lit >> 1, lit & 1)

    def value(self, lit):
        """リテラルlitの現在の真理値を返す．真理値が割当てられていなければNoneを返す．

        Args:
            lit (int): リテラル．

        Returns:
            litの真理値 (割当てられていなければNone)．
        """
        (v, n) = self.toVar(lit)
        t = self.vAssigns[v]
        return None if t is None else t ^ n

    def newVar(self, vname=None):
        """新しい変数を追加する．

        Args:
            vname (str, optional): 変数名 (デフォールト値はNone)．

        Returns:
            新しい変数番号．
        """
        v = self.nVars()
        self.vName.append(vname)
        self.watches.append([]) # for positive literal
        self.watches.append([]) # for negative literal
        self.vReason.append(None)
        self.vAssigns.append(None)
        self.vLevel.append(None)
        return v

    def decisionLevel(self):
        """現在の決定レベルを返す．

        Returns:
            決定レベル．
        """
        return len(self.trail_lim)

    def enqueue(self, lit, reason=None):
        """リテラルlitを伝播用キューに追加する．

        リテラルlitの値がすでに1であればTrueを返し，伝播用キューには追加しない．
        リテラルlitの値がすでに0であればFalseを返し，伝播用キューには追加しない．
        リテラルlitの値が決まっていなければ伝播用キューには追加し，Trueを返す．

        Args:
            lit (int): リテラル．
            reason (list, optional): 理由の節 (デフォールト値はNone)．

        Returns:
            衝突が生じなければTrue，衝突があればFalse．
        """
        if self.value(lit) is not None:
            return self.value(lit) == 1
        if reason is not None:
            self.info(1, f"propagate '{self.lit2repr(lit)}' by {self.clause2repr(reason)}")
        (v, n) = self.toVar(lit)
        assert self.vAssigns[v] is None
        self.vAssigns[v] = n ^ 1
        self.vLevel[v] = self.decisionLevel()
        self.vReason[v] = reason
        self.trail.append(lit)
        self.propQueue.append(lit)
        return True

    def assume(self, lit):
        """リテラルlitを仮に真と決定する．

        Args:
            lit (int): リテラル．

        Returns:
            衝突が生じなければTrue，衝突があればFalse．
        """
        self.trail_lim.append(len(self.trail))
        self.stat["decisions"] += 1
        self.info(1, f"decide '{self.lit2repr(lit)}'")
        return self.enqueue(lit)

    def propagateClause(self, num, lit):
        """num番目の節に対しリテラルlitの伝播処理を行う．

        Args:
            num (int): 節の番号．
            lit (int): リテラル．

        Returns:
            衝突が生じなければTrue，衝突があればFalse．
        """
        clause = self.clauses[num]
        if clause[0] == lit ^ 1:
            clause[0] = clause[1]
            clause[1] = lit ^ 1
        if self.value(clause[0]) == 1:
            self.watches[lit].append(num)
            return True
        for i in range(2,len(clause)):
            if self.value(clause[i]) != 0:
                clause[1] = clause[i]
                clause[i] = lit ^ 1
                self.watches[clause[1] ^ 1].append(num)
                return True
        self.watches[lit].append(num)
        return self.enqueue(clause[0], clause)

    def propagate(self):
        """伝播処理を行う．

        Returns:
            衝突が生じなければNone，衝突が生じれば衝突のあった節の番号．
        """
        while self.propQueue:
            self.stat["propagations"] += 1
            lit = self.propQueue.pop()
            tmp = self.watches[lit].copy()
            self.watches[lit] = []
            for i, num in enumerate(tmp):
                if not self.propagateClause(num, lit):
                    # confliction at clause number num
                    for j in range(i+1,len(tmp)):
                        self.watches[lit].append(tmp[j])
                    self.propQueue.clear()
                    return self.clauses[num]
        return None

    def cancel(self):
        """1レベルのバックトラック処理を行う．
        """
        c = len(self.trail) - self.trail_lim[-1]
        while c > 0:
            lit = self.trail.pop()
            (v, _) = self.toVar(lit)
            self.vAssigns[v] = None
            self.vReason[v] = None
            self.vLevel[v] = None
            c -= 1
        self.trail_lim.pop()

    def cancelUntil(self, level):
        """決定レベルlevelまでバックトラック処理を行う．

        Args:
            level (int): 決定レベル．
        """
        self.info(1, f"backjump to level {level}")
        while self.decisionLevel() > level:
            self.cancel()

    def newDecision(self):
        """値が未割当ての変数を選択し，それを偽と仮に決定する．

        Returns:
           負リテラル．未割当ての変数がなければNone．
        """
        for v in range(self.nVars()):
            if self.vAssigns[v] is None:
                lit = self.toLit(v, negative=1)
                self.assume(lit)
                return lit
        return None

    def backtrack(self):
        """バックトラック処理を行う．

        最後の決定まで戻り，それが正リテラルの決定の間はさらに前の決定に戻る．
        負リテラルの決定なら，正リテラルにして決定する．
        """
        while True:
            level = self.decisionLevel()
            if level == 0:
                self.isSat = False
                return
            lastDecision = self.trail[self.trail_lim[level-1]]
            self.cancelUntil(level-1)
            (v, t) = self.toVar(lastDecision)
            if t == 1:
                lit = self.toLit(v, t^1)
                self.assume(lit)
                return

    def getModel(self):
        """モデルを返す．

        Returns:
           モデル．
        """
        model = {}
        for v in range(self.nVars()):
            model[self.var2repr(v)] = self.vAssigns[v]
        return model

    def search(self, num=1):
        """モデルの探索を行い，見つかったモデルをyieldするジェネレータ関数．

        Args:
            num (int,optional): 探索するモデルの個数 (0なら全解を探索)．
        
        Yields:
           モデル．
        """
        if self.verbose >= 2:
            self.info(2, "clauses")
            for i, clause in enumerate(self.clauses):
                print(f"  Clause {i}: {self.clause2repr(clause)}", flush=True)
        self.isSat = None
        count = 0
        while self.isSat is not False:
            clause = self.propagate()
            self.info(2, f"decisions {self.decisions2repr()}")
            self.info(2, f"assigns {self.assigns2repr()}")
            self.info(2, f"watches {self.watches2repr()}")
            if clause is not None:
                self.info(1, f"conflict at {self.clause2repr(clause)}")
                self.stat["conflicts"] += 1
                self.backtrack()
            else:
                lit = self.newDecision()
                if lit is None:
                    # Model found
                    self.isSat = True
                    count += 1
                    model = self.getModel()
                    self.info(1, f"found model {count}")
                    yield model
                    if not num and count >= num:
                        break
                    self.backtrack()

    def getStat(self):
        """実行の統計情報を返す．

        Returns:
            統計情報．
        """
        return self.stat

    def solutions(self, clauses=None, num=1):
        """与えられた節を追加し，モデルをyieldするジェネレータ関数．

        Args:
            clauses (list,optional): 追加する節のリスト．
            num (int,optional): 探索するモデルの個数 (0なら全解を探索)．

        Yields:
            モデル．
        """
        if clauses:
            self.add(*clauses)
        t = time.time()
        self.trail_lim = []
        self.trail = []
        self.isSat = None
        self.stat = {}
        self.stat["variables"] = self.nVars()
        self.stat["clauses"] = self.nClauses()
        self.stat["decisions"] = 0
        self.stat["conflicts"] = 0
        self.stat["propagations"] = 0
        for model in self.search(num=num):
            self.stat["solving"] = time.time() - t
            yield model
        if self.decisionLevel() > 0:
            self.cancelUntil(0)
        self.stat["solving"] = time.time() - t

    def solve(self, clauses=None, num=1, stat=False):
        """与えられた節を追加し，モデルを表示する．

        Args:
            clauses (list,optional): 追加する節のリスト．
            num (int,optional): 探索するモデルの個数 (0なら全解を探索)．
            stat (bool,optional): 真なら統計情報も表示する．
        """
        if clauses:
            self.add(*clauses)
        count = 0
        for model in self.solutions(num=num):
            if count == 0:
                print("SATISFIABLE", flush=True)
            count += 1
            print(f"Model {count}: {model}", flush=True)
            if stat:
                print(f"Stat: {self.stat}", flush=True)
        if count == 0:
            print("UNSATISFIABLE", flush=True)
        if stat and (count == 0 or num == 0):
            print(f"Stat: {self.stat}", flush=True)

    def info(self, l, msg):
        """メッセージを表示する．

        表示レベルが self.verbose より大きければ表示しない．
        self.interactive が真ならユーザ入力を待つ．

        Args:
            l (int): 表示レベル．
            msg (str): メッセージ．
        """
        if l <= self.verbose:
            print(f"# {self.decisionLevel()}: {msg}", flush=True)
            if self.interactive:
                print("Press Ctrl-Enter to proceed", flush=True)
                input()

    def var2repr(self, v, negative=0):
        """変数の表記文字列を返す関数．

        newVarで変数名が与えられていればそれを使用する．

        Args:
            v (int): 変数番号．
            negative (int,optional): 1なら負リテラル．

        Returns:
            変数の表記文字列．
        """
        s = self.vName[v]
        if s is None:
            s = str(v+1)
        return s if negative == 0 else "~" + s

    def lit2repr(self, lit):
        """リテラルの表記文字列を返す関数．

        Args:
            lit (int): リテラル．

        Returns:
            リテラルの表記文字列．
        """
        (v, n) = self.toVar(lit)
        return self.var2repr(v, n)

    def clause2repr(self, clause):
        """節の表記 (リテラルの表記文字列のリスト)を返す関数．

        Args:
            clause (list): 節．

        Returns:
            節の表記．
        """
        return None if clause is None else [ self.lit2repr(lit) for lit in clause ]

    def assigns2repr(self):
        """現在の値割当ての表記を返す関数．

        Returns:
            値割当ての表記．
        """
        a = {}
        for v in range(self.nVars()):
            if self.vAssigns[v] is not None:
                a[self.var2repr(v)] = self.vAssigns[v]
        return a

    def decisions2repr(self):
        """現在の決定リストの表記を返す関数．

        Returns:
            決定リストの表記．
        """
        d = []
        for t in self.trail_lim:
            lit = self.trail[t]
            d.append(self.lit2repr(lit))
        return d

    def watches2repr(self):
        """現在の監視リストの表記を返す関数．

        Returns:
            監視リストの表記．
        """
        w = []
        for (lit, num) in enumerate(self.watches):
            s = self.lit2repr(lit)
            w.append((s, num))
        return w

    def addInternalClause(self, clause):
        """内部表現の節 (リテラル番号のリスト)を追加する．

        Args:
            clause (list): 節．
        """
        assert len(clause) > 0
        if len(clause) == 1:
            self.enqueue(clause[0])
        else:
            i = len(self.clauses)
            self.clauses.append(clause)
            self.watches[clause[0]^1].append(i)
            self.watches[clause[1]^1].append(i)

    def addDimacsClause(self, dimacsLits):
        """DIMACS形式の節 (正負の変数番号のリスト)を追加する．

        Args:
            dimacsLits (list): DIMACS形式の節．
        """
        clause = []
        for dimacsLit in dimacsLits:
            v = abs(dimacsLit) - 1
            while v >= self.nVars():
                self.newVar()
            lit = self.toLit(v, 1 if dimacsLit < 0 else 0)
            clause.append(lit)
        self.addInternalClause(clause)

    def toDimacsLit(self, vname):
        """変数名がvnameの変数をDIMACS形式のリテラル (正負の変数番号)に変換する．

        変数名が `~' から始まれば負リテラルとする．

        Args:
            vname (str): 変数名．

        Returns:
            DIMACS形式のリテラル．
        """
        positive = True
        if vname.startswith("~"):
            vname = vname[1:]
            positive = False
        dimacsLit = self.vname2dimacslit.get(vname, 0)
        if dimacsLit == 0:
            dimacsLit = self.newVar(vname) + 1
            self.vname2dimacslit[vname] = dimacsLit
        return dimacsLit if positive else -dimacsLit

    def toDimacsLits(self, clause):
        """変数名のリストをDIMACS形式の節に変換する．

        変数名が `~' から始まれば負リテラルとする．

        Args:
            clause (list): 変数名のリスト．

        Returns:
            DIMACS形式の節．
        """
        dimacsLits = [ self.toDimacsLit(str(lit)) for lit in clause ]
        return dimacsLits

    def addClause(self, clause):
        """変数名のリストの形式の節を追加する．

        変数名が `~' から始まれば負リテラルとする．

        Args:
            clause (list): 変数名のリスト．
        """
        dimacsLits = clause
        if isinstance(clause, str):
            return
        dimacsLits = self.toDimacsLits(clause)
        self.addDimacsClause(dimacsLits)

    def add(self, *clauses):
        """節のリスト (変数名のリストの形式)を追加する．

        Args:
            clauses (list): 節のリスト．
        """
        for clause in clauses:
            self.addClause(clause)

    def load(self, file=None):
        """ファイルから節を読込んで追加する．

        Args:
            file (str): ファイル名 (Noneなら標準入力)．
        """
        if file is None:
            inp = sys.stdin
        elif isinstance(file, str):
            inp = open(file, encoding="utf-8")
        else:
            inp = file
        for line in inp:
            line = re.sub(r"\s+", " ", line).strip()
            if line == "" or line.startswith("#"):
                continue
            clause = []
            for s in line.split(" "):
                if s.startswith("~"):
                    clause.append(Bool(s[1:], False))
                else:
                    clause.append(Bool(s))
            self.addClause(clause)
        input.close()

import sys
import re
import time
from .util import Bool
