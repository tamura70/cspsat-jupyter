"""制約充足問題(CSP)のSAT符号化のモジュール．

以下のクラスからなる．

* Solverクラス: 制約ソルバーのクラス．
* Encoderクラス: SAT符号化の基底クラス
* DirectEncoderクラス: 直接符号化・支持符号化のクラス
* OrderEncoderクラス: 順序符号化のクラス
* LogEncoderクラス: 対数符号化のクラス

以下の記法の説明でA1, A2, ..., Anは制約．X1, X2, ..., Xnは式，L1, L2, ..., Lnはリテラルまたはドメインが0,1の整数変数を表す．

.. csv-table:: 宣言の記法
   :header: "名称など", "宣言の記法", "備考"
   :widths: 20, 40, 30

   "整数変数", "[""int"", lb, ub]"
   "最小化", "[""minimize"", X]"
   "最大化", "[""maximize"", X]"
   "コメント", "[""comment"", ...], [""#"", ...]"

.. csv-table:: 式の記法
   :header: "名称など", "式の記法", "備考"
   :widths: 20, 40, 30

   "整数定数", n, "0, 1, -1, 2, -2などの整数"
   "整数変数", "Var(:math:`x`)", ":math:`x` は変数名の文字列"
   "リテラル", "p, ~p", "pは :obj:`.Bool` オプジェクト"
   "マイナス", "[""-"", X1]"
   "加算", "[""+"", X1, ..., Xn]"
   "減算", "[""-"", X1, X2]"
   "定数乗算", "[""*"", X1, X2]", "X1またはX2は整数定数"
   "定数除算", "[""div"", X, n], [""//"", X, n]", nは正の整数定数. :obj:`cspsat.hooks.defaultFunctionHook` で定義
   "定数剰余", "[""mod"", X, n], [""%"", X, n]", nは正の整数定数. :obj:`cspsat.hooks.defaultFunctionHook` で定義
   "絶対値", "[""abs"", X]", :obj:`cspsat.hooks.defaultFunctionHook` で定義
   "最小値", "[""min"", X1, ..., Xn]", :obj:`cspsat.hooks.defaultFunctionHook` で定義
   "最大値", "[""max"", X1, ..., Xn]", :obj:`cspsat.hooks.defaultFunctionHook` で定義
   "if式", "[""if"", A, X, Y]", :obj:`cspsat.hooks.defaultFunctionHook` で定義
   "Xのiビット目", "[""bit"", X, i]", iは0以上の整数定数. :obj:`cspsat.hooks.defaultFunctionHook` で定義

.. csv-table:: 制約の記法
   :header: "名称など", "制約の記法", "備考"
   :widths: 20, 40, 30

   "真", "TRUE"
   "偽", "FALSE"
   "命題変数", "Bool(:math:`p`)", ":math:`p` は変数名の文字列"
   "負リテラル", "~p", "pは :obj:`.Bool` オブジェクト"
   "否定", "[""not"", A1], [""!"", A1]"
   "論理積", "[""and"", A1, ..., An], [""&&"", A1, ..., An]"
   "論理和", "[""or"", A1, ..., An], [""||"", A1, ..., An]"
   "含意", "[""imp"", A1, A2], [""=>"", A1, A2]"
   "同値", "[""equ"", A1, A2], [""<=>"", A1, A2]"
   "排他的論理和", "[""xor"", A1, ..., An], [""^"", A1, ..., An]"
   ":math:`X_1 = X_2`", "[""eq"", X1, X2], [""=="", X1, X2]"
   ":math:`X_1 \\ne X_2`", "[""ne"", X1, X2], [""!="", X1, X2]"
   ":math:`X_1 \\ge X_2`", "[""ge"", X1, X2], ["">="", X1, X2]"
   ":math:`X_1 > X_2`", "[""gt"", X1, X2], ["">"", X1, X2]"
   ":math:`X_1 \\le X_2`", "[""le"", X1, X2], [""<="", X1, X2]"
   ":math:`X_1 < X_2`", "[""lt"", X1, X2], [""<"", X1, X2]"
   ":math:`\\sum L_i = k` (exact-k制約)", "[""eqK"", [L1,...,Ln], k]", "kは整数定数"
   ":math:`\\sum L_i \\ne k`", "[""neK"", [L1,...,Ln], k]", "kは整数定数"
   ":math:`\\sum L_i \\ge k` (at-least-k制約)", "[""geK"", [L1,...,Ln], k]", "kは整数定数"
   ":math:`\\sum L_i > k`", "[""gtK"", [L1,...,Ln], k]", "kは整数定数"
   ":math:`\\sum L_i \\le k` (at-most-k制約)", "[""leK"", [L1,...,Ln], k]", "kは整数定数"
   ":math:`\\sum L_i < k`", "[""ltK"", [L1,...,Ln], k]", "kは整数定数"

.. csv-table:: グローバル制約の記法
   :header: "名称など", "グローバル制約の記法", "備考"
   :widths: 20, 40, 30

   "alldifferent制約", "[""alldifferent"", X1, ..., Xn]", :obj:`cspsat.hooks.defaultConstraintHook` で定義
   "辞書順比較", "[""lexCmp"", cmp, [X1,...,Xn], [Y1,...,Yn]]", :obj:`cspsat.hooks.defaultConstraintHook` で定義
   "乗算比較", "[""mulCmp"", cmp, X, Y, Z]", :obj:`cspsat.hooks.defaultConstraintHook` で定義
   "ベキ乗比較", "[""powCmp"", cmp, X, n, Y]", :obj:`cspsat.hooks.defaultConstraintHook` で定義
   "ビット列", "[""bits"", [X1,...,Xn], X]", :obj:`cspsat.hooks.defaultConstraintHook` で定義
   "ビット", "[""bit"", X, i]", :obj:`cspsat.hooks.defaultConstraintHook` で定義

Note:
    本プログラムは学習用の目的で作成されている．
    実用上の問題への適用は想定していない．
    Copyright (c) 2025-- Naoyuki Tamura
    Licensed under the MIT License
"""

import itertools
import time
import math
from .hooks import defaultFunctionHook, defaultConstraintHook

class Solver():
    """制約ソルバーのクラス．

    Args:
        encoder (Encoder): 使用する :obj:`Encoder` オブジェクト．
        sat (SAT, optional): 使用する :obj:`.SAT` ソルバーオブジェクト．指定しなければcommandで指定したSATソルバーを用いる．
        command (str, optional): 利用するSATソルバーのコマンド．指定しなければ ``./sat4j`` が利用される(Windowsなら ``.\\sat4j.bat``)．
            :obj:`.SAT` のcommand参照．
        positiveOnly (bool, optional): Trueならモデルに正リテラルのみを含める (デフォールトはFalse)．
            :obj:`.SAT` のpositiveOnly参照．
        includeAux (bool, optional): Trueならモデルに補助変数を含める (デフォールトはFalse)．
            :obj:`.SAT` のincludeAux参照．
        verbose (int, optional): 正なら実行の詳細情報を表示する (デフォールトは0)．

    Attributes:
        model (dict): 最後に得られたモデル． 整数変数(:obj:`.Var`)あるいは命題変数(:obj:`.Bool`)をキーとし，変数の値が保持された辞書(dict)．
        stats (dict): ソルバー実行の統計データ．
        minimize (Wsum): 最小化する線形和．Noneなら最小化しない．
        maximize (Wsum): 最大化する線形和．Noneなら最大化しない．minimizeが設定されていればそちらを優先する．
    """

    def __init__(self, encoder=None, sat=None, command=None, positiveOnly=False, includeAux=False, verbose=0):
        self.startTime = time.time()
        self.encoder = encoder or OrderEncoder(verbose=verbose)
        self.encoder.solver = self
        self.sat = sat or SAT(command=command)
        self.positiveOnly = positiveOnly
        self.includeAux = includeAux
        self.verbose = self.sat.verbose = verbose

        Var._auxCount = 0
        self.stats = {
            "result":None, "ncalls":0, "nmodels":0, "time":0, "encoding":0, "solving":0, "intVars":0, "constraints":0,
            "encoder":self.encoder.__class__.__name__, "command":self.sat.command, "sat":[] }
        self.model = None
        self.minimize = None
        self.maximize = None

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        if self.sat:
            del self.sat

    def add(self, *constraints):
        """制約をSAT符号化したCNF式をSATソルバーに追加する．
        :obj:`Encoder.put`, :obj:`Encoder.encode`, :obj:`.SAT.add` を呼び出す．

        ただし制約が ``["minimize", w]`` あるいは ``["maximize", w]`` の場合は最小化あるいは最大化する線形和を設定する．

        Args:
            constraints (list): 制約のリスト．
        """
        t = time.time()
        for constraint in constraints:
            match constraint:
                case ["comment", *_] | ["#", *_]:
                    pass
                case ["minimize", _] | ["maximize", _]:
                    pass
                case ["int", _, _, _]:
                    self.stats["intVars"] += 1
                case _:
                    self.stats["constraints"] += 1
            self.encoder.put(constraint)
            self.sat.add(*self.encoder.encode())
        t = time.time() - t
        self.stats["encoding"] = self.stats.get("encoding", 0) + t

    def find(self):
        """SATソルバーに追加されている制約充足問題の解をSATソルバーで求めて制約充足問題の解に変換して返す．

        Returns:
            見つかった制約充足問題のモデル．モデルが見つからなければNone．
        """
        self.sat.find()
        for k in ["result", "ncalls", "nmodels", "solving", "sat"]:
            self.stats[k] = self.sat.stats[k]
        if self.sat.model is None:
            self.model = model = None
        else:
            self.model = model = self.encoder.decode(self.sat.model, includeAux=self.includeAux)
            if self.positiveOnly:
                model = dict(filter(lambda kv: not (isinstance(kv[0], Bool) and kv[1] == 0), model.items()))
        self.stats["time"] = time.time() - self.startTime
        return model

    def _solutionsSat(self, num=1):
        count = 0
        while num == 0 or count < num:
            model = self.find()
            if model is None:
                break
            yield model
            self.sat.add(self.encoder.getBlock(self.model, self.sat.model))
            count += 1

    def _solutionsOpt(self, wsum, minimize, num=1, includeSat=False):
        self.sat.commit()
        model = self.find()
        if model is None:
            return
        (lb, ub) = self.encoder.wsumBound(wsum)
        if minimize:
            ub = wsum.value(model)
        else:
            lb = wsum.value(model)
        while True:
            self.sat.cancel()
            if lb == ub:
                break
            if includeSat: # experimental
                self.stats["result"] = f"BOUND {lb} {ub}"
                yield model
            if minimize:
                mid = math.floor((lb+ub)/2)
                # v <= mid
                self.sat.add(*self.encoder.encodeWsumLe0(wsum.sub(mid)))
                model1 = self.find()
                if model1 is not None:
                    model = model1
                    ub = wsum.value(model)
                else:
                    lb = mid + 1
            else:
                mid = math.ceil((lb+ub)/2)
                # v >= mid
                self.sat.add(*self.encoder.encodeWsumLe0(wsum.sub(mid).neg()))
                model1 = self.find()
                if model1 is not None:
                    model = model1
                    lb = wsum.value(model)
                else:
                    ub = mid - 1
        if minimize:
            # v <= ub
            self.sat.add(*self.encoder.encodeWsumLe0(wsum.sub(ub)))
        else:
            # v >= lb
            self.sat.add(*self.encoder.encodeWsumLe0(wsum.sub(lb).neg()))
        count = 0
        while True:
            self.stats["result"] = f"MINIMUM {lb}" if minimize else f"MAXIMUM {ub}"
            yield model
            count += 1
            if num and count >= num:
                break
            self.sat.add(self.encoder.getBlock(model, self.sat.model))
            model = self.find()
            if model is None:
                break

    def getStat(self, includeAll=False):
        """このソルバーによる実行の統計データを返す．

        なお，実行時間などの単位は秒で，CPU時間ではなく経過時間である．

        Args:
            includeAll (bool, optional): Falseならsatフォールドの値は最後のSATソルバー実行の統計データのみである．
                TrueならすべてのSATソルバー実行の統計データのリストになる (デフォールトはFalse)．

        Returns:
            以下からなる辞書(dict)を返す．

            * result: 実行結果 (SATISFIABLE, UNSATISFIABLE, MINIMUM n, MAXIMUM n, TIMEOUT, UNKNOWN, None)．
            * ncalls: SATソルバーを呼び出した回数．
            * nmodels: SATソルバーでモデルを見つけた回数．
            * time: 関数の実行時間 (秒)．
            * encoding: SAT符号化にかかった時間 (累積値，秒)．
            * solving: SATソルバーの実行にかかった時間 (累積値，秒)．
            * command: SATソルバーのコマンド．
            * sat: SATソルバーの統計データ (includeAllがFalseなら最後の統計データ，Trueならすべての統計データのリスト)．

            SATソルバーの統計データ(satフィールド)は，以下からなるdictである．

            * result: SATソルバーの実行結果 (SATISFIABLE, UNSATISFIABLE, TIMEOUT, UNKNOWN, None)．
            * variables: CNF式の命題変数の個数．
            * clauses: CNF式の節の個数．
            * conflicts: SATソルバー実行での衝突の回数．
            * decisions: SATソルバー実行での決定の回数．
            * propagations: SATソルバー実行での伝播の回数．
            * solving: SATソルバー呼び出しの実行時間 (秒)．
        """
        if includeAll:
            return self.stats
        infos = self.stats["sat"]
        info = infos[-1] if len(infos) > 0 else {}
        return { **self.stats, "sat":info }

    def solutions(self, csp, num=1):
        """与えられた制約充足問題の複数のモデルを探索し，それらをyieldするジェネレータ関数．

        指定された上限の個数までモデルを探索し，モデルが見つかればそれをyieldする．
        最小化あるいは最大化する線形和が指定されている場合は，制約最適問題として最適解を二分探索する．
        ただし，最小化問題の場合，線形和が最適値未満なら常に充足不能，最適値以上なら常に充足可能と仮定している．
        同様に，最大化問題の場合，線形和が最適値より大きいなら常に充足不能，最適値以下なら常に充足可能と仮定している．
        なお，複数のモデルを探索において，補助変数の値のみが異なっている場合は同じモデルとみなされる．

        Args:
            csp (list): 制約充足問題 (制約のリスト)．
            num (int, optional): 探索するモデルの最大個数．0なら全解を探索する (デフォールトは1)．

        Yields:
            見つかったモデル．
        """
        self.add(*csp)
        if self.minimize:
            yield from self._solutionsOpt(self.minimize, minimize=True, num=num)
        elif self.maximize:
            yield from self._solutionsOpt(self.maximize, minimize=False, num=num)
        else:
            yield from self._solutionsSat(num=num)

    def solve(self, csp, num=1, stat=False):
        """与えられた制約充足問題の複数のモデルを探索し，表示する関数．

        Args:
            csp (list): 制約充足問題 (制約のリスト)．
            num (int, optional): 探索するモデルの最大個数．0なら全解を探索する (デフォールトは1)．
            stat (bool, optional): Trueなら統計データも表示する (デフォールトはFalse)．
        """
        count = 0
        for model in self.solutions(csp, num=num):
            if count == 0:
                print(self.stats["result"], flush=True)
            count += 1
            if self.verbose >= 0:
                print(f"Model {count}: {model}", flush=True)
            if stat:
                print(f"Stat: {self.getStat()}", flush=True)
        if count == 0:
            print(self.stats["result"], flush=True)
        if stat and (count == 0 or num == 0):
            print(f"Stat: {self.getStat()}", flush=True)

class Encoder():
    """制約充足問題(CSP)をSAT符号化する抽象基底クラス．

    Args:
        functionHooks (optional): 式の符号化時に呼び出すフック関数のリスト．指定しなければ :obj:`defaultFunctionHooks` を使用する．
        constraintHooks (optional): 制約の符号化時に呼び出すフック関数のリスト．指定しなければ :obj:`defaultConstraintHooks` を使用する．
        verbose (int, optional): 正なら実行の詳細情報を表示する (デフォールトは0)．

    Atrributes:
        solver (Solver): このオブジェクトを利用する制約ソルバーのクラス．

    """

    defaultFunctionHooks = [defaultFunctionHook]
    """式の符号化時に呼び出されるフック関数のリスト．デフォールト値は :obj:`.defaultFunctionHook` のみのリスト．

    Examples:
        >>> Encoder.defaultFunctionHooks
        [<function defaultFunctionHook at 0x722f0efa0540>]
        >>> def myFunctionHook(f, encoder):
        ...   match f:
        ...     case ["++", x]:
        ...       f = ["+", x, 1]
        ...   return f
        ... 
        >>> Encoder.defaultFunctionHooks.append(myFunctionHook)
    """

    defaultConstraintHooks = [defaultConstraintHook]
    """制約の符号化時に呼び出されるフック関数のリスト．デフォールト値は :obj:`.defaultConstraintHook` のみのリスト．

    Examples:
        >>> Encoder.defaultConstraintHooks
        >>> def myConstraintHook(c, encoder):
        ...   match c:
        ...     case ["even", x]:
        ...       (lb, ub) = encoder.getBound(x)
        ...       y = Var()
        ...       encoder.put(["int", y, int(lb/2), int(ub/2)])
        ...       c = ["==", x, ["+", y, y]]
        ...   return c
        ... 
        >>> Encoder.defaultConstraintHooks.append(myConstraintHook)
    """

    delimEq = "=="
    """v=k を表す命題変数の区切り文字．
    """

    delimGe = ">="
    """v>=k を表す命題変数の区切り文字．
    """

    delimBit = "@"
    """v+lbのkビット目を表す命題変数の区切り文字．
    """

    def __init__(self, functionHooks=None, constraintHooks=None, verbose=0):
        self.functionHooks = functionHooks or Encoder.defaultFunctionHooks
        self.constraintHooks = constraintHooks or Encoder.defaultConstraintHooks
        self.verbose = verbose

        self.solver = None
        self.remaining = []
        self.pre = Preprocessor(self)

        self.lb = {}
        self.ub = {}
        self.name2var = {}

        if self.__class__ == Encoder:
            raise CspsatException("Encoderはインスタンス化できない")

    def defInt(self, v, lb, ub):
        """整数変数をCSPに追加する．節は生成しない．

        節は内部的な制約 ["_int", v, lb, ub] で別に生成する．

        Args:
            v (Var): 整数変数．
            lb (int): 整数変数の下限値．
            ub (int): 整数変数の上限値．

        Raises:
            CspsatException: vがVarオブジェクトでない，あるいは下限値が上限値より大きい，あるいはすでにvが宣言されている．
        """
        if not isinstance(v, Var):
            raise CspsatException(f"{v}はVarオブジェクトでない")
        if int(lb) > int(ub):
            raise CspsatException(f"整数変数{v}のドメインエラー: {lb}..{ub}")
        if v in self.lb:
            raise CspsatException(f"重複した宣言: {v}")
        self.lb[v] = int(lb)
        self.ub[v] = int(ub)
        self.name2var[str(v)] = v

    def variables(self):
        """これまで追加された変数のリストを返す．

        Returns:
            整数変数(:obj:`.Var`)のリストを返す．
        """
        return self.lb.keys()

    def intLb(self, v):
        """変数vの下限値を返す．

        Args:
            v (Bool | Var): 変数．

        Returns:
            vがVarオブジェクトなら指定されている下限値．Boolオブジェクトなら0．
        """
        if isinstance(v, Bool):
            return 0
        if isinstance(v, Var):
            lb = self.lb.get(v)
            if lb is None:
                raise CspsatException(f"整数変数{v}はintで宣言されていない")
            return lb
        raise CspsatException(f"{v}は整数変数(Var)でない")

    def intUb(self, v):
        """変数vの上限値を返す．

        Args:
            v (Bool | Var): 変数．

        Returns:
            vがVarオブジェクトなら指定されている上限値．Boolオブジェクトなら1．
        """
        if isinstance(v, Bool):
            return 1
        if isinstance(v, Var):
            ub = self.ub.get(v)
            if ub is None:
                raise CspsatException(f"整数変数{v}はintで宣言されていない")
            return ub
        raise CspsatException(f"{v}は整数変数(Var)でない")

    def intRange(self, v):
        """変数vの下限値から上限値の値のrangeオブジェクトを返す．

        Args:
            v (Bool | Var): 変数．

        Returns:
            vの下限値から上限値の値のrangeオブジェクト．
        """
        return range(self.intLb(v), self.intUb(v)+1)

    def wsumBound(self, w):
        """線形和wの下限値と上限値の対を返す．

        Args:
            w (Wsum): 線形和．

        Returns:
            wの下限値lbと上限値ubの対 (lb,ub)．
        """
        lb = w.c
        ub = w.c
        for v in w.variables():
            a = w.coef(v)
            if a < 0:
                lb += a * self.intUb(v)
                ub += a * self.intLb(v)
            else:
                lb += a * self.intLb(v)
                ub += a * self.intUb(v)
        return (lb, ub)

    def getBound(self, s):
        """制約充足問題の式sの下限値と上限値を返す．

        Args:
            s (list): 制約充足問題の式．

        Returns:
            sの下限値lbと上限値ubの対 (lb,ub)．

        Raises:
            CspsatException: 式の構文が間違っている．
        """
        return self.wsumBound(self.pre.toWsum(s))

    def varEqK(self, v, k, a=1):
        """a*v==kを表す命題変数を返す．

        :obj:`DirectEncoder`, :obj:`OrderEncoder` で使用．

        Args:
            v (Bool | Var): 変数．
            k (int): 定数．
            a (int, optional): 係数．

        Returns:
            vがBoolオブジェクトの時，k=0なら~v，k=1ならv，その他ならFALSEを返す．vがVarオブジェクトの時，kがvのドメインに含まれればv==kを表す命題変数，その他ならFALSEを返す．
        """
        if k % a != 0:
            return FALSE
        k = k // a
        # v == k
        match (v,k):
            case (Bool(), 0):
                return ~v
            case (Bool(), 1):
                return v
            case (Bool(), _):
                return FALSE
            case (Var(), _) if k < self.intLb(v) or self.intUb(v) < k:
                return FALSE
            case (Var(), _):
                return Bool(f"{v}{Encoder.delimEq}{k}", internal=True)
            case _:
                raise CspsatException(f"varEqKの引数エラー: varEqK({v},{k})")

    def varToBool(self, v):
        """変数vが0から1の値だけを取るとき，vに対する命題変数を返す．

        Encoderの派生クラスで実装する必要がある．

        Args:
            v (Bool | Var): 変数．

        Returns:
            vがBoolオブジェクトならvを返す．vがVarオブジェクトの場合は以下の通り．

            * DirectEncodingの場合 v=1 に対応する命題変数．
            * OrderEncodingの場合 v>=1 に対応する命題変数．
            * LogEncodingの場合 v の最下位ビットに対応する命題変数．
        """
        raise CspsatException("実装されていない")

    def encodeInt(self, v):
        """整数変数をSAT符号化したCNF式をyieldするジェネレータ関数．

        Encoderの派生クラスで実装する必要がある．

        Args:
            v (Var): 整数変数．

        Yields:
            CNF式．
        """
        raise CspsatException("実装されていない")

    def isBoolLike(self, v):
        """与えられた整数変数が0-1変数なら真を返す関数．

        Args:
            v (Var): 整数変数．

        Returns:
            vが0-1変数なら真．それ以外は偽．

        """
        if isinstance(v, Bool):
            return True
        return isinstance(v, Var) and self.intLb(v) == 0 and self.intUb(v) == 1

    def encodeWsumEq0(self, w):
        """線形和の式 w==0 をSAT符号化したCNF式をyieldするジェネレータ関数．

        Encoderの派生クラスで実装する必要がある．

        Args:
            w (Wsum): 線形和．

        Yields:
            CNF式．
        """
        raise CspsatException("実装されていない")

    def _pickV(self, w):
        (v1, s1, a1) = (None, None, None)
        for v in sorted(list(w.variables())):
            s = self.intUb(v) - self.intLb(v) + 1
            a = abs(w.coef(v))
            if v1 is None or s < s1 or (s == s1 and a < a1):
                (v1, s1, a1) = (v, s, a)
        return v1

    def encodeWsumNe0(self, w):
        """線形和の式 w!=0 をSAT符号化したCNF式をyieldするジェネレータ関数．

        Encoderの派生クラスで実装する必要がある．

        Args:
            w (Wsum): 線形和．

        Yields:
            CNF式．
        """
        raise CspsatException("実装されていない")

    def encodeWsumLe0(self, w):
        """線形和の式 w<=0 をSAT符号化したCNF式をyieldするジェネレータ関数．

        Encoderの派生クラスで実装する必要がある．

        Args:
            w (Wsum): 線形和．

        Yields:
            CNF式．
        """
        raise CspsatException("実装されていない")

    def _encodeXclause(self, xclause):
        (lits, cc) = ([], [])
        for lit in xclause:
            if isinstance(lit, Bool):
                lits.append(lit)
            else:
                cc.append(lit)
        if len(cc) >= 2:
            raise CspsatException(f"XCNFの構文エラー: {xclause}")
        if len(cc) == 0:
            yield xclause
            return
        constraint = cc[0]
        match constraint:
            case ["_eq0", w]:
                for c in self.encodeWsumEq0(self.pre.toWsum(w)):
                    yield [ *lits, *c ]
            case ["_ne0", w]:
                for c in self.encodeWsumNe0(self.pre.toWsum(w)):
                    yield [ *lits, *c ]
            case ["_le0", w]:
                for c in self.encodeWsumLe0(self.pre.toWsum(w)):
                    yield [ *lits, *c ]
            case ["eqK", xx, k]:
                for c in SeqCounter.eqK(xx, k):
                    yield [ *lits, *c ]
            case ["neK", xx, k]:
                for c in SeqCounter.neK(xx, k):
                    yield [ *lits, *c ]
            case ["leK", xx, k]:
                for c in SeqCounter.leK(xx, k):
                    yield [ *lits, *c ]
            case _:
                raise CspsatException(f"XCNF制約の構文エラー: {constraint}")

    def encode(self):
        """追加された制約をSAT符号化した節をyieldするジェネレータ関数．

        ただし制約が ``["minimize", x]`` あるいは ``["maximize", x]`` の場合は最小化あるいは最大化する線形和を設定する．

        Yields:
            SAT符号化した節．
        """
        while self.remaining:
            constraint = self.remaining.pop(0)
            yield f"# {constraint}"
            match constraint:
                case ["comment", *_] | ["#", *_]:
                    pass
                case ["minimize", x]:
                    if self.solver.minimize or self.solver.maximize:
                        raise CspsatException(f"minimizeあるいはmaxmizeの二重宣言: {constraint}")
                    self.solver.minimize = self.pre.toWsum(x)
                case ["maximize", x]:
                    if self.solver.minimize or self.solver.maximize:
                        raise CspsatException(f"minimizeあるいはmaxmizeの二重宣言: {constraint}")
                    self.solver.maximize = self.pre.toWsum(x)
                case ["_int", v, _, _]: # internal use
                    for clause in self.encodeInt(v):
                        if TRUE not in clause:
                            yield [ lit for lit in clause if lit != FALSE ]
                case _:
                    for xclause in self.pre.toXCNF(constraint):
                        if self.verbose >= 1:
                            yield f"# xclause: {xclause}"
                        for clause in self._encodeXclause(xclause):
                            if TRUE not in clause:
                                yield [ lit for lit in clause if lit != FALSE ]

    def put(self, *constraints):
        """制約を追加する．SAT符号化は行わない．

        Args:
            constraints (list): 制約のリスト．
        """
        for constraint in constraints:
            match constraint:
                case ["int", v, lb, ub]:
                    self.defInt(v, lb, ub)
                    self.put(["_int", v, lb, ub]) # internal use
                case _:
                    self.remaining.append(constraint)

    def decode(self, satModel, includeAux=False):
        """SATのモデルを制約充足問題のモデルに変換する．

        Args:
            satModel (dict): SATのモデル．

        Returns:
            制約充足問題のモデル．
        """
        raise CspsatException("実装されていない")

    def toCNF(self, csp):
        """制約充足問題をSAT符号化したCNF式をyieldするジェネレータ関数．

        Args:
            csp (list): 制約充足問題 (制約のリスト)．

        Yields:
            CNF式．
        """
        for constraint in csp:
            self.put(constraint)
            yield from self.encode()

    def getBlock(self, model, satModel):
        """制約充足問題のモデルを否定する条件をCNF式として返す関数．

        Args:
            model (dict): 制約充足問題のモデル．
            satModel (dict): SATソルバーのモデル．

        Returns:
            モデルを否定したCNF式．
        """
        raise CspsatException("実装されていない")

class DirectEncoder(Encoder):
    """直接符号化・支持符号化を実装したクラス．
    """

    def __init__(self, functionHooks=None, constraintHooks=None, verbose=0):
        super().__init__(functionHooks, constraintHooks, verbose)
        self.pre = Preprocessor(self, introduceAux=True)
        self.useSC = False

    def varToBool(self, v):
        """:obj:`Encoder.varToBool` の :obj:`DirectEncoder` での実装
        """
        assert self.intLb(v) == 0 and self.intUb(v) == 1
        return self.varEqK(v, 1)

    def encodeInt(self, v):
        """DirectEncoderでの実装
        """
        ps = [ self.varEqK(v, d) for d in self.intRange(v) ]
        if self.useSC:
            # Sequential counter
            yield from SeqCounter.eqK(ps, 1)
        else:
            # Pair wise
            yield ps
            for qs in itertools.combinations(ps, 2):
                yield [ ~q for q in qs ]

    def encodeWsumEq0(self, w):
        (lb, ub) = self.wsumBound(w)
        if ub < 0 or 0 < lb:
            yield [] # False clause
            return
        match len(w.variables()):
            case 0 if w.c != 0:
                yield [] # False clause
            case 0:
                yield from [] # True
            case 1:
                v = self._pickV(w)
                yield [ self.varEqK(v, -w.c, a=w.coef(v)) ]
            case _:
                v = self._pickV(w)
                for d in self.intRange(v):
                    for clause in self.encodeWsumEq0(w.where(v, d)):
                        yield [ ~self.varEqK(v,d), *clause ]

    def encodeWsumNe0(self, w):
        (lb, ub) = self.wsumBound(w)
        if ub < 0 or 0 < lb:
            yield from [] # True
            return
        match len(w.variables()):
            case 0 if w.c == 0:
                yield [] # False clause
            case 0:
                yield from [] # True
            case 1:
                v = self._pickV(w)
                yield [ ~self.varEqK(v, -w.c, a=w.coef(v)) ]
            case _:
                v = self._pickV(w)
                for d in self.intRange(v):
                    for clause in self.encodeWsumNe0(w.where(v, d)):
                        yield [ ~self.varEqK(v,d), *clause ]

    def encodeWsumLe0(self, w):
        (lb, ub) = self.wsumBound(w)
        if lb > 0:
            yield [] # False clause
            return
        if ub <= 0:
            yield from [] # True
            return
        match len(w.variables()):
            case 0 if w.c > 0:
                yield [] # False clause
            case 0:
                yield from [] # True
            case 1:
                v = self._pickV(w)
                a = w.coef(v)
                if a > 0:
                    # v <= -w.c/a
                    b = math.floor(-w.c/a)
                    yield [ self.varEqK(v, k) for k in range(self.intLb(v),b+1) ]
                else:
                    # v >= -w.c/a
                    b = math.ceil(-w.c/a)
                    yield [ self.varEqK(v, k) for k in range(b,self.intUb(v)+1) ]
            case _:
                v = self._pickV(w)
                for d in self.intRange(v):
                    for clause in self.encodeWsumLe0(w.where(v, d)):
                        yield [ ~self.varEqK(v,d), *clause ]

    def decode(self, satModel, includeAux=False):
        if satModel is None:
            return None
        model = {}
        for x in satModel:
            s = str(x)
            if Encoder.delimEq in s:
                if satModel[x] == 1:
                    (v,d) = s.split(Encoder.delimEq, 2)
                    v = self.name2var[v]
                    if not v.isAux() or includeAux:
                        model[v] = int(d)
            elif not x.isAux() or includeAux:
                model[x] = satModel[x]
        return model

    def getBlock(self, model, _):
        block = []
        for v in model:
            block.append(~self.varEqK(v, model[v]))
        return block

class OrderEncoder(Encoder):
    """順序符号化を実装したクラス．
    """

    def __init__(self, functionHooks=None, constraintHooks=None, verbose=0):
        super().__init__(functionHooks, constraintHooks, verbose)
        self.pre = Preprocessor(self, introduceAux=True)

    def varGeK(self, v, k):
        """v>=kを表す命題変数を返す．

        Args:
            v (Bool | Var): 変数．
            k (int): 定数．

        Returns:
            vがBoolオブジェクトの時，k<=0ならTRUE，k=1ならv，その他ならFALSEを返す．vがVarオブジェクトの時，kがvの下限値以下ならTRUE，上限値より大きいならFALSE，その他ならv>=kを表す命題変数を返す．
        """
        # v >= k
        match (v,k):
            case (Bool(), _) if k <= 0:
                return TRUE
            case (Bool(), 1):
                return v
            case (Bool(), _) if k > 1:
                return FALSE
            case (Var(), _) if k <= self.intLb(v):
                return TRUE
            case (Var(), _) if k > self.intUb(v):
                return FALSE
            case (Var(), _):
                return Bool(f"{v}{Encoder.delimGe}{k}", internal=True)
            case _:
                raise CspsatException(f"varGeKの引数エラー: varGeK({v},{k})")

    def varToBool(self, v):
        """:obj:`Encoder.varToBool` の :obj:`OrderEncoder` での実装
        """
        assert self.intLb(v) == 0 and self.intUb(v) == 1
        return self.varGeK(v, 1)

    def encodeInt(self, v):
        """OrderEncoderでの実装
        """
        for d in range(self.intLb(v)+1,self.intUb(v)):
            yield [ self.varGeK(v,d), ~self.varGeK(v,d+1) ]
        for d in self.intRange(v):
            yield [ ~self.varEqK(v,d), self.varGeK(v,d) ]
            yield [ ~self.varEqK(v,d), ~self.varGeK(v,d+1) ]
            yield [ self.varEqK(v,d), ~self.varGeK(v,d), self.varGeK(v,d+1) ]

    def encodeWsumEq0(self, w):
        (lb, ub) = self.wsumBound(w)
        if ub < 0 or 0 < lb:
            yield [] # False clause
            return
        match len(w.variables()):
            case 0 if w.c != 0:
                yield [] # False clause
            case 0:
                yield from [] # True
            case 1:
                v = self._pickV(w)
                yield [ self.varEqK(v, -w.c, a=w.coef(v)) ]
            case _:
                yield from self.encodeWsumLe0(w)
                yield from self.encodeWsumLe0(w.neg())

    def encodeWsumNe0(self, w):
        (lb, ub) = self.wsumBound(w)
        if ub < 0 or 0 < lb:
            yield from [] # True
            return
        match len(w.variables()):
            case 0 if w.c == 0:
                yield [] # False clause
            case 0:
                yield from [] # True
            case 1:
                v = self._pickV(w)
                yield [ ~self.varEqK(v, -w.c, a=w.coef(v)) ]
            case _:
                p = Bool()
                expr = w.toExpr()
                self.put(["or", ~p(1), ["<", expr, 0]])
                self.put(["or", ~p(2), [">", expr, 0]])
                yield [ p(1), p(2) ]

    def encodeWsumLe0(self, w):
        def _isSatLe0(w):
            (lb, ub) = self.wsumBound(w)
            return lb <= 0 < ub
        def _isUnsatLe0(w):
            (lb, _) = self.wsumBound(w)
            return lb > 0

        (lb, ub) = self.wsumBound(w)
        if lb > 0:
            yield [] # False clause
            return
        if ub <= 0:
            yield from [] # True
            return
        match len(w.variables()):
            case 0 if w.c > 0:
                yield [] # False clause
            case 0:
                yield from [] # True
            case 1:
                v = self._pickV(w)
                a = w.coef(v)
                if a > 0:
                    k = math.floor(-w.c/a)
                    yield [ ~self.varGeK(v, k+1) ]
                else:
                    k = math.ceil(-w.c/a)
                    yield [ self.varGeK(v, k) ]
            case _:
                v = self._pickV(w)
                ds1 = [ d for d in self.intRange(v) if _isSatLe0(w.where(v, d)) ]
                ds2 = [ d for d in self.intRange(v) if _isUnsatLe0(w.where(v, d)) ]
                if w.coef(v) > 0:
                    for d in ds1:
                        for clause in self.encodeWsumLe0(w.where(v, d)):
                            yield [ ~self.varGeK(v, d), *clause ]
                    if ds2:
                        yield [ ~self.varGeK(v, min(ds2)) ]
                else:
                    for d in ds1:
                        for clause in self.encodeWsumLe0(w.where(v, d)):
                            yield [ self.varGeK(v, d+1), *clause ]
                    if ds2:
                        yield [ self.varGeK(v, max(ds2)+1) ]

    def decode(self, satModel, includeAux=False):
        if satModel is None:
            return None
        model = {}
        for v in self.variables():
            if not v.isAux() or includeAux:
                model[v] = self.intLb(v)
        for x in satModel:
            s = str(x)
            if Encoder.delimGe in s:
                if satModel[x] == 1:
                    (v,d) = s.split(Encoder.delimGe, 2)
                    v = self.name2var[v]
                    if not v.isAux() or includeAux:
                        model[v] = max(model[v], int(d))
            elif Encoder.delimEq in s:
                pass
            elif not x.isAux() or includeAux:
                model[x] = satModel[x]
        return model

    def getBlock(self, model, _):
        block = []
        for v in model:
            block.append(~self.varGeK(v, model[v]))
            block.append(self.varGeK(v, model[v]+1))
        return block

class LogEncoder(Encoder):
    """対数符号化を実装したクラス．
    """

    def varEqK(self, v, k, a=1):
        raise CspsatException("実装されていない")

    def _nbits(self, v):
        return (self.intUb(v) - self.intLb(v)).bit_length()

    def varBitK(self, v, k):
        """与えられた整数変数vのkビット目を返す関数．

        Args:
            v (Var): 整数変数．
            k (int): ビット位置．

        Returns:
            vのkビット目を表す命題変数 (:obj:`.Bool`)．
        """
        if 0 <= k < self._nbits(v):
            return Bool(f"{v}{Encoder.delimBit}{k}", internal=True)
        return FALSE

    def getBools(self, v):
        """与えられた整数変数を表現するビット列を返す関数．

        Args:
            v (Var): 整数変数．

        Returns:
            vを表現するビット列．
        """
        return [ self.varBitK(v, k) for k in range(self._nbits(v)) ]

    def varToBool(self, v):
        """:obj:`Encoder.varToBool` の :obj:`LogEncoder` での実装
        """
        assert self.intLb(v) == 0 and self.intUb(v) == 1
        return self.varBitK(v, 0)

    def encodeInt(self, v):
        (lb, ub) = (self.intLb(v), self.intUb(v))
        yield from Binary.leK(self.getBools(v), ub - lb)

    def encodeWsumEq0(self, w):
        binEqu = BinaryEquation(self)
        binEqu.addNum(w.c)
        for (v, a) in w.wsum.items():
            binEqu.add(v, a=a)
        yield from binEqu.cmp0("==")

    def encodeWsumNe0(self, w):
        binEqu = BinaryEquation(self)
        binEqu.addNum(w.c)
        for (v, a) in w.wsum.items():
            binEqu.add(v, a=a)
        yield from binEqu.cmp0("!=")

    def encodeWsumLe0(self, w):
        binEqu = BinaryEquation(self)
        binEqu.addNum(w.c)
        for (v, a) in w.wsum.items():
            binEqu.add(v, a=a)
        yield from binEqu.cmp0("<=")

    def decode(self, satModel, includeAux=False):
        if satModel is None:
            return None
        model = {}
        for v in self.variables():
            if not v.isAux() or includeAux:
                model[v] = self.intLb(v)
        for x in satModel:
            s = str(x)
            if Encoder.delimBit in s:
                if satModel.get(x) == 1:
                    (v,d) = s.split(Encoder.delimBit, 2)
                    v = self.name2var[v]
                    if not v.isAux() or includeAux:
                        model[v] += 1 << int(d)
            elif not x.isAux() or includeAux:
                model[x] = satModel[x]
        return model

    def getBlock(self, model, satModel):
        block = []
        for v in model:
            if isinstance(v, Bool):
                block.append(~v if model[v] else v)
            else:
                for x in self.getBools(v):
                    block.append(~x if satModel.get(x) else x)
        return block

from .sat import SAT
from .util import CspsatException, Bool, TRUE, FALSE, Var, Binary, BinaryEquation, SeqCounter, Preprocessor
