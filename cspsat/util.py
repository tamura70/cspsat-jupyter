"""cspsat のユーティリティー．

以下の定義からなる．

* CspsatException例外: 例外を表すクラス．
* CspsatTimeout例外: タイムアイト例外を表すクラス．
* Boolクラス: 命題変数およびリテラルを表すクラス．
* TRUEオブジェクト: 真を表す命題定数．
* FALSEオブジェクト: 偽を表す命題定数．
* Varクラス: 整数変数を表すクラス．
* Wsumクラス: 線形和を表すクラス．
* Binaryクラス: ビット列演算を符号化するためのクラス．クラスメソッドを含む．
* SeqCounterクラス: 基数制約を符号化するためのクラス．クラスメソッドからなる．
* Preprocessorクラス: 制約のSAT符号化の前処理のためのクラス．

Note:
    本プログラムは学習用の目的で作成されている．
    実用上の問題への適用は想定していない．
"""

class CspsatException(Exception):
    """cspsatモジュールの例外を表すクラス．
    """

class CspsatTimeout(CspsatException):
    """cspsatモジュールのタイムアウト例外を表すクラス．
    """

class Bool():
    """命題変数およびリテラルを表すクラス．

    同じ命題変数名と同じ符号を持つBoolオブジェクトは同一とみなされる．

    Args:
        name (str, optional): 命題変数名の文字列．命題変数名を指定しなければ新たな名前を付けた補助変数を生成する．
            変数名が "TRUE" あるいは "FALSE" は命題定数とみなされる．
            変数名が "?" から始まる場合，補助変数とみなされる．
            変数名に "#", "~", "=", "@" を含んではならない (内部的に利用しているため)．
        positive (bool, optional): リテラルの符号．Trueなら正リテラル，Falseなら負リテラルである．指定しなければTrue．
        internal (bool, optional): 内部的に作成することを表すフラグ．指定しなければFalse．

    Examples:
        >>> p = Bool("p") # 命題変数名が"p"のBoolオブジェクトを返す
        >>> Bool() # 新たな補助変数を生成して返す
        ?1
        >>> p == Bool("p")
        True
        >>> ~p # 負リテラル
        ~p
        >>> Bool("p", False) == ~p
        True
        >>> p == ~~p
        True
        >>> abs(~p) # 符号を除いた命題変数を返す
        p
        >>> TRUE # 真の命題定数
        TRUE
        >>> FALSE # 偽の命題定数
        FALSE
        >>> TRUE == ~FALSE
        True
        >>> p(1,2) # 命題変数名が"p(1,2)"のBoolオブジェクトを生成する
        p(1,2)
        >>> p(1) == p("1") # どちらの命題変数名も"p(1)"になり，同じBoolオブジェクトとみなされる
        True
    """
    __auxCount = 0
    __auxPrefix = "?"
    __badChars = "#~=@"

    def __init__(self, name=None, positive=True, internal=False):
        if str(name) == "FALSE":
            name = "TRUE"
            positive = not positive
        self.name = name = str(name or "")
        self.positive = positive
        self.internal = internal
        if not internal and any(c in name for c in self.__badChars):
            raise CspsatException(f"Boolのname指定エラー: {name}")
        if not name:
            Bool.__auxCount += 1
            self.name = Bool.__auxPrefix + str(Bool.__auxCount)

    def isAux(self):
        """このBoolオブジェクトが補助変数ならTrueを返す．

        変数名が "?" から始まれば補助変数とみなされる．

        Returns:
            このBoolオブジェクトが補助変数ならTrue．そうでないならFalse．

        Examples:
            >>> Bool("p").isAux()
            False
            >>> Bool().isAux()
            True
        """
        return self.name.startswith(Bool.__auxPrefix)

    def __call__(self, *args):
        if len(args) == 0:
            return self
        name = self.name + "(" + ",".join(map(str, args)) + ")"
        return Bool(name, self.positive, internal=self.internal)

    # def __getitem__(self, arg):
    #     if isinstance(arg, tuple):
    #         return self(*arg)
    #     return self(arg)

    def __invert__(self):
        return Bool(self.name, not self.positive, internal=self.internal)

    def __abs__(self):
        """このBoolオブジェクトから符号を除いた(positive=Trueにした)Boolオブジェクト(正リテラル)を返す．

        Returns:
            このBoolオブジェクトが正リテラルならこのBoolオブジェクトそのもの，
            負リテラルなら正の符号にしたBoolオブジェクト．

        Examples:と
            >>> p = Bool("p")
            >>> p == abs(~p)
            True
        """
        if self.positive:
            return self
        return Bool(self.name, True, internal=self.internal)

    def __lt__(self, other):
        return self.name < other.name or (self.name == other.name and self.positive < other.positive)

    def __eq__(self, other):
        return isinstance(other, Bool) and self.name == other.name and self.positive == other.positive

    def __hash__(self):
        return hash(self.name) + (1 if self.positive else 0)

    def __repr__(self):
        if self.name == "TRUE":
            return self.name if self.positive else "FALSE"
        return self.name if self.positive else "~" + self.name

TRUE = Bool("TRUE")
"""真を表す命題定数 (Boolオブジェクト)．
"""

FALSE = ~TRUE
"""偽を表す命題定数 (Boolオブジェクト)．
"""

class Var():
    """整数変数を表すクラス．

    同じ整数変数名を持つVarオブジェクトは同一とみなされる．

    Args:
        name (str, optional): 整数変数名の文字列．整数変数名を指定しなければ新たな名前を付けた補助変数を生成する．
            変数名が "?" から始まる場合，補助変数とみなされる．
            変数名に "#", "~", "=", "@" を含んではならない (内部的に利用しているため)．

    Examples:
        >>> x = Var("x") # 整数変数名が"x"のVarオブジェクトを返す
        >>> Var() # 新たな補助変数を生成して返す
        ?1
        >>> x(1) == x("1") # どちらの整数変数名も"x(1)"になり，同じVarオブジェクトとみなされる
        True
    """
    __auxCount = 0
    __auxPrefix = "?"
    __badChars = "#~=@"

    def __init__(self, name=None):
        self.name = name = str(name or "")
        if any(c in name for c in self.__badChars):
            raise CspsatException(f"Varのname指定エラー: {name}")
        if not name:
            Var.__auxCount += 1
            self.name = Var.__auxPrefix + str(Var.__auxCount)

    def isAux(self):
        """このVarオブジェクトが補助変数ならTrueを返す．

        変数名が "?" から始まれば補助変数とみなされる．

        Returns:
            このVarオブジェクトが補助変数ならTrue．そうでないならFalse．

        Examples:
            >>> Var("x").isAux()
            False
            >>> Var().isAux()
            True
        """
        return self.name.startswith(Var.__auxPrefix)

    def __call__(self, *args):
        if len(args) == 0:
            return self
        name = self.name + "(" + ",".join(map(str, args)) + ")"
        return Var(name)

    # def __getitem__(self, arg):
    #     if isinstance(arg, tuple):
    #         return self(*arg)
    #     return self(arg)

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name

    def __hash__(self):
        return hash(self.name) + 314

    def __repr__(self):
        return self.name

class Wsum():
    """線形和を表すクラス．

    整数変数(:obj:`Var`)あるいは命題変数(:obj:`Bool`)の線形和を表す．
    線形和 :math:`a_1 x_1 + a_2 x_2 + \\cdots + a_n x_n + c` は
    :math:`\\textrm{Wsum}(\\{x_1:a_1, x_2:a_2, \\ldots, x_n:a_n\\}, c)` として表される．

    Args:
        wsum (dict, optional): 変数をキーとし，値が係数(int)の辞書(dict)．
            定数のみ整数変数のみ命題変数のみでもよい．
        c (int, optional): 定数部分．

    Attributes:
        wsum (dict): 変数をキーとして，値が係数の辞書(dict)．
            係数が0の場合は含まれない．
        c (int): 定数部分．

    Examples:
        >>> x = Var("x")
        >>> Wsum() # 0
        Wsum({}, 0)
        >>> Wsum(x) # x
        Wsum({x: 1}, 0)
        >>> Wsum(x).mul(3).add(1) # 3*x+1
        Wsum({x: 3}, 1)
    """
    def __init__(self, wsum=None, c=0):
        s = wsum or {}
        wsum = {}
        self.c = c
        match s:
            case _ if isinstance(s, dict):
                wsum = s.copy()
            case _ if isinstance(s, int):
                self.c += s
            case _ if s == FALSE:
                pass
            case _ if s == TRUE:
                self.c += 1
            case Bool():
                if s.positive:
                    wsum[s] = 1
                else:
                    wsum[~s] = -1
                    self.c += 1
            case Var():
                wsum[s] = 1
            case _:
                raise CspsatException(f"Wsumの引数エラー: {s}")
        self.wsum = {}
        for (v, a) in wsum.items():
            if a != 0:
                self.wsum[v] = a

    def variables(self):
        """この線形和に含まれる整数変数のリストを返す．

        Returns:
            整数変数(:obj:`Var`)のリストを返す．

        Examples:
            >>> x = Var("x")
            >>> Wsum(x(1)).add(x(2)).variables()
            dict_keys([x(1), x(2)])
        """
        return self.wsum.keys()

    def coef(self, v):
        """この線形和に含まれる変数vの係数を返す．

        Args:
            v (Var | Bool): 変数．

        Returns:
            変数vの係数．変数vが含まれなければ0．

        Examples:
            >>> x = Var("x")
            >>> Wsum(x).mul(3).coef(x)
            3
            >>> Wsum(x).mul(3).coef(x(1))
            0
        """
        return self.wsum.get(v, 0)

    def isConstant(self):
        """この線形和が定数ならTrueを返す．

        Returns:
            この線形和が定数ならTrue，そうでないならFalse．

        Examples:
            >>> x = Var("x")
            >>> Wsum(x).isConstant()
            False
            >>> Wsum(x).sub(x).isConstant()
            True
        """
        return len(self.variables()) == 0

    def where(self, v, d):
        """この線形和で変数vの値がdに等しいとした場合の線形和を返す．

        Args:
            v (Var | Bool): 整数変数．
            d (int): vの値．

        Returns:
            整数変数vがdに等しいとした場合の線形和を返す．整数変数vが含まれなければ，この線形和そのものを返す．

        Examples:
            >>> x = Var("x")
            >>> Wsum(x(1)).add(x(2)).where(x(1), 2)
            Wsum({x(2): 1}, 2)
            >>> Wsum(x(1)).add(x(2)).where(x(3), 2)
            Wsum({x(1): 1, x(2): 1}, 0)
        """
        if self.coef(v) == 0:
            return self
        c = self.c + self.coef(v) * d
        wsum = self.wsum.copy()
        del wsum[v]
        return Wsum(wsum, c)

    def mul(self, a):
        """この線形和を定数倍した線形和を返す．

        Args:
            a (int): 定数倍の値．

        Returns:
            この線形和をa倍した線形和．

        Examples:
            >>> x = Var("x")
            >>> Wsum(x).add(1).mul(3)
            Wsum({x: 3}, 3)
        """
        if not isinstance(a, int):
            raise CspsatException(f"Wsumのmul引数エラー: {a}")
        wsum = self.wsum.copy()
        for v in wsum:
            wsum[v] *= a
        return Wsum(wsum, self.c * a)

    def neg(self):
        """この線形和をマイナスにした線形和を返す．

        Returns:
            この線形和をマイナスにした線形和を返す．

        Examples:
            >>> x = Var("x")
            >>> Wsum(x).add(1).neg()
            Wsum({x: -1}, -1)
        """
        return self.mul(-1)

    def add(self, other):
        """この線形和に他の線形和を加えた線形和を返す．

        Args:
            other (Wsum): 加える線形和．Wsumオブジェクトでない場合は，いったんWsumオブジェクトに変換したのち加えられる．

        Returns:
            和の線形和を返す．

        Examples:
            >>> x = Var("x")
            >>> Wsum(x).add(1)
            Wsum({x: 1}, 1)
            >>> Wsum(x).add(x)
            Wsum({x: 2}, 0)
        """
        if not isinstance(other, Wsum):
            other = Wsum(other)
        wsum = self.wsum.copy()
        for (v, a) in other.wsum.items():
            wsum[v] = wsum.get(v, 0) + a
        return Wsum(wsum, self.c + other.c)

    def sub(self, other):
        """この線形和から他の線形和を引いた線形和を返す．

        Args:
            other (Wsum): 引く線形和．Wsumオブジェクトでない場合は，いったんWsumオブジェクトに変換したのち加えられる．

        Returns:
            差の線形和を返す．

        Examples:
            >>> x = Var("x")
            >>> Wsum(x).sub(1)
            Wsum({x: 1}, -1)
            >>> Wsum(x).sub(x)
            Wsum({}, 0)
        """
        if not isinstance(other, Wsum):
            other = Wsum(other)
        return self.add(other.neg())

    def value(self, model):
        """与えられた値割当てにおけるこの線形和の値を返す．

        Args:
            model (dict): 変数に対する値が定められた辞書(dict)．

        Returns:
            線形和の値．

        Examples:
            >>> x = Var("x")
            >>> Wsum(x).add(1).value({ x:2 })
            3
        """
        s = self.c
        for v, a in self.wsum.items():
            s += a * model[v]
        return s

    def toExpr(self):
        """この線形和を表す式のデータ(:obj:`cspsat.csp` 参照)を返す．

        Returns:
            線形和を表す式 (list)．

        Examples:
            >>> x = Var("x")
            >>> w = Wsum({x(1): 2, x(2): 1}, 3)
            >>> w.toExpr()
            ['+', ['*', x(1), 2], ['*', x(2), 1], 3]
        """
        expr = ["+"]
        for v, a in self.wsum.items():
            expr.append(["*",v,a])
        expr.append(self.c)
        return expr

    def __eq__(self, other):
        return type(self) == type(other) and self.wsum == other.wsum and self.c == other.c

    def __hash__(self):
        return hash([self.wsum, self.c])

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"Wsum({self.wsum}, {self.c})"

class Binary():
    """ビット列演算を符号化するためのクラス．クラスメソッドを含む．

    ビット列は :obj:`Bool` のリストで，最下位ビットが先頭とする．

    Attributes:
        bins (list of list): bins[i] は i ビット目に対する命題論理式のリストを保持する．
            その和の最下位ビットが i ビット目を表す．
            bins[i][j] に現れる命題論理式は以下のいずれかの形をしている (xiはリテラル)．

             * リテラル (:obj:`Bool`)
             * ["and", x1, x2, ..., xn]
             * ["or", x1, x2, ..., xn]
             * ["xor", x1, x2]
    """

    @classmethod
    def _bits(cls, k):
        yy = []
        while k > 0:
            yy.append(TRUE if k & 1 else FALSE)
            k = k >> 1
        return yy

    @classmethod
    def eqK(cls, xx, k):
        """ビット列xxについて xx == k を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            k (int): 比較する整数．

        Yields:
            xx == k を表すCNF式．
        """
        if k < 0 or (1<<len(xx)) <= k:
            yield []
        else:
            yield from Binary.eq(xx, Binary._bits(k))

    @classmethod
    def neK(cls, xx, k):
        """ビット列xxについて xx != k を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            k (int): 比較する整数．

        Yields:
            xx != k を表すCNF式．
        """
        if k < 0 or (1<<len(xx)) <= k:
            yield from []
        else:
            yield from Binary.ne(xx, Binary._bits(k))

    @classmethod
    def geK(cls, xx, k):
        """ビット列xxについて xx >= k を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            k (int): 比較する整数．

        Yields:
            xx >= k を表すCNF式．
        """
        if k <= 0:
            yield from []
        elif k >= (1<<len(xx)):
            yield []
        else:
            yield from Binary.ge(xx, Binary._bits(k))

    @classmethod
    def gtK(cls, xx, k):
        """ビット列xxについて xx > k を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            k (int): 比較する整数．

        Yields:
            xx > k を表すCNF式．
        """
        if k < 0:
            yield from []
        elif k >= (1<<len(xx))-1:
            yield []
        else:
            yield from Binary.gt(xx, Binary._bits(k))

    @classmethod
    def leK(cls, xx, k):
        """ビット列xxについて xx <= k を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            k (int): 比較する整数．

        Yields:
            xx <= k を表すCNF式．
        """
        if k < 0:
            yield []
        elif k >= (1<<len(xx))-1:
            yield from []
        else:
            yield from Binary.le(xx, Binary._bits(k))

    @classmethod
    def ltK(cls, xx, k):
        """ビット列xxについて xx < k を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            k (int): 比較する整数．

        Yields:
            xx < k を表すCNF式．
        """
        if k <= 0:
            yield []
        elif k >= (1<<len(xx)):
            yield from []
        else:
            yield from Binary.lt(xx, Binary._bits(k))

    @classmethod
    def eq(cls, xx, yy):
        """ビット列xx, yyについて xx == yy を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            yy (list): :obj:`Bool` のリスト．

        Yields:
            xx == y を表すCNF式．
        """
        def x(i):
            return xx[i] if i < len(xx) else FALSE
        def y(i):
            return yy[i] if i < len(yy) else FALSE
        n = max(len(xx), len(yy))
        for i in range(n):
            yield [ ~x(i), y(i) ]
            yield [ x(i), ~y(i) ]

    @classmethod
    def ne(cls, xx, yy):
        """ビット列xx, yyについて xx != yy を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            yy (list): :obj:`Bool` のリスト．

        Yields:
            xx != y を表すCNF式．
        """
        def x(i):
            return xx[i] if i < len(xx) else FALSE
        def y(i):
            return yy[i] if i < len(yy) else FALSE
        n = max(len(xx), len(yy))
        p = Bool()
        yield [ p(i) for i in range(n) ]
        for i in range(n):
            yield [ ~p(i), ~x(i), ~y(i) ]
            yield [ ~p(i), x(i), y(i) ]

    @classmethod
    def ge(cls, xx, yy):
        """ビット列xx, yyについて xx >= yy を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            yy (list): :obj:`Bool` のリスト．

        Yields:
            xx >= y を表すCNF式．
        """
        yield from Binary.le(yy, xx)

    @classmethod
    def gt(cls, xx, yy):
        """ビット列xx, yyについて xx > yy を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            yy (list): :obj:`Bool` のリスト．

        Yields:
            xx > y を表すCNF式．
        """
        yield from Binary.lt(yy, xx)

    @classmethod
    def le(cls, xx, yy, less=False):
        """ビット列xx, yyについて xx <= yy を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            yy (list): :obj:`Bool` のリスト．

        Yields:
            xx <= y を表すCNF式．
        """
        def x(i):
            return xx[i] if i < len(xx) else FALSE
        def y(i):
            return yy[i] if i < len(yy) else FALSE
        n = max(len(xx), len(yy))
        a = Bool()
        yield [ a(n) ]
        for i in reversed(range(n)):
            yield [ ~a(i+1), ~x(i), a(i) ]
            yield [ ~a(i+1), ~x(i), y(i) ]
            yield [ ~a(i+1), a(i), y(i) ]
        yield [ ~a(0) ] if less else [ a(0) ]

    @classmethod
    def lt(cls, xx, yy):
        """ビット列xx, yyについて xx < yy を表すCNF式を生成するジェネレータ関数．

        Args:
            xx (list): :obj:`Bool` のリスト．
            yy (list): :obj:`Bool` のリスト．

        Yields:
            xx < y を表すCNF式．
        """
        yield from Binary.le(xx, yy, less=True)

    def __init__(self):
        self.bins = []

    def put(self, i, x):
        """このBinaryオブジェクトのiビット目にxを加える．

        Args:
            i (int): ビット位置．
            x (Bool | list): 加える命題論理式．
        """
        while len(self.bins) <= i:
            self.bins.append([])
        self.bins[i].append(x)

    def addNum(self, n):
        """このBinaryオブジェクトに整数nを加える．

        Args:
            n (int): 加える値．
        """
        i = 0
        while n > 0:
            if n & 1:
                self.put(i, TRUE)
            n = n >> 1
            i = i + 1

    def add(self, xx, a=1):
        """このBinaryオブジェクトにビット列xxを加える．

        Args:
            xx (list of Bool): 加えるビット列 (リテラルのリスト)．
            a (int, optional): 指定すれば a*xx を加える．
        """
        b = 0
        while a >= (1 << b):
            if a & (1 << b):
                for (i,x) in enumerate(xx):
                    self.put(i+b, x)
            b += 1

    def _addMul(self, xx, yy, b=0):
        for (i,x) in enumerate(xx):
            for (j,y) in enumerate(yy):
                z = x if x == y else ["and", x, y]
                self.put(i+j+b, z)

    def addMul(self, xx, yy, a=1):
        """このBinaryオブジェクトに xx*yy*a を加える．

        Args:
            xx (list of Bool): ビット列 (リテラルのリスト)．
            yy (list of Bool): ビット列 (リテラルのリスト)．
            a (int, optional): 指定しなければ1．
        """
        b = 0
        while a >= (1 << b):
            if a & (1 << b):
                self._addMul(xx, yy, b=b)
            b += 1

    def _addPower(self, xx, n, b=0):
        for ii in itertools.product(range(len(xx)), repeat=n):
            jj = sorted(list(set(ii)))
            z = ["and", *[ xx[i] for i in jj ]]
            self.put(sum(ii)+b, z)

    def addPower(self, xx, n, a=1):
        """このBinaryオブジェクトに xx^n*a を加える．

        Args:
            xx (list of Bool): ビット列 (リテラルのリスト)．
            n (int): ベキ乗．
            a (int, optional): 指定しなければ1．
        """
        b = 0
        while a >= (1 << b):
            if a & (1 << b):
                self._addPower(xx, n, b=b)
            b += 1

    def isSimpleSeq(self):
        """現在の bins がビット列 (リテラルの列)を表しているなら真を返す．

        Returns:
            現在の bins がビット列 (リテラルの列)を表しているなら真．
        """
        for b in self.bins:
            if len(b) > 1 or (len(b) == 1 and not isinstance(b[0], Bool)):
                return False
        return True

    def toSimpleSeq(self):
        """すべての bins[i] の長さが1以下のとき，それらのビット列 (リテラルの列)を返す．

        Returns:
            ビット列 (:obj:`Bool` の列)．
        """
        return [ FALSE if len(b) == 0 else b[0] for b in self.bins ]

    def toBchain(self, zz):
        """このオブジェクトとビット列zzが等しいという制約を生成するジェネレータ関数．

        Daddaのアルゴリズムを使用している．

        Args:
            zz (list of Bool): ビット列．

        Yields:
            このオブジェクトとビット列zzが等しいという制約．
        """
        bins = self.bins
        bchain = []

        def z(i):
            return zz[i] if i < len(zz) else FALSE

        def toBool(x):
            b = x
            if not isinstance(b, Bool):
                b = Bool()
                bchain.append(["equ", b, x])
            return b

        def preprocess():
            while len(bins) < len(zz):
                bins.append([])
            i = 0
            while i < len(bins):
                j = 0
                while j < len(bins[i]):
                    b = bins[i][j]
                    if b in bins[i][j+1:]:
                        bins[i].remove(b)
                        bins[i].remove(b)
                        if i+1 >= len(bins):
                            bins.append([])
                        bins[i+1].append(b)
                    else:
                        j += 1
                i += 1

        preprocess()
        i = 0
        while i < len(bins):
            match bins[i]:
                case []:
                    bchain.append(["equ", z(i), FALSE])
                    i += 1
                case [x1]:
                    bins[i] = []
                    bchain.append(["equ", z(i), x1])
                    i += 1
                case [x1,x2]:
                    bins[i] = []
                    (b1, b2) = (toBool(x1), toBool(x2))
                    bchain.append(["equ", z(i), ["xor",b1,b2]])
                    self.put(i+1, ["and",b1,b2])
                    i += 1
                case [x1,x2,x3,*xx]:
                    bins[i] = xx
                    (b1, b2, b3) = (toBool(x1), toBool(x2), toBool(x3))
                    t = Bool()
                    bchain.append(["equ", t(1), ["xor",b1,b2]])
                    bchain.append(["equ", t(2), ["xor",t(1),b3]])
                    bchain.append(["equ", t(3), ["and",b1,b2]])
                    bchain.append(["equ", t(4), ["and",t(1),b3]])
                    self.put(i, t(2))
                    self.put(i+1, ["or",t(3),t(4)])
        return bchain

    def toCNF(self, zz):
        """このオブジェクトとビット列zzが等しいというCNF式を生成するジェネレータ関数．

        Daddaのアルゴリズムを使用している．

        Args:
            zz (list of Bool): ビット列．

        Yields:
            このオブジェクトとビット列zzが等しいという節．
        """
        for formula in self.toBchain(zz):
            match formula:
                case ["equ", x, y] if isinstance(y, Bool):
                    yield [~x, y]
                    yield [x, ~y]
                case ["equ", x, ["and", *yy]]:
                    yield from [ [~x, y] for y in yy ]
                    yield [x, *[ ~y for y in yy]]
                case ["equ", x, ["or", *yy]]:
                    yield [~x, *yy]
                    yield from [ [x, ~y] for y in yy ]
                case ["equ", x, ["xor", y1, y2]]:
                    yield [~x, ~y1, ~y2]
                    yield [~x, y1, y2]
                    yield [x, ~y1, y2]
                    yield [x, y1, ~y2]
                case _:
                    raise CspsatException(f"使用できない論理式: {formula}")

class BinaryEquation():
    """ビット列演算のためのクラス．

    Args:
        encoder (LogEncoder): 利用するLogEncoderを指定する．

    Examples:
        >>> encoder = LogEncoder()
        >>> b = BinaryEquation(encoder)
        >>> x = Var("x")
        >>> encoder.defInt(x, 0, 10)
        >>> b.addPower(x, 2) 
        >>> b.addNum(-5)
        >>> cnf = b.cmp0(">=") # x**2 - 5 >= 0 のCNF式
    """
    def __init__(self, encoder):
        if not isinstance(encoder, LogEncoder):
            raise CspsatException("LogEncoder以外では利用できない")
        self.encoder = encoder
        self.binaryL = Binary()
        self.ubL = 0
        self.binaryR = Binary()
        self.ubR = 0
        self.c = 0

    def _toBits(self, x):
        if isinstance(x, int):
            xx = []
            i = 0
            while x >= (1<<i):
                xx.append(TRUE if x & (1<<i) else FALSE)
                i += 1
            (lb, ub) = (0, x)
        elif isinstance(x, Bool):
            xx = [x]
            (lb, ub) = (0, 1)
        elif isinstance(x, Var):
            xx = self.encoder.getBools(x) # x-lb(x)
            (lb, ub) = (self.encoder.intLb(x), self.encoder.intUb(x))
        else:
            raise CspsatException(f"整数でも変数でもない: {x}")
        return (xx, lb, ub)

    def addNum(self, n):
        """整数定数を加える．
        
        Args:
            n (int): 加える整数定数．
        """
        self.c += n

    def add(self, x, a=1):
        """整数変数の定数倍 (x*a)を加える．

        Args:
            x (Var): 加える整数変数．
            a (int,optional): 定数倍の値 (指定しなければ1)．
        """
        (xx, lb, ub) = self._toBits(x)
        # a*x = a*(xx+lb) = a*xx + a*lb
        if a > 0:
            self.binaryL.add(xx, a=a)
            self.ubL += (ub-lb) * a
        elif a < 0:
            self.binaryR.add(xx, a=-a)
            self.ubR += (ub-lb) * -a
        self.addNum(a * lb)

    def addMul(self, x, y, a=1):
        """整数変数の積の定数倍 (x*y*a)を加える．

        Args:
            x (Var): 積の整数変数．
            y (Var): 積の整数変数．
            a (int,optional): 定数倍の値 (指定しなければ1)．
        """
        (xx, lbx, ubx) = self._toBits(x)
        (yy, lby, uby) = self._toBits(y)
        # a*x*y = a*(xx+lbx)*(yy+lby) = a*xx*yy + a*lby*xx + a*lbx*yy + a*lbx*lby
        # = a*xx*yy + a*lby*x + a*lbx*y - a*lbx*lby
        if a > 0:
            self.binaryL.addMul(xx, yy, a=a)
            self.ubL += (ubx-lbx) * (uby-lbx) * a
        else:
            self.binaryR.addMul(xx, yy, a=-a)
            self.ubR += (ubx-lbx) * (uby-lbx) * -a
        self.add(x, a=a*lby)
        self.add(y, a=a*lbx)
        self.addNum(-a*lbx*lby)

    def addPower(self, x, n, a=1):
        """整数変数のベキ乗の定数倍 ((x**n)*a)を加える．

        Args:
            x (Var): 整数変数．
            n (int): ベキ乗．
            a (int,optional): 定数倍の値 (指定しなければ1)．
        """
        (xx, lb, ub) = self._toBits(x)
        if lb != 0:
            raise CspsatException(f"変数の下限が0でない: {x}")
        if a > 0:
            self.binaryL.addPower(xx, n, a=a)
            self.ubL += ub**n * a
        else:
            self.binaryR.addPower(xx, n, a=-a)
            self.ubL += ub**n * -a

    def _encodeLR(self):
        def _encodeBinary(binary, ub):
            if binary.isSimpleSeq():
                ss = binary.toSimpleSeq()
                cnf = []
            else:
                m = ub.bit_length()
                s = Bool()
                ss = [ s(i) for i in range(m) ]
                cnf = binary.toCNF(ss)
            return (ss, cnf)

        if self.c > 0:
            self.binaryL.addNum(self.c)
            self.ubL += self.c
        elif self.c < 0:
            self.binaryR.addNum(-self.c)
            self.ubR += -self.c
        (ssL, cnfL) = _encodeBinary(self.binaryL, self.ubL)
        (ssR, cnfR) = _encodeBinary(self.binaryR, self.ubR)
        return (ssL, cnfL, ssR, cnfR)

    def cmp0(self, cmp):
        """0との比較のCNF式をyieldするジェネレータ関数．

        Args:
            cmp (str): 比較演算子 ("==", "!=", "<=", "<", ">=", ">"のいずれか)．

        Yields:
            CNF式．
        """
        (ssL, cnfL, ssR, cnfR) = self._encodeLR()
        yield from cnfL
        yield from cnfR
        match cmp:
            case "==":
                yield from Binary.eq(ssL, ssR)
            case "!=":
                yield from Binary.ne(ssL, ssR)
            case "<=":
                yield from Binary.le(ssL, ssR)
            case "<":
                yield from Binary.lt(ssL, ssR)
            case ">=":
                yield from Binary.ge(ssL, ssR)
            case ">":
                yield from Binary.gt(ssL, ssR)
            case _:
                raise CspsatException(f"比較演算子の指定エラー: {cmp}")

class SeqCounter():
    """基数制約を符号化するためのクラス．クラスメソッドからなる．
    """

    @classmethod
    def eqK(cls, xx, k):
        """基数制約 xx[0] + xx[1] + ... == k を逐次カウンタ法でSAT符号化したCNF式をyieldするジェネレータ関数．

        Args:
            xx (list of Bool): リテラルのリスト．
            k (int): 整数．

        Yields:
            CNF式．
        """
        yield from cls.leK(xx, k)
        yield from cls.geK(xx, k)

    @classmethod
    def neK(cls, xx, k):
        """基数制約 xx[0] + xx[1] + ... != k を逐次カウンタ法でSAT符号化したCNF式をyieldするジェネレータ関数．

        Args:
            xx (list of Bool): 命題変数のリスト．
            k (int): 整数．

        Yields:
            CNF式．
        """
        p = Bool()
        yield from [ [~p, *clause] for clause in cls.ltK(xx, k) ]
        yield from [ [ p, *clause] for clause in cls.gtK(xx, k) ]

    @classmethod
    def geK(cls, xx, k):
        """基数制約 xx[0] + xx[1] + ... >= k を逐次カウンタ法でSAT符号化したCNF式をyieldするジェネレータ関数．

        Args:
            xx (list of Bool): 命題変数のリスト．
            k (int): 整数．

        Yields:
            CNF式．
        """
        yield from cls.leK([ ~x for x in xx ], len(xx)-k)

    @classmethod
    def gtK(cls, xx, k):
        """基数制約 xx[0] + xx[1] + ... > k を逐次カウンタ法でSAT符号化したCNF式をyieldするジェネレータ関数．

        Args:
            xx (list of Bool): 命題変数のリスト．
            k (int): 整数．

        Yields:
            CNF式．
        """
        yield from cls.geK(xx, k+1)

    @classmethod
    def leK(cls, xx, k):
        """基数制約 xx[0] + xx[1] + ... <= k を逐次カウンタ法でSAT符号化したCNF式をyieldするジェネレータ関数．

        s を指定すれば，s(i,j) (i=0,...,n, j=0,...k+1)は，xx[0] + ... + x[i-1] >= j を表す．

        Args:
            xx (list of Bool): 命題変数のリスト．
            k (int): 整数．
            s (Bool, optional): 命題変数．

        Yields:
            CNF式．
        """
        n = len(xx)
        if k < 0:
            yield []
        elif k == 0:
            yield from [ [ ~x ] for x in xx ]
        elif k == n-1:
            yield [ ~x for x in xx ]
        elif k >= n:
            yield from []
        else:
            s = Bool()
            # s(i,j) : xx[0] + ... xx[i-1] >= j
            for l in range(n-k):
                for j in range(1, k+2):
                    i = l + j
                    if j == 1:
                        yield [ ~xx[i-1], s(i,j) ]
                    elif j <= k:
                        yield [ ~xx[i-1], ~s(i-1,j-1), s(i,j) ]
                    else:
                        yield [ ~xx[i-1], ~s(i-1,j-1) ]
                    if l > 0 and j <= k:
                        yield [ ~s(i-1,j), s(i,j) ]

    @classmethod
    def ltK(cls, xx, k):
        """基数制約 xx[0] + xx[1] + ... < k を逐次カウンタ法でSAT符号化したCNF式をyieldするジェネレータ関数．

        Args:
            xx (list of Bool): 命題変数のリスト．
            k (int): 整数．

        Yields:
            CNF式．
        """
        yield from cls.leK(xx, k-1)

class Preprocessor():
    """SAT符号化の前処理のためのクラス．

    制約を符号化し拡張CNF式を生成する．

    拡張CNF (XCNF)の形式は以下の通り．ここで pi はリテラルで，n>=0 である．

    * [p1, ..., pn] (通常の節)
    * [p1, ..., pn, C] (Cを含む節)

    C は以下のいずれかである．ここで W は線形和を表している式，kは整数定数である．

    * ["_eq0", W] (W == 0 を表す)
    * ["_ne0", W] (W != 0 を表す)
    * ["_le0", W] (W <= 0 を表す)
    * ["leK", [p1, ..., pn], K] (p1+...+pn <= k を表す)

    Args:
        encoder (Encoder): 使用するEncoder．
        introduceAux (bool,optional): 真なら符号化時に補助変数を導入する (指定しないなら偽)．
            :obj:`.DirectEncoder`, :obj:`.OrderEncoder` からの利用では真を指定している．
    """
    def __init__(self, encoder, introduceAux=False):
        self.encoder = encoder
        self.introduceAux = introduceAux

    def toWsum(self, s):
        """制約充足問題の式sを表す線形和を返す．

        Args:
            s (list): 制約充足問題の式．

        Returns:
            sを表す線形和．

        Raises:
            CspsatException: 式の構文が間違っている．
        """
        for hook in self.encoder.functionHooks:
            s = hook(s, self.encoder)
        wsum = None
        match s:
            case _ if isinstance(s, (int, Var, Bool)):
                wsum = Wsum(s)
            case ["+", *ss]:
                wsum = Wsum()
                for w in ss:
                    wsum = wsum.add(self.toWsum(w))
            case ["-", s1]:
                wsum = self.toWsum(s1).neg()
            case ["-", s1, s2]:
                wsum = self.toWsum(s1).sub(self.toWsum(s2))
            case ["*", s1, s2]:
                (w1,w2) = (self.toWsum(s1), self.toWsum(s2))
                if w1.isConstant():
                    wsum = w2.mul(w1.c)
                elif w2.isConstant():
                    wsum = w1.mul(w2.c)
                else:
                    raise CspsatException(f"定数乗算でない乗算: {s}")
            case _:
                raise CspsatException(f"式の構文エラー: {s}")
        return wsum

    def isBC(self, w):
        """線形和wが基数制約に変換できるかどうかを調べる．

        すなわち w のすべての変数のドメインが{0,1}で，係数が-1か1である．

        Args:
            w (Wsum): 線形和．

        Returns:
            基数制約に変換できるなら真を返す．
        """
        for (x,a) in w.wsum.items():
            if a < -1 or a > 1 or not self.encoder.isBoolLike(x):
                return False
        return True

    def toBC(self, w):
        """線形和wを基数制約に変換するためのタプル(xx,k)を返す．

        Args:
            w (Wsum): 線形和．

        Returns:
            タプル(xx,k)．xxは命題変数のリスト．kは基数の値．
        """
        xx = []
        k = - w.c
        for (x,a) in w.wsum.items():
            if isinstance(x, Var):
                x = self.encoder.varToBool(x)
            if a == -1:
                x = ~x
                k += 1
            xx.append(x)
        return (xx, k)

    def _decompose(self, w, le=True):
        if len(w.wsum) <= 3:
            return w
        enc = self.encoder
        (choice, lb, ub) = (None, None, None)
        for (v1,v2) in itertools.combinations(w.variables(), 2):
            (lb1, ub1, a1) = (enc.intLb(v1), enc.intUb(v1), w.coef(v1))
            (lb2, ub2, a2) = (enc.intLb(v2), enc.intUb(v2), w.coef(v2))
            (lb1, ub1) = (min(lb1*a1, ub1*a1), max(lb1*a1, ub1*a1))
            (lb2, ub2) = (min(lb2*a2, ub2*a2), max(lb2*a2, ub2*a2))
            (lb0, ub0) = (lb1+lb2, ub1+ub2)
            if choice is None or ub0-lb0 < ub-lb:
                (choice, lb, ub) = ((v1,v2), lb0, ub0)
        (v1, v2) = choice
        (a1, a2) = (w.coef(v1), w.coef(v2))
        w = w.where(v1, 0).where(v2, 0)
        (lbw, _) = enc.wsumBound(w)
        s = Var()
        enc.put(["int", s, lb, min(ub, -lbw)])
        if le:
            enc.put(["<=", ["+", ["*", v1, a1], ["*", v2, a2]], s])
        else:
            enc.put(["==", ["+", ["*", v1, a1], ["*", v2, a2]], s])
        w = w.add(s)
        return self._decompose(w, le=le)

    def _processEq0(self, w):
        (lb, ub) = self.encoder.wsumBound(w)
        if lb == ub == 0:
            yield from [] # True
            return
        if lb > 0 or ub < 0:
            yield [] # False
            return
        if self.introduceAux:
            w = self._decompose(w, le=False)
        yield [ ["_eq0", w.toExpr()] ]

    def _processNe0(self, w):
        (lb, ub) = self.encoder.wsumBound(w)
        if lb > 0 or ub < 0:
            yield from [] # True
            return
        if lb == ub == 0:
            yield [] # False
            return
        if self.introduceAux:
            w = self._decompose(w, le=False)
        yield [ ["_ne0", w.toExpr()] ]

    def _processLe0(self, w):
        (lb, ub) = self.encoder.wsumBound(w)
        if ub <= 0:
            yield from [] # True
            return
        if lb > 0:
            yield [] # False
            return
        if self.introduceAux:
            w = self._decompose(w, le=True)
        yield [ ["_le0", w.toExpr()] ]

    def _processLeK(self, xx, k):
        n = len(xx)
        if k > n:
            yield from [] # True
        elif k < 0:
            yield [] # False
        else:
            yield [ ["leK", xx, k] ]

    def _toXCNFdisj(self, constraints):
        xclause = []
        for constraint in constraints:
            cnf = self.toXCNF(constraint)
            c1 = next(cnf, None)
            c2 = next(cnf, None)
            match (c1, c2):
                case (None, _): # cnf is true
                    return
                case ([], _): # cnf is false
                    continue
                case (_, None):
                    xclause.extend(c1)
                case _:
                    p = Bool()
                    xclause.append(p)
                    for c in itertools.chain([c1, c2], cnf):
                        if c is not None:
                            yield [ ~p, *c ]
        ii = []
        for (i,lit) in enumerate(xclause):
            if not isinstance(lit, Bool):
                ii.append(i)
        if len(ii) >= 2:
            for i in ii[:-1]:
                p = Bool()
                yield [ ~p, xclause[i] ]
                xclause[i] = p
        yield xclause

    def _convertBoolLikeList(self, xx, k):
        yy = []
        for x in xx:
            if x in (FALSE, 0):
                pass
            elif x == (TRUE, 1):
                k -= 1
            elif self.encoder.isBoolLike(x):
                if isinstance(x, Var):
                    x = self.encoder.varToBool(x)
                yy.append(x)
            else:
                raise CspsatException(f"基数制約の引数エラー: {x}")
        return (yy, k)

    def toXCNF(self, constraint, positive=True):
        """与えられた制約を拡張CNF式に符号化してyieldするジェネレータ関数．

        Args:
            constraint (list): 制約．
            positive (bool,optional): 偽なら制約の否定を符号化する (指定しなければ真)．

        Yields:
            拡張CNF式．
        """
        for hook in self.encoder.constraintHooks:
            constraint = hook(constraint, self.encoder)
        match constraint:
            case Bool():
                c = constraint if positive else ~constraint
                if c == TRUE:
                    yield from []
                elif c == FALSE:
                    yield []
                else:
                    yield [c]
            case ["not", a] | ["!", a]:
                yield from self.toXCNF(a, not positive)
            case ["and", *args] | ["&&", *args]:
                if not positive:
                    args = [ ["not",a] for a in args ]
                    yield from self.toXCNF(["or", *args])
                else:
                    for a in args:
                        yield from self.toXCNF(a)
            case ["or", *args] | ["||", *args]:
                if not positive:
                    args = [ ["not",a] for a in args ]
                    yield from self.toXCNF(["and", *args])
                else:
                    yield from self._toXCNFdisj(args)
            case ["imp", a1, a2] | ["=>", a1, a2]:
                yield from self.toXCNF(["or", ["not",a1], a2], positive)
            case ["equ", a1, a2] | ["<=>", a1, a2]:
                yield from self.toXCNF(["and", ["imp",a1,a2], ["imp",a2,a1]], positive)
            case ["xor", *args] | ["^", *args]:
                if len(args) == 0:
                    yield from self.toXCNF(FALSE, positive)
                elif len(args) == 1:
                    yield from self.toXCNF(args[0], positive)
                else:
                    (a1, a2) = (args[0], ["xor", *args[1:]])
                    yield from self.toXCNF(["and", ["or",a1,a2], ["or",["not",a1],["not",a2]]], positive)
            case ["eq", a1, a2] | ["==", a1, a2]:
                if not positive:
                    yield from self.toXCNF(["!=", a1, a2])
                else:
                    w = self.toWsum(["-", a1, a2])
                    if self.isBC(w):
                        (xx, k) = self.toBC(w)
                        yield from self.toXCNF(["eqK", xx, k])
                    else:
                        yield from self._processEq0(w)
            case ["ne", a1, a2] | ["!=", a1, a2]:
                if not positive:
                    yield from self.toXCNF(["==", a1, a2])
                else:
                    w = self.toWsum(["-", a1, a2])
                    if self.isBC(w):
                        (xx, k) = self.toBC(w)
                        yield from self.toXCNF(["neK", xx, k])
                    else:
                        yield from self._processNe0(w)
            case ["ge", a1, a2] | [">=", a1, a2]:
                yield from self.toXCNF(["<=", a2, a1], positive)
            case ["gt", a1, a2] | [">", a1, a2]:
                yield from self.toXCNF(["<=", ["+",a2,1], a1], positive)
            case ["lt", a1, a2] | ["<", a1, a2]:
                yield from self.toXCNF(["<=", ["+",a1,1], a2], positive)
            case ["le", a1, a2] | ["<=", a1, a2]:
                if not positive:
                    yield from self.toXCNF(["<=", ["+",a2,1], a1])
                else:
                    w = self.toWsum(["-", a1, a2])
                    if self.isBC(w):
                        (xx, k) = self.toBC(w)
                        yield from self.toXCNF(["leK", xx, k])
                    else:
                        yield from self._processLe0(w)
            case ["eqK", xx, k]:
                if not positive:
                    yield from self.toXCNF(["neK", xx, k])
                else:
                    yield from self.toXCNF(["leK", xx, k])
                    yield from self.toXCNF(["geK", xx, k])
            case ["neK", xx, k]:
                if not positive:
                    yield from self.toXCNF(["eqK", xx, k])
                else:
                    p = Bool()
                    yield from [ [~p, *c] for c in self.toXCNF(["ltK", xx, k]) ]
                    yield from [ [p, *c] for c in self.toXCNF(["gtK", xx, k]) ]
            case ["gtK", xx, k]:
                yield from self.toXCNF(["geK", xx, k+1], positive)
            case ["ltK", xx, k]:
                yield from self.toXCNF(["leK", xx, k-1], positive)
            case ["geK", xx, k]:
                if not positive:
                    yield from self.toXCNF(["leK", xx, k-1])
                else:
                    xx = [ ~x for x in xx ]
                    yield from self.toXCNF(["leK", xx, len(xx)-k])
            case ["leK", xx, k]:
                (xx, k) = self._convertBoolLikeList(xx, k)
                if not positive:
                    xx = [ ~x for x in xx ]
                    k = len(xx) - k - 1
                yield from self._processLeK(xx, k)
            case _:
                raise CspsatException(f"制約の構文エラー: {constraint}")

import itertools
from .csp import LogEncoder
