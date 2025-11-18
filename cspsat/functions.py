"""便利な関数群．

cspsat モジュールの便利な関数群をまとめている．

.. csv-table:: 命題論理式の記法 (Pは命題変数．A1, A2, ..., Anは命題論理式)
   :header: "名称など", "命題論理式の記法"
   :widths: 20, 40

   "真", "TRUE"
   "偽", "FALSE"
   "命題変数 p", "Bool(""p"")"
   "補リテラル", "~P"
   "否定", "[""not"", A1], [""!"", A1]"
   "論理積", "[""and"", A1, ..., An], [""&&"", A1, ..., An]"
   "論理和", "[""or"", A1, ..., An], [""||"", A1, ..., An]"
   "含意", "[""imp"", A1, A2], [""=>"", A1, A2]"
   "同値", "[""equ"", A1, A2], [""<=>"", A1, A2]"
   "排他的論理和", "[""xor"", A1, ..., An], [""^"", A1, ..., An]"

Note:
    本プログラムは学習用の目的で作成されている．
    実用上の問題への適用は想定していない．
    Copyright (c) 2025-- Naoyuki Tamura
    Licensed under the MIT License
"""

def variables(f):
    """命題論理式に含まれている命題変数の集合を返す．

    Args:
        f: 命題論理式．命題論理式の記法については :obj:`.functions` 参照．

    Returns:
        :obj:`.Bool` オブジェクトの集合．

    Examples:
        >>> p = Bool("p")
        >>> variables(["and", p(1), ~p(2), p(2)])
        {p(1), p(2)}
    """
    if isinstance(f, Bool) and f != TRUE and f != FALSE:
        return { abs(f) }
    if isinstance(f, (list,tuple,set)):
        return { v for g in f for v in variables(g) }
    return set()

def assignments(vs):
    """与えられた命題変数のリストに対し，可能な値割当てをyieldするジェネレータ関数．
    各値割当ては，命題変数(:obj:`.Bool` オブジェクト)をキーとする辞書(dict)で，値は0か1．

    Args:
        vs: 命題変数(:obj:`.Bool` オブジェクト)のリスト．

    Yields:
        値割当てを表す辞書(dict)．

    Examples:
        >>> p = Bool("p")
        >>> list(assignments([p(1),p(2)]))
        [{p(1): 0, p(2): 0}, {p(1): 0, p(2): 1}, {p(1): 1, p(2): 0}, {p(1): 1, p(2): 1}]
    """
    if len(vs) == 0:
        yield {}
    else:
        vs = vs.copy()
        v = vs.pop()
        for a in assignments(vs):
            yield { **a, v: 0 }
            yield { **a, v: 1 }

def value(f, a):
    """与えられた命題論理式 f と値割当て a に対し，f の真理値を返す関数．

    命題論理式 f はいったん :obj:`toNF` 関数で否定・論理積・論理和のみの式に変換され，その後，真理値を計算する．
    値割当て中に含まれない命題変数の真理値はNoneになる．
    否定の引数の式の真理値がNoneの場合，否定の真理値はNoneになる．
    論理積の引数の真理値に0がある場合，論理積の真理値は0になる．
    論理積の引数の真理値に0がなくNoneがある場合，論理積の真理値はNoneになる．
    論理和の引数の真理値に1がある場合，論理和の真理値は1になる．
    論理和の引数の真理値に1がなくNoneがある場合，論理和の真理値はNoneになる．

    Args:
        f: 命題論理式．命題論理式の記法については :obj:`.functions` 参照．
        a (dict): 値割当ての辞書(dict)．キーは :obj:`.Bool` オブジェクトで表される命題変数，値は0か1．

    Returns:
        f の真理値 (0か1かNone)．

    Raises:
        CspsatException: 論理式の構文エラー．

    Examples:
        >>> p = Bool("p")
        >>> value(["not",p], {p: 1})
        0
        >>> value(["not",p], {}) # Noneが返される
    """
    def _value(f):
        match f:
            case Bool() if f == FALSE:
                return 0
            case Bool() if f == TRUE:
                return 1
            case Bool():
                if f.positive:
                    return a.get(f, None)
                t1 = a.get(~f, None)
                return 1 - t1 if t1 is not None else None
            case ["not", g1]:
                t1 = _value(g1)
                return 1 - _value(g1) if t1 is not None else None
            case ["and", *gs]:
                t = 1
                for g in gs:
                    match _value(g):
                        case 0:
                            return 0
                        case None:
                            t = None
                return t
            case ["or", *gs]:
                t = 0
                for g in gs:
                    match _value(g):
                        case 1:
                            return 1
                        case None:
                            t = None
                return t
        raise CspsatException(f"論理式の構文エラー: {f}")

    return _value(toNF(f))

def truthTable(*fs):
    """与えられた命題論理式 f1, f2, ... の真理値表を出力する関数．

    Args:
        *fs: 命題論理式 f1, f2, ... の列．命題論理式の記法については :obj:`.functions` 参照．

    Raises:
        CspsatException: 論理式の構文エラー．

    Examples:
        >>> p = Bool("p")
        >>> truthTable(["and",p(1),p(2)], ["or",p(1),p(2)])
        | p(1) | p(2) | ['and', p(1), p(2)] | ['or', p(1), p(2)] |
        |------|------|---------------------|--------------------|
        |   0  |   0  |          0          |          0         |
        |   0  |   1  |          0          |          1         |
        |   1  |   0  |          0          |          1         |
        |   1  |   1  |          1          |          1         |
    """
    def row(xs, ws):
        ss = []
        for i, x in enumerate(xs):
            s = str(x)
            s = " "*(ws[i]>>1) + s + " "*((ws[i]-len(s))>>1)
            ss.append(s[-ws[i]:])
        return "| " + " | ".join(ss) + " |"
    fs = list(fs)
    vs = sorted(variables(fs))
    ws = [ len(str(x)) for x in vs + fs ]
    print(row(vs + fs, ws), flush=True)
    print("|" + "|".join([ "-"*(w+2) for w in ws ]) + "|", flush=True)
    for a in assignments(vs):
        xs1 = [ a[v] for v in vs ]
        xs2 = [ value(f, a) for f in fs ]
        print(row(xs1 + xs2, ws), flush=True)

def isValid(f):
    """与えられた命題論理式 f が恒真かどうかを判定する関数．

    命題論理式 f に対するすべての値割当てで f の真理値が1ならば恒真としてTrueを返す．
    f に含まれる命題変数の個数を :math:`n` とすると :math:`2^n` の時間がかかる．

    Args:
        f: 命題論理式．命題論理式の記法については :obj:`.functions` 参照．

    Returns:
        f が恒真ならTrue．そうでなければFalse．

    Raises:
        CspsatException: 論理式の構文エラー．

    Examples:
        >>> p = Bool("p")
        >>> isValid(["or",p,~p])
        True
    """
    vs = variables(f)
    if len(vs) == 0:
        return value(f, {}) == 1
    for a in assignments(vs):
        if value(f, a) == 0:
            return False
    return True

def isSat(f):
    """与えられた命題論理式 f が充足可能かどうかを判定する関数．

    命題論理式 f に対するある値割当てで f の真理値が1ならば充足可能としてTrueを返す．
    f に含まれる命題変数の個数を :math:`n` とすると :math:`2^n` の時間がかかる．

    Args:
        f: 命題論理式．命題論理式の記法については :obj:`.functions` 参照．

    Returns:
        f が充足可能ならTrue．そうでなければFalse．

    Raises:
        CspsatException: 論理式の構文エラー．

    Examples:
        >>> p = Bool("p")
        >>> isSat(["or",p(1),p(2)])
        True
        >>> isSat(["and",p,~p])
        False
    """
    vs = variables(f)
    if len(vs) == 0:
        return value(f, {}) == 1
    for a in assignments(vs):
        if value(f, a) == 1:
            return True
    return False

def isEquiv(f1, f2):
    """与えられた命題論理式 f1, f2 が論理的に同値かどうかを判定する関数．

    命題論理式 ["equ",f1,f2] が恒真なら f1 と f2 が論理的に同値としてTrueを返す．
    f1, f2 に含まれる命題変数の個数を :math:`n` とすると :math:`2^n` の時間がかかる．

    Args:
        f1: 命題論理式．命題論理式の記法については :obj:`.functions` 参照．
        f2: 命題論理式．

    Returns:
        f1 と f2 が論理的に同値ならTrue．そうでなければFalse．

    Raises:
        CspsatException: 論理式の構文エラー．

    Examples:
        >>> p = Bool("p")
        >>> isEquiv(p, ["and",p,p])
        True
    """
    return isValid(["equ", f1, f2])

def models(f, num=1):
    """与えられた命題論理式 f の複数のモデルを探索し，それらをyieldするジェネレータ関数．

    命題論理式 f のすべての値割当て a に対し，a が f のモデルなら a をyieldする．
    探索するモデルの最大個数を num で指定できる．
    f に含まれる命題変数の個数を :math:`n` とすると :math:`2^n` の時間がかかる．

    Args:
        f: 命題論理式．命題論理式の記法については :obj:`.functions` 参照．
        num (int): 探索するモデルの最大個数．0なら全解を探索する (デフォルト値は1)．

    Yields:
        見つかったモデル．キーが :obj:`.Bool` オブジェクトで値が0か1の辞書(dict)である．

    Raises:
        CspsatException: 論理式の構文エラー．

    Examples:
        >>> p = Bool("p")
        >>> for model in models(["or",p(1),p(2)], num=0): print(model)
        {p(1): 0, p(2): 1}
        {p(1): 1, p(2): 0}
        {p(1): 1, p(2): 1}
    """
    vs = variables(f)
    count = 0
    for a in assignments(vs):
        if num != 0 and count >= num:
            break
        if value(f, a) == 1:
            yield a
            count += 1

def toNF(f):
    """与えられた命題論理式 f を，論理演算子として否定・論理積・論理和だけを含む同値な式に変換する．

    含意 :math:`A \\Rightarrow B` は :math:`(\\lnot A) \\lor B` に変換する．
    同値 :math:`A \\Leftrightarrow B` は :math:`((\\lnot A) \\lor B) \\land (A \\lor (\\lnot B))` に変換する．
    排他的論理和 :math:`A \\oplus B` は :math:`(A \\lor B) \\land ((\\lnot A) \\lor (\\lnot B))` に変換する．

    Args:
        f: 命題論理式．

    Returns:
        論理演算子として否定・論理積・論理和だけを含む命題論理式．

    Raises:
        CspsatException: 論理式の構文エラー．

    Examples:
        >>> p = Bool("p")
        >>> toNF(["not",["xor",p(1),p(2)]])
        ['not', ['and', ['or', p(1), p(2)], ['or', ['not', p(1)], ['not', p(2)]]]]
    """
    match f:
        case Bool():
            return f
        case ["imp", g1, g2] | ["=>", g1, g2]:
            return toNF(["or", ["not", g1], g2])
        case ["equ", g1, g2] | ["<=>", g1, g2]:
            return toNF(["and", ["or", ["not", g1], g2], ["or", g1, ["not", g2]]])
        case ["xor", *gs] | ["^", *gs]:
            if len(gs) == 0:
                return FALSE
            if len(gs) == 1:
                return gs[0]
            (g1, g2) = (gs[0], toNF(["xor", *gs[1:]]))
            return toNF(["and", ["or", g1, g2], ["or", ["not", g1], ["not", g2]]])
        case ["not", g1] | ["!", g1]:
            return ["not", toNF(g1)]
        case ["and", *gs] | ["&&", *gs]:
            return ["and", *[ toNF(g) for g in gs]]
        case ["or", *gs] | ["||", *gs]:
            return ["or", *[ toNF(g) for g in gs]]
        case _:
            raise CspsatException(f"論理式の構文エラー: {f}")

def toNNF(f, positive=True):
    """与えられた命題論理式 f を否定標準形に変換する．

    否定標準形は，リテラル(:obj:`.Bool` オブジェクト)および論理積と論理和のみが現れる命題論理式である．
    いったん f を :obj:`toNF` 関数で否定・論理積・論理和だけを含む式に変換したのち，
    :math:`\\lnot \\lnot A` を :math:`A` に，
    :math:`\\lnot (A \\land B)` を :math:`(\\lnot A) \\lor (\\lnot B)` に，
    :math:`\\lnot (A \\lor B)` を :math:`(\\lnot A) \\land (\\lnot B)` に置き換える操作を再帰的に繰り返している．

    Args:
        f: 命題論理式．
        positive (bool, optional): Falseなら ["not",f] を否定標準形に変換する(デフォルト値はTrue)．

    Returns:
        否定標準形の命題論理式．

    Raises:
        CspsatException: 論理式の構文エラー．

    Examples:
        >>> p = Bool("p")
        >>> toNNF(["not",["xor",p(1),p(2)]])
        ['or', ['and', ~p(1), ~p(2)], ['and', p(1), p(2)]]
    """
    def _toNNF(f, positive):
        match f:
            case Bool():
                return f if positive else ~f
            case ["not", g1]:
                return _toNNF(g1, not positive)
            case ["and", *gs]:
                op = "and" if positive else "or"
                return [op, *[ _toNNF(g, positive) for g in gs ]]
            case ["or", *gs]:
                op = "or" if positive else "and"
                return [op, *[ _toNNF(g, positive) for g in gs ]]
    return _toNNF(toNF(f), positive)

def toCNF(f, simplify=True, formula=False):
    """与えられた命題論理式 f を論理積標準形(CNF)に変換する．

    CNF式は複数の節の論理積であり，各節は複数のリテラルの論理和である．
    いったん f を :obj:`toNNF` 関数で否定標準形に変換したのち，
    分配法則を用いて :math:`A \\lor (B \\land C)` を :math:`(A \\lor B) \\land (A \\lor C)` に置き換える操作を再帰的に繰り返している．

    simplify=True の場合，以下の方法でCNF式を簡単化する．

    * 節から命題定数 FALSE を削除する
    * 節に命題定数 TRUE が含まれていれば，その節をCNF式から削除する
    * 節にリテラル p と補リテラル ~p が含まれていれば，その節をCNF式から削除する
    * 節中に同じリテラルが重複して現れていれば，1つにまとめる

    formula=True を指定すれば，結果が節のリストではなく，命題論理式を返す．

    f に含まれる命題変数の個数を :math:`n` とすると最悪の場合CNF式のサイズは :math:`2^n` になる
    (たとえば :math:`(p_1 \\land q_1)\\lor(p_2 \\land q_2)\\lor\\cdots\\lor(p_n \\land q_n)` の場合)．

    Args:
        f: 命題論理式．命題論理式の記法については :obj:`.functions` 参照．
        simplify (bool, optional): Falseなら得られたCNF式を簡単化しない(デフォルト値はTrue)．
        formula (bool, optional): 節のリストでなく，命題論理式を返す．

    Returns:
        CNF式．CNF式は節のリストであり，各節はリテラルのリストである．

    Raises:
        CspsatException: 論理式の構文エラー．

    Examples:
        >>> p = Bool("p")
        >>> toCNF(["not",["xor",p(1),p(2)]])
        [[~p(1), p(2)], [p(1), ~p(2)]]
        >>> toCNF(["not",["xor",p(1),p(2)]], simplify=False)
        [[~p(1), p(1)], [~p(1), p(2)], [~p(2), p(1)], [~p(2), p(2)]]
        >>> toCNF(["not",["xor",p(1),p(2)]], formula=True)
        ['and', ['or', ~p(1), p(2)], ['or', p(1), ~p(2)]]
    """
    def simplifyClause(clause):
        clause = { lit for lit in clause if lit != FALSE }
        for lit in clause:
            if lit == TRUE or ~lit in clause:
                return TRUE
        return sorted(list(clause))

    def _toCNF(f):
        match f:
            case Bool() if f == FALSE:
                return [ [] ]
            case Bool() if f == TRUE:
                return  []
            case Bool():
                return [ [f] ]
            case ["and", *gs]:
                cnf = [ clause for g in gs for clause in _toCNF(g) ]
                if simplify and [] in cnf:
                    cnf = [ [] ]
                return cnf
            case ["or"]:
                return [ [] ]
            case ["or", g1, *gs]:
                cnf = []
                for clause1 in _toCNF(g1):
                    for clause2 in _toCNF(["or", *gs]):
                        clause = clause1 + clause2
                        if simplify:
                            clause = simplifyClause(clause)
                        if clause == []:
                            return [ [] ]
                        if clause != TRUE:
                            cnf.append(clause)
                return cnf

    def simplifyCNF(cnf):
        tuples = { tuple(c) for c in cnf }
        clauses = [ set(c) for c in tuples ]
        cnf = []
        for c in clauses:
            if any(c >= c1 for c1 in clauses if c != c1):
                continue
            cnf.append(sorted(list(c)))
        return sorted(cnf)

    cnf = _toCNF(toNNF(f))
    if simplify:
        cnf = simplifyCNF(cnf)
    if formula:
        cnf = ["and", *[ ["or",*clause] for clause in cnf ]]
    return cnf

def toDNF(f, simplify=True, formula=False):
    """与えられた命題論理式 f を論理和標準形(DNF)に変換する．

    DNF式は複数の連言節の論理和であり，各連言節は複数のリテラルの論理積である．
    いったん ["not",f] を :obj:`toCNF` 関数でCNF式に変換したのち，
    CNF式中の各リテラルを否定することでDNF式を求めている．

    simplify=True の場合，簡単化したDNF式を返す．

    formula=True を指定すれば，命題論理式を返す．

    f に含まれる命題変数の個数を :math:`n` とすると最悪の場合DNF式のサイズは :math:`2^n` になる

    Args:
        f: 命題論理式．命題論理式の記法については :obj:`.functions` 参照．
        simplify (bool, optional): Falseなら得られたDNF式を簡単化しない(デフォルト値はTrue)．
        formula (bool, optional): 連言節のリストでなく，命題論理式を返す．

    Returns:
        DNF式．DNF式は連言節のリストであり，各連言節はリテラルのリストである．

    Raises:
        CspsatException: 論理式の構文エラー．

    Examples:
        >>> p = Bool("p")
        >>> toDNF(["not",["xor",p(1),p(2)]])
        [[p(1), p(2)], [~p(1), ~p(2)]]
        >>> toDNF(["not",["xor",p(1),p(2)]], formula=True)
        ['or', ['and', p(1), p(2)], ['and', ~p(1), ~p(2)]]
    """
    dnf = [ [ ~lit for lit in clause ] for clause in toCNF(["not",f], simplify) ]
    if formula:
        dnf = ["or", *[ ["and",*clause] for clause in dnf ]]
    return dnf

def ge1(xx):
    """at-least-one基数制約の節をyieldするジェネレータ関数．

    xx = [x1, x2, ..., xn] のとき，節 {x1, x2, ..., xn} をyieldする．

    Args:
        xx (list): リテラル(:obj:`.Bool` オブジェクト)リスト．

    Yields:
        基数制約を符号化した節．

    Examples:
        >>> p = Bool("p")
        >>> for clause in ge1([p(1), p(2), p(3)]): print(clause)
        [p(1), p(2), p(3)]
    """
    yield xx

def le1(xx):
    """at-most-one基数制約の節をyieldするジェネレータ関数．

    xx = [x1, x2, ..., xn] のとき，すべての xi, xj の組合せに対し節 {xi, xj} をyieldする．
    この方法はペアワイズ法と呼ばれる．
    一般には :obj:`.Encoder` クラスに実装されている逐次カウンタ法のほうが良い．

    Args:
        xx (list): リテラル(:obj:`.Bool` オブジェクト)リスト．

    Yields:
        基数制約を符号化した節．

    Examples:
        >>> p = Bool("p")
        >>> for clause in le1([p(1), p(2), p(3)]): print(clause)
        [~p(1), ~p(2)]
        [~p(1), ~p(3)]
        [~p(2), ~p(3)]
    """
    for i in range(len(xx)):
        for j in range(i+1, len(xx)):
            yield [~xx[i], ~xx[j]]

def eq1(xx):
    """exact-one基数制約の節をyieldするジェネレータ関数．

    :obj:`ge1` 関数，:obj:`le1` 関数を用いて必要な節をyieldする．
    一般には :obj:`.Encoder` クラスに実装されている逐次カウンタ法のほうが良い．

    Args:
        xx (list): リテラル(:obj:`.Bool` オブジェクト)リスト．

    Yields:
        基数制約を符号化した節．

    Examples:
        >>> p = Bool("p")
        >>> for clause in eq1([p(1), p(2), p(3)]): print(clause)
        [p(1), p(2), p(3)]
        [~p(1), ~p(2)]
        [~p(1), ~p(3)]
        [~p(2), ~p(3)]
    """
    yield from ge1(xx)
    yield from le1(xx)

def geK(xx, k):
    """at-least-k基数制約の節をyieldするジェネレータ関数．

    xx = [x1, x2, ..., xn] のとき，xi から n-k+1 個のリテラルを選択するすべての組合せに対し，選んだリテラルからなる節をyieldする．
    すなわち n-k+1 個のリテラルがすべて偽になる場合を排除している．
    この方法はバイノミナル法と呼ばれる．
    一般には :obj:`.Encoder` クラスに実装されている逐次カウンタ法のほうが良い．

    Args:
        xx (list): リテラル(:obj:`.Bool` オブジェクト)リスト．
        k (int): kの値．

    Yields:
        基数制約を符号化した節．

    Examples:
        >>> p = Bool("p")
        >>> for clause in geK([p(1), p(2), p(3)], 2): print(clause)
        [p(1), p(2)]
        [p(1), p(3)]
        [p(2), p(3)]
    """
    for yy in itertools.combinations(xx, len(xx)-k+1):
        yield list(yy)

def leK(xx, k):
    """at-most-k基数制約の節をyieldするジェネレータ関数．

    xx = [x1, x2, ..., xn] のとき，xi から k+1 個のリテラルを選択するすべての組合せに対し，選んだリテラルの否定からなる節をyieldする．
    すなわち k+1 個のリテラルがすべて真になる場合を排除している．
    この方法はバイノミナル法と呼ばれる．
    一般には :obj:`.Encoder` クラスに実装されている逐次カウンタ法のほうが良い．

    Args:
        xx (list): リテラル(:obj:`.Bool` オブジェクト)リスト．
        k (int): kの値．

    Yields:
        基数制約を符号化した節．

    Examples:
        >>> p = Bool("p")
        >>> for clause in leK([p(1), p(2), p(3)], 2): print(clause)
        [~p(1), ~p(2), ~p(3)]

    """
    for yy in itertools.combinations(xx, k+1):
        yield [ ~x for x in yy ]

def eqK(xx, k):
    """exact-k基数制約の節をyieldするジェネレータ関数．

    :obj:`geK` 関数，:obj:`leK` 関数を用いて必要な節をyieldする．
    一般には :obj:`.Encoder` クラスに実装されている逐次カウンタ法のほうが良い．

    Args:
        xx (list): リテラル(:obj:`.Bool` オブジェクト)リスト．
        k (int): kの値．

    Yields:
        基数制約を符号化した節．

    Examples:
        >>> p = Bool("p")
        >>> for clause in eqK([p(1), p(2), p(3)], 2): print(clause)
        [p(1), p(2)]
        [p(1), p(3)]
        [p(2), p(3)]
        [~p(1), ~p(2), ~p(3)]
    """
    yield from geK(xx, k)
    yield from leK(xx, k)

statData = None
"""ソルバー実行の統計データを保存するグローバル変数
"""

def status(lastOnly=True):
    """SATソルバー (:obj:`.SAT`)またはCSPソルバー (:obj:`.Solver`)実行の統計データ．

    :obj:`.SAT.getStat`, :obj:`.Solver.getStat` 参照．

    Returns:
        ソルバー実行の統計データ．
    """
    if not lastOnly:
        return statData
    infos = statData["sat"]
    info = infos[-1] if len(infos) > 0 else {}
    return { **statData, "sat":info }

from contextlib import contextmanager

defaultTimeout = 600
"""タイムアウト秒数のデフォルト値を保存するグローバル変数
"""

def getTimeout():
    """タイムアウト秒数のデフォルト値を返す．

    Returns:
        タイムアウト秒数のデフォルト値．

    >>> getTimeout()
    600
    """
    return defaultTimeout

def setTimeout(timeout):
    """タイムアウト秒数のデフォルト値を設定する．

    Args:
        timeout (int): タイムアウト秒数．

    >>> setTimeout(100)
    >>> getTimeout()
    100
    """
    global defaultTimeout
    defaultTimeout = timeout

@contextmanager
def _cspsatTimer(timeout, verbose=0):
    """timeout秒を超えると，メインのスレッドにSIGINTのシグナルを送り，:obj:`.CspsatTimeout` 例外をraiseする．

    Args:
        timeout (int): タイムアウト秒．
    """
    timer = threading.Timer(timeout, _thread.interrupt_main)
    timer.start()
    if verbose >= 1:
        print(f"# タイムアウト{timeout}秒でプログラム開始", file=sys.stderr)
    try:
        yield
    except KeyboardInterrupt as e:
        if timer.is_alive():
            raise
        raise CspsatTimeout(f"プログラムの実行時間が{timeout}秒を超えた") from e
    finally:
        timer.cancel()

def solutionsSAT(cnf, command=None, num=1, positiveOnly=False, includeAux=False, verbose=0, timeout=None, tempdir=None):
    """与えられたCNF式のモデルをSATソルバーで探索しyieldするジェネレータ関数．

    :obj:`.SAT.solutions` 参照．

    Args:
        cnf (list of list of Bool): CNF式 (節のリスト)．
        command (str, optional): 利用するSATソルバーのコマンド．
            :obj:`.SAT` のcommand参照．
        num (int, optional): 探索するモデルの最大個数．0なら全解を探索する (デフォルト値は1)．
            :obj:`.SAT.solutions` のnum参照．
        positiveOnly (bool, optional): Trueならモデルに正リテラルのみを含める (デフォルト値はFalse)．
            :obj:`.SAT` のpositveOnly参照．
        includeAux (bool, optional): Trueならモデルに補助変数を含める (デフォルト値はFalse)．
            :obj:`.SAT` のincludeAux参照．
        verbose (int, optional): 負なら結果を出力しない．正ならSATソルバーの情報を表示する (デフォルト値は0)．
            :obj:`.SAT` のverobse参照．
        timeout (int, optional): この関数の実行時間のタイムアウト秒数を指定する (デフォルト値は600秒)．
            :obj:`setTimeout` でデフォルト値を変更できる．
        tempdir (str, optional): 一時ファイルのディレクトリ名 (デフォルト値はNone)．

    Yields:
        CNF式のモデル．命題変数(:obj:`.Bool` オブジェクト)をキーとする辞書(dict)で，値は0か1．

    Examples:
        >>> p = Bool("p")
        >>> cnf = [ [p(1), p(2)] ]
        >>> for sol in solutionsSAT(cnf, num=0): print(sol)
        {p(1): 0, p(2): 1}
        {p(1): 1, p(2): 0}
        {p(1): 1, p(2): 1}
        >>> for sol in solutionsSAT(cnf, num=0): print(f"p(1)={sol[p(1)]}, p(2)={sol[p(2)]}")
        p(1)=0, p(2)=1
        p(1)=1, p(2)=0
        p(1)=1, p(2)=1
        >>> for sol in solutionsSAT(cnf, num=0): print((sol, status()))
        ({p(1): 0, p(2): 1}, {'variables': 2, 'clauses': 1, 'conflicts': 0, 'decisions': 1, 'propagations': 2, 'solving': 0.709282636642456})
        ({p(1): 1, p(2): 0}, {'variables': 2, 'clauses': 2, 'conflicts': 1, 'decisions': 2, 'propagations': 3, 'solving': 0.30728840827941895})
        ({p(1): 1, p(2): 1}, {'variables': 2, 'clauses': 3, 'conflicts': 1, 'decisions': 1, 'propagations': 3, 'solving': 0.33997058868408203})
    """
    global statData
    timeout = timeout or getTimeout()
    sat = SAT(command=command, positiveOnly=positiveOnly, includeAux=includeAux, verbose=verbose, tempdir=tempdir)
    try:
        with _cspsatTimer(timeout, verbose):
            sat.add(*cnf)
            for sol in sat.solutions(num=num):
                statData = sat.stats
                yield sol
    except CspsatTimeout as e:
        sat.stats["result"] = "TIMEOUT"
        raise e
    finally:
        statData = sat.stats
        sat.__del__()

def solveSAT(cnf, command=None, num=1, positiveOnly=False, includeAux=False, verbose=0, timeout=None, stat=False, tempdir=None):
    """与えられたCNF式のモデルをSATソルバーで探索し出力する．

    :obj:`.SAT.solve` 参照．

    Args:
        cnf (list of list of Bool): CNF式 (節のリスト)．
        command (str, optional): 利用するSATソルバーのコマンド．
            :obj:`.SAT` のcommand参照．
        num (int, optional): 探索するモデルの最大個数．0なら全解を探索する (デフォルト値は1)．
            :obj:`.SAT.solve` のnum参照．
        positiveOnly (bool, optional): Trueならモデルに正リテラルのみを含める (デフォルト値はFalse)．
            :obj:`.SAT` のpositveOnly参照．
        includeAux (bool, optional): Trueならモデルに補助変数を含める (デフォルト値はFalse)．
            :obj:`.SAT` のincludeAux参照．
        verbose (int, optional): 負なら結果を出力しない．正ならSATソルバーの情報を表示する (デフォルト値は0)．
            :obj:`.SAT` のverobse参照．
        timeout (int, optional): この関数の実行時間のタイムアウト秒数を指定する (デフォルト値は600秒)．
            :obj:`setTimeout` でデフォルト値を変更できる．
        stat (bool, optional): Trueなら統計データも表示する (デフォルト値はFalse)．
            :obj:`.SAT.getStat` 参照．
        tempdir (str, optional): 一時ファイルのディレクトリ名 (デフォルト値はNone)．

    Examples:
        >>> p = Bool("p")
        >>> cnf = [ [p(1), p(2)] ]
        >>> solveSAT(cnf, num=0)
        SATISFIABLE
        Model 1: {p(1): 0, p(2): 1}
        Model 2: {p(1): 1, p(2): 0}
        Model 3: {p(1): 1, p(2): 1}
        >>> solveSAT(cnf, num=0, stat=True)
        SATISFIABLE
        Model 1: {p(1): 0, p(2): 1}
        Stat: {'variables': 2, 'clauses': 1, 'conflicts': 0, 'decisions': 1, 'propagations': 2, 'solving': 0.30565834045410156}
        Model 2: {p(1): 1, p(2): 0}
        Stat: {'variables': 2, 'clauses': 2, 'conflicts': 1, 'decisions': 2, 'propagations': 3, 'solving': 0.34708118438720703}
        Model 3: {p(1): 1, p(2): 1}
        Stat: {'variables': 2, 'clauses': 3, 'conflicts': 1, 'decisions': 1, 'propagations': 3, 'solving': 0.3393073081970215}
        Stat: {'variables': 2, 'clauses': 4, 'conflicts': 2, 'decisions': 1, 'propagations': 2, 'solving': 0.303924560546875}
    """
    global statData
    timeout = timeout or getTimeout()
    sat = SAT(command=command, positiveOnly=positiveOnly, includeAux=includeAux, verbose=verbose, tempdir=tempdir)
    try:
        with _cspsatTimer(timeout, verbose):
            sat.add(*cnf)
            sat.solve(num=num, stat=stat)
    except CspsatTimeout as e:
        sat.stats["result"] = "TIMEOUT"
        raise e
    finally:
        statData = sat.stats
        sat.__del__()

def solutionsCSP(csp, encoder=None, command=None, num=1, positiveOnly=False, includeAux=False, verbose=0, timeout=None, tempdir=None):
    """与えられたCSP(制約充足問題)をCNF式にSAT符号化し，得られたCNF式の解(モデル)をSATソルバーで探索し，
    CSPの解(モデル)に変換してyieldするジェネレータ関数．

    :obj:`.Solver.solutions` 参照．

    Args:
        csp: CSP．制約のシーケンス．制約の記法については :obj:`cspsat.csp` 参照．
        encoder (Encoder | str, optional): 使用するSAT符号化．Encoderのインスタンスならそれを用いる．
            文字列の場合，"d"から始まるなら :obj:`.DirectEncoder`, "l"から始まるなら :obj:`.LogEncoder`, それ以外なら :obj:`.OrderEncoder` を用いる．
        command (str, optional): 利用するSATソルバーのコマンド．
            :obj:`.SAT` のcommand参照．
        num (int, optional): 探索するモデルの最大個数．0なら全解を探索する (デフォルト値は1)．
            :obj:`.Solver.solutions` のnum参照．
        positiveOnly (bool, optional): Trueならモデルに正リテラルのみを含める (デフォルト値はFalse)．
            :obj:`.Solver` のpositveOnly参照．
        includeAux (bool, optional): Trueならモデルに補助変数を含める (デフォルト値はFalse)．
            :obj:`.Solver` のincludeAux参照．
        verbose (int, optional): 負なら結果を出力しない．正ならSATソルバーの情報を表示する (デフォルト値は0)．
            :obj:`.Solver` のverbose参照．
        timeout (int, optional): この関数の実行時間のタイムアウト秒数を指定する (デフォルト値は600秒)．
            :obj:`setTimeout` でデフォルト値を変更できる．
        tempdir (str, optional): 一時ファイルのディレクトリ名 (デフォルト値はNone)．

    Yields:
        CSPのモデル．命題変数(:obj:`.Bool` オブジェクト)あるいは整数変数(:obj:`.Var` オブジェクト)をキーとする辞書(dict)で，値は整数．

    Examples:
        >>> x = Var("x")
        >>> csp = [ ["int",x(1),1,3], ["int",x(2),1,3], [">",x(1),x(2)] ]
        >>> for sol in solutionsCSP(csp, num=0): print(sol)
        {x(1): 2, x(2): 1}
        {x(1): 3, x(2): 1}
        {x(1): 3, x(2): 2}
    """
    global statData
    if isinstance(encoder, Encoder):
        pass
    elif isinstance(encoder, str) and encoder.lower().startswith("d"):
        encoder = DirectEncoder(verbose=verbose)
    elif isinstance(encoder, str) and encoder.lower().startswith("l"):
        encoder = LogEncoder(verbose=verbose)
    else:
        encoder = OrderEncoder(verbose=verbose)
    encoder.verbose = verbose
    if verbose >= 1:
        print(f"# Encoderは{encoder.__class__.__name__}", file=sys.stderr)
    sat = SAT(command=command, verbose=verbose, tempdir=tempdir)
    solver = Solver(encoder, sat=sat, positiveOnly=positiveOnly, includeAux=includeAux, verbose=verbose)
    try:
        timeout = timeout or getTimeout()
        with _cspsatTimer(timeout, verbose):
            for sol in solver.solutions(csp, num=num):
                statData = solver.stats
                yield sol
    except CspsatTimeout as e:
        solver.stats["result"] = "TIMEOUT"
        raise e
    finally:
        statData = solver.stats
        sat.__del__()

def solveCSP(csp, encoder=None, command=None, num=1, positiveOnly=False, includeAux=False, verbose=0, timeout=None, stat=False, tempdir=None):
    """与えられたCSP(制約充足問題)をCNF式にSAT符号化し，得られたCNF式の解(モデル)をSATソルバーで探索し，
    CSPの解(モデル)に変換して出力する．

    :obj:`.Solver.solve` 参照．

    Args:
        csp: CSP．制約のシーケンス．制約の記法については :obj:`cspsat.csp` 参照．
        encoder (Encoder | str, optional): 使用するSAT符号化．Encoderのインスタンスならそれを用いる．
            文字列の場合，"d"から始まるなら :obj:`.DirectEncoder`, "l"から始まるなら :obj:`.LogEncoder`, それ以外なら :obj:`.OrderEncoder` を用いる．
        command (str, optional): 利用するSATソルバーのコマンド．
            :obj:`.SAT` のcommand参照．
        num (int, optional): 探索するモデルの最大個数．0なら全解を探索する (デフォルト値は1)．
            :obj:`.Solver.solutions` のnum参照．
        positiveOnly (bool, optional): Trueならモデルに正リテラルのみを含める (デフォルト値はFalse)．
            :obj:`.Solver` のpositveOnly参照．
        includeAux (bool, optional): Trueならモデルに補助変数を含める (デフォルト値はFalse)．
            :obj:`.Solver` のincludeAux参照．
        verbose (int, optional): 負なら結果を出力しない．正ならSATソルバーの情報を表示する (デフォルト値は0)．
            :obj:`.Solver` のverbose参照．
        timeout (int, optional): この関数の実行時間のタイムアウト秒数を指定する (デフォルト値は600秒)．
            :obj:`setTimeout` でデフォルト値を変更できる．
        stat (bool, optional): Trueなら統計データも表示する (デフォルト値はFalse)．
            :obj:`.Solver.getStat` 参照．
        tempdir (str, optional): 一時ファイルのディレクトリ名 (デフォルト値はNone)．

    Examples:
        >>> x = Var("x")
        >>> csp = [ ["int",x(1),1,3], ["int",x(2),1,3], [">",x(1),x(2)] ]
        >>> solveCSP(csp, num=0)
        SATISFIABLE
        Model 1: {x(1): 2, x(2): 1}
        Model 2: {x(1): 3, x(2): 1}
        Model 3: {x(1): 3, x(2): 2}
        >>> solveCSP(csp, num=0, stat=True) # 順序符号化 (OrderEncoder)
        SATISFIABLE
        Model 1: {x(1): 2, x(2): 1}
        Stat: {'variables': 4, 'clauses': 5, 'conflicts': 0, 'decisions': 2, 'propagations': 4, 'solving': 0.3538937568664551, 'encoding': 0.0009377002716064453}
        Model 2: {x(1): 3, x(2): 1}
        Stat: {'variables': 4, 'clauses': 6, 'conflicts': 0, 'decisions': 1, 'propagations': 4, 'solving': 0.36089038848876953, 'encoding': 0.0009377002716064453}
        Model 3: {x(1): 3, x(2): 2}
        Stat: {'variables': 4, 'clauses': 7, 'conflicts': 1, 'decisions': 1, 'propagations': 5, 'solving': 0.39816951751708984, 'encoding': 0.0009377002716064453}
        Stat: {'variables': 4, 'clauses': 7, 'conflicts': 1, 'decisions': 1, 'propagations': 5, 'solving': 0.39816951751708984, 'encoding': 0.0009377002716064453}
        >>> solveCSP(csp, num=0, stat=True, encoder="direct") # 直接符号化 (DirectEncoder)
        SATISFIABLE
        Model 1: {x(1): 2, x(2): 1}
        Stat: {'variables': 6, 'clauses': 11, 'conflicts': 0, 'decisions': 3, 'propagations': 6, 'solving': 0.40071797370910645, 'encoding': 0.0010752677917480469}
        Model 2: {x(1): 3, x(2): 1}
        Stat: {'variables': 6, 'clauses': 12, 'conflicts': 0, 'decisions': 2, 'propagations': 6, 'solving': 0.39261484146118164, 'encoding': 0.0010752677917480469}
        Model 3: {x(1): 3, x(2): 2}
        Stat: {'variables': 6, 'clauses': 13, 'conflicts': 1, 'decisions': 2, 'propagations': 10, 'solving': 0.38961291313171387, 'encoding': 0.0010752677917480469}
        Stat: {'variables': 6, 'clauses': 13, 'conflicts': 1, 'decisions': 2, 'propagations': 10, 'solving': 0.38961291313171387, 'encoding': 0.0010752677917480469}
    """
    global statData
    if isinstance(encoder, Encoder):
        pass
    elif isinstance(encoder, str) and encoder.lower().startswith("d"):
        encoder = DirectEncoder(verbose=verbose)
    elif isinstance(encoder, str) and encoder.lower().startswith("l"):
        encoder = LogEncoder(verbose=verbose)
    else:
        encoder = OrderEncoder(verbose=verbose)
    encoder.verbose = verbose
    if verbose >= 1:
        print(f"# Encoderは{encoder.__class__.__name__}", file=sys.stderr)
    sat = SAT(command=command, verbose=verbose, tempdir=tempdir)
    solver = Solver(encoder, sat=sat, positiveOnly=positiveOnly, includeAux=includeAux, verbose=verbose)
    try:
        timeout = timeout or getTimeout()
        with _cspsatTimer(timeout, verbose):
            solver.solve(csp, num=num, stat=stat)
            statData = solver.stats
    except CspsatTimeout as e:
        solver.stats["result"] = "TIMEOUT"
        raise e
    finally:
        statData = solver.stats
        sat.__del__()

def saveSAT(cnf, fileName):
    """CNF式をファイルに保存する．

    CNF式は節のシーケンス(リストなど)であり，各節はリテラル(:obj:`.Bool` オブジェクト)のリストである．
    節が文字列の場合はコメントとして扱われ，そのままファイルに書き込まれる
    (文字列の先頭が "#" でない場合は先頭に "# " を追加する)．
    保存したCNF式はloadSAT関数でロードできる．

    Args:
        cnf: CNF式．リテラルのリストのシーケンス．
        fileName (str): ファイル名．

    Examples:
        >>> p = Bool("p")
        >>> cnf = [ [ p(1), p(2) ]]
        >>> saveSAT(cnf, "/tmp/foo.sat")
        >>> cnf = loadSAT("/tmp/foo.sat")
        >>> solveSAT(cnf)
        SATISFIABLE
        Model 1: {p(1): 0, p(2): 1}
    """
    with open(fileName, mode="w", encoding="utf-8") as file:
        for clause in cnf:
            if isinstance(clause, list):
                file.write(" ".join(map(str, clause)))
            else:
                comment = str(clause)
                if not comment.startswith("#"):
                    file.write("# ")
                file.write(comment)
            file.write("\n")

def loadSAT(fileName):
    """ファイルからCNF式を読み込み，各節をyieldするジェネレータ関数．

    各行が節を表すが，空行および ``#`` から始まる行はコメントとして処理される．
    各リテラルは1つ以上の空白文字で区切られているとし，負リテラルは ``~`` から始まるとする．
    命題変数名には ``#`` あるいは ``~`` から始まらない任意の文字列を使用できる．

    Args:
        fileName (str): ファイル名．

    Yields:
        読み込んだCNF式．
    """
    with open(fileName, encoding="utf-8") as file:
        for line in file:
            line = re.sub(r"\s+", " ", line).strip()
            if line == "":
                continue
            if line.startswith("#"):
                yield line
                continue
            clause = []
            for s in line.split(" "):
                if s.startswith("~"):
                    clause.append(Bool(s[1:], False))
                else:
                    clause.append(Bool(s))
            yield clause

def saveDimacs(cnf, fileName):
    """CNF式をDIMACS CNF形式に変換しファイルに保存する．

    Args:
        cnf: CNF式．リテラルのリストのシーケンス．
        fileName (str): ファイル名．

    Examples:
        >>> p = Bool("p")
        >>> cnf = [ [ p(1), p(2) ], [ ~p(1), ~p(2) ] ]
        >>> saveDimacs(cnf, "/tmp/foo.cnf")
    """
    sat = SAT(cnfFile=fileName, delete=False)
    sat.add(*cnf)
    sat.updateDimacsHeader()
    sat.cnf.close()
    sat.__del__()

def saveCSP(csp, fileName):
    """CSPをJSON形式でファイルに保存する．
    Boolオブジェクトは先頭に "$" を付け加えた文字列に変換する．
    Varオブジェクトは先頭に "@" を付け加えた文字列に変換する．

    Args:
        cnf: CSP．
        fileName (str): ファイル名．
    """
    def serialize(data):
        if isinstance(data, Bool):
            n = data.name
            return "$" + n if data.positive else "$~" + n
        if isinstance(data, Var):
            return "@" + data.name
        if isinstance(data, (int,float,bool,str)):
            return data
        try:
            return [ serialize(d) for d in iter(data) ]
        except TypeError as e:
            raise CspsatException(f"制約の構文エラー: {data}") from e
    with open(fileName, mode="w", encoding="utf-8") as file:
        file.write("[\n")
        for c in csp:
            s = json.JSONEncoder().encode(serialize(c))
            file.write(s + ",\n")
        file.write(json.JSONEncoder().encode(["#", "END of CSP"]) + "\n")
        file.write("]\n")

def loadCSP(fileName):
    """JSON形式でCSPが保存されたファイルからCSPを読み込みyieldするジェネレータ関数．

    先頭に "$" がある文字列はBoolオブジェクトに変換する．
    先頭に "@" がある文字列はVarオブジェクトに変換する．

    Args:
        fileName (str): ファイル名．

    Yield:
        読み込んだCSPのデータ．
    """
    def deserialize(data):
        if isinstance(data, (int,float,bool)):
            return data
        if isinstance(data, str):
            if data.startswith("$"):
                return Bool(data[1:]) if data[1] != "~" else ~Bool(data[2:])
            if data.startswith("@"):
                return Var(data[1:])
            return data
        return [ deserialize(d) for d in iter(data) ]
    with open(fileName, encoding="utf-8") as file:
        for c in deserialize(json.load(file)):
            yield c

import sys
import re
import itertools
import json
import threading
import _thread
from .sat import SAT
from .csp import Solver, Encoder, DirectEncoder, OrderEncoder, LogEncoder
from .util import CspsatException, CspsatTimeout, Bool, TRUE, FALSE, Var
