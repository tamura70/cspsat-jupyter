"""SATソルバーのモジュール．

以下のクラスからなる．

* SATクラス: SATソルバーのラッパークラス

Note:
    本プログラムは学習用の目的で作成されている．
    実用上の問題への適用は想定していない．
    Copyright (c) 2025-- Naoyuki Tamura
    Licensed under the MIT License
"""

import os

class SAT():
    """SATソルバーのラッパークラス．

    実際のモデル探索には，指定された外部のSATソルバーを使用する．
    SATソルバーとしては，DIMACS CNFを入力形式とし，SAT solver competitionの出力形式にしたがって結果を標準出力に出力するものなら任意のプログラムを利用できる．

    Args:
        command (str, optional): 利用するSATソルバーのコマンド．指定しなければ :obj:`defaultCommand` を使用する (デフォルト値はWindows系なら"sat4j.bat"，その他は"./sat4j")．
        tmpdir (str, optional): 一時ファイルのディレクトリ名を指定する．指定しなければシステム設定にしたがう．
        cnfFile (str, optional): DIMACS CNFファイル名を指定する．指定しなければ一時ファイルが作成される．
        outFile (str, optional): SATソルバーの出力ファイル名を指定する．指定しなければ一時ファイルが作成される．
        delete (bool, optional): TrueならSATオブジェクトが削除される際にcnfFile, outFileを削除する (デフォルト値はTrue)．
        maxClauses (int, optional): 節数がこの値を超えたら例外を発生する．指定しなければ :obj:`defaultMaxClauses` を使用する (デフォルト値は10000000)．
        limit (int, optional): 実行時間の制限秒数を指定する．指定しなければ :obj:`defaultLimit` を使用する (デフォルト値は3600秒)．
        positiveOnly (bool, optional): Trueならモデルに正リテラルのみを含める (デフォルト値はFalse)．
        includeAux (bool, optional): Trueならモデルに補助変数を含める (デフォルト値はFalse)．
        verbose (int, optional): 正ならSATソルバーの情報を表示する (デフォルト値は0)．

    Attributes:
        variables (list): 変数名のリスト．DIMACS CNFファイル中では，このリスト順に変数番号が付けられている．
        nclauses (int): 節の個数．
        model (dict): 最後にSATソルバーで求めたモデル (負リテラルも含む)．
        stats (dict): SATソルバー実行の統計データ．
        commitPosition (dict): :obj:`commit` 関数で保存されたDIMACS CNFファイルの状態(変数数，節数，ファイルサイズ)．
        buffer (bytearray): DIMACS CNFファイル書き込み用バッファー．
        bufLimit (int): DIMACS CNFファイル書き込み用バッファーの最大サイズ (65000バイト)．

    Examples:
        >>> sat = SAT() # "./sat4j" (Windowsなら"sat4j.bat")をSATソルバーとして利用する
        >>> p = Bool("p") # 命題変数 p
        >>> sat.add([ p(1), p(2) ]) # 節 {p(1),p(2)} の追加
        >>> sat.solve(num=0) # すべての解 (モデル)を表示する
        SATISFIABLE
        Model 1: {p(1): 0, p(2): 1}
        Model 2: {p(1): 1, p(2): 0}
        Model 3: {p(1): 1, p(2): 1}
    """

    defaultCommand = "sat4j.bat" if os.name == "nt" else "./sat4j"
    """デフォルトのSATソルバーコマンド (Windows系なら"sat4j.bat"，その他は"./sat4j")．

    Examples:
        >>> SAT.defaultCommand
        './sat4j'
        >>> SAT.defaultCommand = "bin/kissat" # Linuxの場合
        >>> SAT.defaultCommand = "wsl bin/kissat" # WindowsでWSLがインストールされている場合
    """

    defaultTempdir = None
    """一時ファイルのディレクトリ名のデフォルト値．Noneならシステム設定にしたがう．
    """

    defaultLimit = 3600
    """実行時間の制限秒数のデフォルト値 (3600秒)．

    Examples:
        >>> SAT.defaultLimit
        3600
        >>> SAT.defaultLimit = 100
    """

    defaultMaxClauses = 10000000
    """最大節数のデフォルト値 (10000000節)．

    Examples:
        >>> SAT.defaultMaxClauses
        10000000
        >>> SAT.defaultMaxClauses = 1000000
    """

    @classmethod
    def _tempfileName(cls, suffix=None, tempdir=None):
        if tempdir is None:
            tempdir = cls.defaultTempdir
        if tempdir == "" or tempdir == ".":
            path = f"tmp_{time.time()}{suffix}"
        else:
            (fd, path) = tempfile.mkstemp(suffix=suffix, dir=tempdir, text=True)
            os.close(fd)
        return path

    def __init__(self, command=None, tempdir=None, cnfFile=None, outFile=None, delete=True, maxClauses=None, limit=None, positiveOnly=False, includeAux=False, verbose=0):
        self.startTime = time.monotonic()
        self.command = command or SAT.defaultCommand
        self.cnfFile = cnfFile or SAT._tempfileName(suffix=".cnf", tempdir=tempdir)
        self.cnf = open(self.cnfFile, mode="w+b")
        self.outFile = outFile or SAT._tempfileName(suffix=".out", tempdir=tempdir)
        self.tempfiles = []
        if delete:
            self.tempfiles.append(self.cnfFile)
            self.tempfiles.append(self.outFile)
        self.maxClauses = SAT.defaultMaxClauses if maxClauses is None else maxClauses
        self.limit = SAT.defaultLimit if limit is None else limit
        self.positiveOnly = positiveOnly
        self.includeAux = includeAux
        self.verbose = verbose or 0

        Bool._auxCount = 0
        self._varname2litnum = {}
        self.variables = []
        self.nclauses = 0
        self.model = None
        self.stats = { "result": None, "ncalls":0, "nmodels":0, "time":0, "solving":0, "command":command, "sat":[] }
        self.commitPosition = None
        self.buffer = bytearray()
        self.bufLimit = 65000
        self.updateDimacsHeader()

    def __del__(self):
        if self.cnf:
            self.cnf.close()
            self.cnf = None
        self._deleteTempfiles()

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex_value, trace):
        self.__del__()

    def _deleteTempfiles(self):
        for file in self.tempfiles:
            if os.path.exists(file):
                os.remove(file)

    def nVars(self):
        """命題変数の個数を返す．

        Returns:
            命題変数の個数．

        Examples:
            >>> p = Bool("p")
            >>> sat = SAT()
            >>> sat.add([ p(1), p(2) ])
            >>> sat.nVars()
            2
        """
        return len(self.variables)

    def nClauses(self):
        """節の個数を返す．

        Returns:
            節の個数．

        Examples:
            >>> p = Bool("p")
            >>> sat = SAT()
            >>> sat.add([ p(1), p(2) ])
            >>> sat.nClauses()
            1
        """
        return self.nclauses

    def _flush(self):
        self.cnf.seek(0, 2)
        self.cnf.write(self.buffer)
        self.cnf.flush()
        self.buffer = bytearray()

    def updateDimacsHeader(self):
        """DIMACS CNFファイルのヘッダー情報を更新する．
        """
        if self.cnf is None:
            self.cnf = open(self.cnfFile, mode="r+b")
        self._flush()
        self.cnf.seek(0)
        h = "%-63s\n" % f"p cnf {self.nVars()} {self.nClauses()}"
        self.cnf.write(bytes(h, "latin-1"))
        self.cnf.seek(0, 2)

    def __toLitNum(self, literal):
        if not isinstance(literal, Bool):
            raise CspsatException(f"引数literalがBoolでない: {literal}")
        name = literal.name
        litNum = self._varname2litnum.get(name, 0)
        if litNum == 0:
            litNum = self.nVars() + 1
            self._varname2litnum[name] = litNum
            self.variables.append(abs(literal))
        if not literal.positive:
            litNum = -litNum
        return litNum

    def __toLitNums(self, clause):
        litNums = [ self.__toLitNum(literal) for literal in clause ]
        return litNums

    def _addClause(self, clause):
        """節を追加する．

        Boolオブジェクトのリストで表現された節を追加する．
        真の命題定数 ``TRUE`` を含む節は追加しない．偽の命題定数 ``FALSE`` は節から削除される．
        clauseが文字列の場合は，コメントとみなし何もしない．

        Args:
            clause (list of Bool): Boolオブジェクトのリスト．文字列ならコメントを表す．

        Raises:
            CspsatException: clause中のリテラルがBoolオブジェクトでない．
            CspsatException: 節数がmaxClausesを超えた．

        Examples:
            >>> p = Bool("p")
            >>> sat = SAT()
            >>> sat._addClause([ p(1), p(2) ])
        """
        if isinstance(clause, str):
            return
        if TRUE in clause:
            return
        clause = [ literal for literal in clause if literal != FALSE ]
        litNums = self.__toLitNums(clause)
        s = " ".join(map(str, litNums)) + " 0\n"
        if self.cnf is None:
            self.cnf = open(self.cnfFile, mode="r+b")
        self.buffer.extend(bytes(s, "latin-1"))
        if len(self.buffer) >= self.bufLimit:
            self._flush()
        self.nclauses += 1
        if self.verbose >= 1 and self.nclauses % 100000 == 0:
            print(f"# {self.nclauses}節を追加", file=sys.stderr)
        if self.nclauses > self.maxClauses:
            raise CspsatException(f"{self.maxClauses}より多い節が追加された ({self.nclauses} clauses)")

    def add(self, *clauses):
        """複数の節を追加する．

        Boolオブジェクトのリストで表現された節を追加する．
        命題定数 :obj:`.TRUE` を含む節は追加しない．命題定数 :obj:`.FALSE` は節から削除される．
        節が文字列の場合は，コメントとみなし何もしない．

        Args:
            clauses (list of list of Bool): 節のリスト．

        Raises:
            CspsatException: 節中のリテラルがBoolオブジェクトでない．
            CspsatException: 節数が合計でmaxClausesを超えた．

        Examples:
            >>> p = Bool("p")
            >>> sat = SAT()
            >>> sat.add([p(0)] , [p(1)]) # 2つの節 {p(0)}, {p(1)} を追加する
            >>> sat.nClauses()
            2
        """
        for clause in clauses:
            self._addClause(clause)

    def _run(self):
        """SATソルバーを実行する
        """
        if isinstance(self.command, str):
            cmd = [*shlex.split(self.command), self.cnfFile]
        else:
            cmd = [*self.command, self.cnfFile]
        out = open(self.outFile, mode="w")
        procData = { "proc": None, "killed": False }
        def kill():
            procData["proc"].kill()
            procData["killed"] = True
        timer = threading.Timer(self.limit, kill)
        time0 = time.monotonic()
        try:
            timer.start()
            if self.verbose >= 1:
                print(f"# SATソルバー開始: {cmd} ({time.monotonic()-time0:.2f}秒)", file=sys.stderr)
            proc = procData["proc"] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, text=True)
            c = 0
            while proc.poll() is None:
                if self.verbose >= 1 and c > 0 and c%10 == 0:
                    print(f"# SATソルバー動作中 ({time.monotonic()-time0:.2f}秒)", file=sys.stderr)
                time.sleep(1)
                c += 1
            proc = procData["proc"] = None
        except FileNotFoundError as e:
            raise CspsatException(f"SATソルバーの実行エラー: {cmd}") from e
        finally:
            timer.cancel()
            if procData.get("proc"):
                procData["proc"].kill()
                if self.verbose >= 1:
                    print(f"# SATソルバー強制停止 ({time.monotonic()-time0:.2f}秒)", file=sys.stderr)
            out.close()
        return not procData["killed"]

    def find(self):
        """モデルを探索する．

        指定されたSATソルバーを用いてモデルを探索し，モデルが見つかればそれを返す．見つからなければNoneを返す．
        モデルは，キーがBoolオブジェクトで値が0か1の辞書(dict)である．

        Returns:
            モデル．キーがBoolオブジェクトで値が0か1の辞書(dict)である．モデルが見つからなければNone．

        Examples:
            >>> p = Bool("p") # 命題変数 p
            >>> sat = SAT()
            >>> sat.add([p(1)], [~p(2)])
            >>> model = sat.find() # モデルを求める
            >>> model[p(1)] # 求めたモデルにおける p(1) の値
            1
            >>> model[p(2)] # 求めたモデルにおける p(2) の値
            0
        """
        t = time.monotonic()
        self.updateDimacsHeader()
        self.cnf.close()
        self.cnf = None
        result = None
        self.model = {}
        model = {}
        info = {}
        info["result"] = None
        info["variables"] = self.nVars()
        info["clauses"] = self.nClauses()
        self.stats["sat"].append(info)
        self.stats["ncalls"] += 1
        ok = self._run()
        if ok:
            with open(self.outFile, encoding="utf-8") as file:
                for line in file:
                    line = line.rstrip()
                    if self.verbose >= 2:
                        print(line, flush=True)
                    if line.startswith("s SAT"):
                        result = "SATISFIABLE"
                    elif line.startswith("s UNSAT"):
                        result = "UNSATISFIABLE"
                    elif line.startswith("s "):
                        result = "UNKNOWN"
                    elif line.startswith("v "):
                        for litNum in map(int, line[2:].split(" ")):
                            if litNum == 0:
                                continue
                            v = self.variables[abs(litNum)-1]
                            self.model[v] = 1 if litNum > 0 else 0
                            if v.isAux() and not self.includeAux:
                                continue
                            if litNum < 0 and self.positiveOnly:
                                continue
                            model[v] = 1 if litNum > 0 else 0
                    else:
                        m = re.match(r"c (conflicts|decisions|propagations)\s*:\s*(\d+)", line)
                        if m:
                            info[m.group(1)] = int(m.group(2))
        t = time.monotonic() - t
        info["solving"] = t
        info["result"] = result = result if ok else "TIMEOUT"
        self.stats["solving"] += t
        self.stats["time"] = time.monotonic() - self.startTime
        if not self.stats.get("result"):
            self.stats["result"] = result
        if not ok:
            raise CspsatTimeout(f"SATソルバーの実行時間が{self.limit}秒を超えた")
        if result == "SATISFIABLE":
            self.stats["nmodels"] += 1
        else:
            self.model = model = None
        return model

    def getStat(self, includeAll=False):
        """このソルバーによる実行の集計データを返す．

        なお，実行時間などの単位は秒で，CPU時間ではなく経過時間である．

        Args:
            includeAll (bool, optional): Falseならsatフォールドの値は最後のSATソルバー実行の集計データのみである．
                TrueならすべてのSATソルバー実行の集計データのリストになる (デフォールトはFalse)．

        Returns:
            以下からなる辞書(dict)を返す．
            * result: 実行結果 (SATISFIABLE, UNSATISFIABLE, TIMEOUT, UNKNOWN, None)．複数回のSATソルバーを呼び出しで一度でもSATISFIABLEなら，この値もSATISFIABLEとなる．
            * ncalls: SATソルバーを呼び出した回数．
            * nmodels: SATソルバーでモデルを見つけた回数．
            * time: 関数の実行時間 (秒)．
            * solving: SATソルバーの実行にかかった時間 (累積値，秒)．
            * command: SATソルバーのコマンド．
            * sat: SATソルバーの集計データ (includeAllがFalseなら最後の集計データ，Trueならすべての集計データのリスト)．

            SATソルバーの集計データ(satフィールド)は，以下からなるdictである．

            * result: SATソルバーの実行結果 (SATISFIABLE, UNSATISFIABLE, TIMEOUT, UNKNOWN, None)．
            * variables: CNF式の命題変数の個数．
            * clauses: CNF式の節の個数．
            * conflicts: SATソルバー実行での衝突の回数．
            * decisions: SATソルバー実行での決定の回数．
            * propagations: SATソルバー実行での伝播の回数．
            * solving: SATソルバー呼び出しの実行時間 (秒)．

        Examples:
            >>> p = Bool("p")
            >>> sat = SAT()
            >>> sat.add([p(1), p(2)])
            >>> sat.solve(num=0)
            SATISFIABLE
            Model 1: {p(1): 0, p(2): 1}
            Model 2: {p(1): 1, p(2): 0}
            Model 3: {p(1): 1, p(2): 1}
            >>> sat.getStat() # satフィールドの値は，最後のSATソルバー呼び出しのみ
            {'result': 'SATISFIABLE', 'ncalls': 4, 'nmodels': 3, 'time': 4.027583360671997, 'solving': 4.026149749755859, 'command': None, 'sat': {'result': 'UNSATISFIABLE', 'variables': 2, 'clauses': 4, 'conflicts': 2, 'decisions': 1, 'propagations': 2, 'solving': 1.0061917304992676}}
            >>> sat.getStat(True) # satフィールドの値は，すべてのSATソルバー呼び出しを含む
            {'result': 'SATISFIABLE', 'ncalls': 4, 'nmodels': 3, 'time': 4.027583360671997, 'solving': 4.026149749755859, 'command': None, 'sat': [{'result': 'SATISFIABLE', 'variables': 2, 'clauses': 1, 'conflicts': 0, 'decisions': 1, 'propagations': 2, 'solving': 1.0059900283813477}, {'result': 'SATISFIABLE', 'variables': 2, 'clauses': 2, 'conflicts': 1, 'decisions': 2, 'propagations': 3, 'solving': 1.005939245223999}, {'result': 'SATISFIABLE', 'variables': 2, 'clauses': 3, 'conflicts': 1, 'decisions': 1, 'propagations': 3, 'solving': 1.0080287456512451}, {'result': 'UNSATISFIABLE', 'variables': 2, 'clauses': 4, 'conflicts': 2, 'decisions': 1, 'propagations': 2, 'solving': 1.0061917304992676}]}
        """
        if includeAll:
            return self.stats
        infos = self.stats["sat"]
        info = infos[-1] if len(infos) > 0 else {}
        return { **self.stats, "sat":info }

    def addBlock(self):
        """最後に求めたモデルをブロックする節を追加する．

        self.model に保存されている最後に求めたモデルを否定する節を作成し追加する．
        これにより複数解を求めることが可能になる．
        なお，補助変数の値の違いは無視される．

        Examples:
            >>> p = Bool("p")
            >>> sat = SAT()
            >>> sat.add([ p(1), p(2) ])
            >>> sat.find()
            {p(1): 0, p(2): 1}
            >>> sat.addBlock()
            >>> sat.find()
            {p(1): 1, p(2): 0}
            >>> sat.addBlock()
            >>> sat.find()
            {p(1): 1, p(2): 1}
            >>> sat.addBlock()
            >>> sat.find() # Noneが返る
        """
        if self.model is not None:
            clause = [ (~v if self.model.get(v) else v) for v in self.model if not v.isAux() ]
            self.add(clause)

    def commit(self):
        """現在のDIMACS CNFファイルの状態を返す．

        現在のDIMACS CNFファイルの状態 (変数数，節数，ファイルサイズ)を :obj:`commitPosition` に保存し，それを返す．
        そのデータを cancel 関数に渡せば，DIMACS CNFファイルをその状態に戻すことができる．

        Returns:
            現在のDIMACS CNFファイルの状態 (変数数，節数，ファイルサイズ)．

        Examples:
            >>> p = Bool("p")
            >>> sat = SAT()
            >>> sat.add([ p(1), p(2) ])
            >>> sat.commit()
            {'variables': 2, 'clauses': 1, 'size': 70}
        """
        self.updateDimacsHeader()
        self.commitPosition = {}
        self.commitPosition["variables"] = self.nVars()
        self.commitPosition["clauses"] = self.nClauses()
        self.commitPosition["size"] = self.cnf.tell()
        return self.commitPosition

    def cancel(self, commitPosition=None):
        """DIMACS CNFファイルの状態を戻す．

        :obj:`commit` が返した状態に，DIMACS CNFファイルの状態を戻す．
        現在のDIMACS CNFファイルの状態 (変数数，節数，ファイルサイズ)を :obj:`commitPosition` に保存し，それを返す．
        そのデータを cancel 関数に渡せば，DIMACS CNFファイルをその状態に戻すことができる．

        Args:
            commitPosition (dict, optional): DIMACS CNFファイルの状態 (変数数，節数，ファイルサイズ)．
                指定されていなければ :obj:`commitPosition` に保存されている状態を使用する．

        Returns:
            使用したcommitPositionの値．

        Examples:
            >>> p = Bool("p")
            >>> sat = SAT()
            >>> sat.add([ p(1), p(2) ])
            >>> sat.commit()
            {'variables': 2, 'clauses': 1, 'size': 70}
            >>> sat.add([ ~p(1), ~p(2) ])
            >>> sat.solve(num=0)
            SATISFIABLE
            Model 1: {p(1): 0, p(2): 1}
            Model 2: {p(1): 1, p(2): 0}
            >>> sat.cancel()
            {'variables': 2, 'clauses': 1, 'size': 70}
            >>> sat.solve(num=0)
            SATISFIABLE
            Model 1: {p(1): 0, p(2): 1}
            Model 2: {p(1): 1, p(2): 0}
            Model 3: {p(1): 1, p(2): 1}
        """
        commitPosition = commitPosition or self.commitPosition
        if commitPosition is not None:
            self.updateDimacsHeader()
            self.variables = self.variables[:commitPosition["variables"]]
            nVars = len(self.variables)
            delNames = []
            for (name, litNum) in self._varname2litnum.items():
                if litNum > nVars:
                    delNames.append(name)
            for name in delNames:
                del self._varname2litnum[name]
            self.nclauses = commitPosition["clauses"]
            self.cnf.seek(commitPosition["size"])
            self.cnf.truncate()
        return commitPosition

    def solutions(self, cnf=None, num=1):
        """複数のモデルを探索し，それらをyieldするジェネレータ関数．

        指定された上限の個数まで解(モデル)を探索し，解が見つかればそれをyieldする．
        各モデルは，キーがBoolオブジェクトで値が0か1の辞書(dict)である．

        複数のモデルを探索する場合は， :obj:`addBlock` メソッドを用いて，最後に見つかったモデルを否定する節を追加したのち，SATソルバーを再度起動することで他のモデルを求めている．
        なお，補助変数の値のみが異なっている場合は同じモデルとみなす．

        Args:
            cnf (list of list of Bool, optional): CNF式 (節のリスト)．
            num (int, optional): 探索するモデルの最大個数．0なら全解を探索する (デフォルト値は1)．

        Yields:
            見つかったモデル．キーがBoolオブジェクトで値が0か1の辞書(dict)である．

        Examples:
            >>> p = Bool("p")
            >>> sat = SAT()
            >>> cnf = [ [p(1), p(2)] ]
            >>> for sol in sat.solutions(cnf, num=0): print(sol)
            {p(1): 0, p(2): 1}
            {p(1): 1, p(2): 0}
            {p(1): 1, p(2): 1}
        """
        cnf = cnf or []
        self.add(*cnf)
        count = 0
        while num == 0 or count < num:
            model = self.find()
            if model is None:
                break
            yield model
            self.addBlock()
            count += 1

    def solve(self, cnf=None, num=1, stat=False):
        """複数のモデルを探索し出力する．

        :obj:`solutions` 関数で指定された上限の個数まで解(モデル)を探索し，出力する．
        出力の最初の行で，SATISFIABLEかUNSATISFIABLEかを表示する．

        Args:
            cnf (list of list of Bool, optional): CNF式 (節のリスト)．
            num (int, optional): 探索するモデルの最大個数．0なら全解を探索する (デフォルト値は1)．
            stat (bool, optional): Trueなら統計データも表示する (デフォルト値はFalse)．

        Examples:
            >>> p = Bool("p")
            >>> sat = SAT()
            >>> cnf = [ [p(1), p(2)] ]
            >>> sat.solve(cnf, num=0, stat=True) # 最後のStatは4つ目のモデルが存在しないことの探索
            SATISFIABLE
            Model 1: {p(1): 0, p(2): 1}
            Stat: {'result': 'SATISFIABLE', 'ncalls': 1, 'nmodels': 1, 'time': 1.0037319660186768, 'solving': 1.0032446384429932, 'sat': {'result': 'SATISFIABLE', 'variables': 2, 'clauses': 1, 'conflicts': 0, 'decisions': 1, 'propagations': 2, 'solving': 1.0032446384429932}}
            Model 2: {p(1): 1, p(2): 0}
            Stat: {'result': 'SATISFIABLE', 'ncalls': 2, 'nmodels': 2, 'time': 2.0091824531555176, 'solving': 2.008450746536255, 'sat': {'result': 'SATISFIABLE', 'variables': 2, 'clauses': 2, 'conflicts': 1, 'decisions': 2, 'propagations': 3, 'solving': 1.0052061080932617}}
            Model 3: {p(1): 1, p(2): 1}
            Stat: {'result': 'SATISFIABLE', 'ncalls': 3, 'nmodels': 3, 'time': 3.012281894683838, 'solving': 3.0112452507019043, 'sat': {'result': 'SATISFIABLE', 'variables': 2, 'clauses': 3, 'conflicts': 1, 'decisions': 1, 'propagations': 3, 'solving': 1.0027945041656494}}
            Stat: {'result': 'SATISFIABLE', 'ncalls': 4, 'nmodels': 3, 'time': 4.0171895027160645, 'solving': 4.01587438583374, 'sat': {'result': 'UNSATISFIABLE', 'variables': 2, 'clauses': 4, 'conflicts': 2, 'decisions': 1, 'propagations': 2, 'solving': 1.004629135131836}}
        """
        cnf = cnf or []
        self.add(*cnf)
        count = 0
        for model in self.solutions(num=num):
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

import sys
# import os
import tempfile
import shlex
import subprocess
import threading
import time
import re
from .util import CspsatException, CspsatTimeout, TRUE, FALSE, Bool
