#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import getopt
import fileinput
from cspsat import SAT

def main():
    verbose = 0
    solver = None
    max = 1
    opts, args = getopt.getopt(sys.argv[1:], "hvs:m:", ["help", "verbose=", "solver=", "max="])
    for o,x in opts:
        if o in ["-h", "--help"]:
            print("Usage: %s [options] file..." % sys.argv[0])
            print("  -h : help")
            print("  -v : verbose output")
            print("  -s solver : SAT solver ('glueminisat -show-model', 'glucose -model', kissat, clasp)")
            print("  -m max : Maximum number of models to be searched (0: all)")
            return
        elif o in ["-v"]:
            verbose = 1
        elif o in ["--verbose"]:
            verbose = int(x)
        elif o in ["-m", "--max"]:
            max = int(x)
    input = fileinput.FileInput(files=args)
    sat = SAT(solver)
    sat.load(input)
    sat.solve(max=max, verbose=verbose, stat=True)

if __name__ == "__main__":
    main()
