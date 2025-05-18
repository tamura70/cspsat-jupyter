#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import getopt
import fileinput
from cspsat import *
from cspsat.dpll import DPLL

def main():
    verbose = 0
    interactive = False
    max = 1
    stat = False
    opts, args = getopt.getopt(sys.argv[1:], "hvim:", ["help", "verbose=", "max="])
    for o,x in opts:
        if o in ["-h", "--help"]:
            print("Usage: %s [options] file..." % sys.argv[0])
            print("  -h : help")
            print("  -v : verbose output")
            print("  -i : interactive mode (implies verbose output)")
            print("  -m max : Maximum number of models to be searched (0: all)")
            return
        elif o in ["-v"]:
            verbose = 1
        elif o in ["--verbose"]:
            verbose = int(x)
        elif o in ["-i"]:
            interactive = True
            verbose = 1
        elif o in ["-m", "--max"]:
            max = int(x)
    input = fileinput.FileInput(files=args)
    dpll = DPLL()
    dpll.load(input)
    dpll.solve(max=max, verbose=verbose, interactive=interactive, stat=True)

if __name__ == "__main__":
    main()
