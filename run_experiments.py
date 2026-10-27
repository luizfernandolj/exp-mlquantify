#!/usr/bin/env python
"""Ponto de entrada dos experimentos. Veja --help para as opcoes."""

import sys

if __name__ == "__main__":
    from expq.cli import main

    sys.exit(main())
