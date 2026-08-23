"""The package is ``fast_math``; this makes the other spelling work anyway.

Sessions type ``import fastmath`` because the launcher on PATH is hyphenated.
It cost ten fleet sessions a failed call before anyone wrote this line down.
"""

import sys

import fast_math

sys.modules[__name__] = fast_math
