from __future__ import absolute_import, division, print_function
PRINT_STAGE = True
def Stage(pStr):
    # apparently you can add attributes to a function?
    Stage.text = pStr # rough equivalent of Static
    if PRINT_STAGE:
        print(pStr)
