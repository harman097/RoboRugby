from __future__ import absolute_import, division, print_function
PRINT_STAGE = True
def Stage(pStr):
    # apparently you can add attributes to a function?
    Stage.text = pStr # rough equivalent of Static
    if PRINT_STAGE:
        print(pStr)

def Div0(pdblNumerator: float, pdblDenominator: float) -> float:
    if pdblDenominator != 0:
        return pdblNumerator / pdblDenominator
    elif pdblNumerator > 0:
        return float("inf")
    elif pdblNumerator < 0:
        return float("-inf")
    else:
        raise ZeroDivisionError("Numerator AND Denominator are both 0.")
