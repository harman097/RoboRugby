from __future__ import absolute_import, division, print_function
from typing import Tuple

PRINT_STAGE = True


def Stage(pStr):
    # apparently you can add attributes to a function?
    Stage.text = pStr  # rough equivalent of Static
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


# pygame rects are just ints, unfortunately, which makes any physics difficult
class FloatRect:
    def __init__(self, dblLeft: float, dblRight: float, dblTop: float, dblBottom: float):
        if dblLeft >= dblRight:
            raise Exception("Left >= Right")
        if dblTop >= dblBottom:
            raise Exception("Top >= Bottom")
        self._dblLeft = dblLeft
        self._dblRight = dblRight
        self._dblTop = dblTop
        self._dblBottom = dblBottom
        self._dblWidth = dblRight - dblLeft
        self._dblHeight = dblBottom - dblTop

    def copy(self) -> 'FloatRect':
        return FloatRect(self.left, self.right, self.top, self.bottom)

    # Getters

    @property
    def top(self) -> float:
        return self._dblTop

    @property
    def right(self) -> float:
        return self._dblRight

    @property
    def bottom(self) -> float:
        return self._dblBottom

    @property
    def left(self) -> float:
        return self._dblLeft

    @property
    def width(self) -> float:
        return self._dblWidth

    @property
    def height(self) -> float:
        return self._dblHeight

    @property
    def centerx(self) -> float:
        return (self.left + self.right) / 2

    @property
    def centery(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def center(self) -> Tuple[float, float]:
        return (self.centerx, self.centery)

    # Setters
    @top.setter
    def top(self, dblTop: float):
        self._dblTop = dblTop
        self._dblBottom = dblTop + self.height

    @bottom.setter
    def bottom(self, dblBottom: float):
        self._dblBottom = dblBottom
        self._dblTop = dblBottom - self.height

    @right.setter
    def right(self, dblRight: float):
        self._dblRight = dblRight
        self._dblLeft = dblRight - self.width

    @left.setter
    def left(self, dblLeft: float):
        self._dblLeft = dblLeft
        self._dblRight = dblLeft + self.width

    @width.setter
    def width(self, dblWidth: float):
        dblHalfDelta = (dblWidth - self._dblWidth)/2
        self._dblLeft -= dblHalfDelta
        self._dblRight += dblHalfDelta
        self._dblWidth = dblWidth

    @height.setter
    def height(self, dblHeight: float):
        dblHalfDelta = (dblHeight - self._dblHeight) / 2
        self._dblTop -= dblHalfDelta
        self._dblBottom += dblHalfDelta
        self._dblHeight = dblHeight

    @centerx.setter
    def centerx(self, dblCenterx: float):
        self.left += dblCenterx - self.centerx

    @centery.setter
    def centery(self, dblCentery: float):
        self.top += dblCentery - self.centery

    @center.setter
    def center(self, tplCenter: Tuple[float, float]):
        self.centerx, self.centery = tplCenter
