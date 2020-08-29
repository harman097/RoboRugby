from __future__ import absolute_import, division, print_function
from typing import Tuple, List, Dict
from collections import namedtuple

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

Point = namedtuple('Point', ['x', 'y'])
Vect2D = namedtuple('Vect2D', ['x', 'y'])

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

class RightTriangle():
    def __init__(self, tpl1 :Tuple[float,float], tpl2 :Tuple[float,float], tpl3 :Tuple[float,float]):
        setX = set()
        setY = set()
        setPoints = {Point(*tpl1), Point(*tpl2), Point(*tpl3)}
        for tplPoint in setPoints:
            setX.add(tplPoint.x)
            setY.add(tplPoint.y)

        if len(setX) != 2 or len(setY) != 2:
            raise Exception(f"Not a valid right triangle: {tpl1}, {tpl2}, {tpl3}")

        self._dblLeft = min(setX)
        self._dblRight = max(setX)
        self._dblTop = min(setY)
        self._dblBottom = max(setY)

        # counter clockwise
        lstPointsOrdered = [
            Point(self._dblLeft, self._dblTop),
            Point(self._dblLeft, self._dblBottom),
            Point(self._dblRight, self._dblBottom),
            Point(self._dblRight, self._dblTop)
        ]

        lngPointMissing = None
        for i in range(len(lstPointsOrdered)):
            if not lstPointsOrdered[i] in setPoints:
                lngPointMissing = i
                break
        if lngPointMissing is None:
            raise Exception("RightTriangle class has issues")

        # assign these consistently based on rotation so that,
        # no matter the orientation of the triangle, we can check
        # collisions consistently (slope_point >= slope_hypotenuse)
        self._tplMissing = lstPointsOrdered[lngPointMissing]
        self._tplHyp0 = lstPointsOrdered[(lngPointMissing + 1) % 4]
        self._tpl90 = lstPointsOrdered[(lngPointMissing + 2) % 4]
        self._tplHyp1 = lstPointsOrdered[(lngPointMissing + 3) % 4]

    @property
    def tplMissing(self):
        # Corner in the rect that isn't included in the triangle
        return self._tplMissing

    @property
    def tplHyp0(self):
        # 0,0 for calculating the slope of the hypotenuse
        return self._tplHyp0

    @property
    def tplHyp1(self):
        # Point to use as the hypotenuse vector
        return self._tplHyp1

    @property
    def tpl90(self):
        # Corner of the triangle that is 90 degrees
        return self._tpl90

    @property
    def left(self):
        return self._dblLeft

    @property
    def top(self):
        return self._dblTop

    @property
    def right(self):
        return self._dblRight

    @property
    def bottom(self):
        return self._dblBottom

    @property
    def corners(self) -> List[Point]:
        return [self.tplHyp0, self.tplHyp1, self.tpl90]

    def contains_point(self, tplPoint: Tuple[float,float]) -> bool:
        dblX, dblY = tplPoint # unpack
        if (self.left <= dblX <= self.right) and (self.top <= dblY <= self.bottom):
            dblSlopeHyp = (self.tplHyp1.y - self.tplHyp0.y) / (self.tplHyp1.x - self.tplHyp0.x)
            dblSlopePnt = (dblY - self.tplHyp0.y) / (dblX - self.tplHyp0.x)
            return dblSlopePnt >= dblSlopeHyp
        else:
            return False









