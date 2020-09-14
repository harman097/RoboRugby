from __future__ import absolute_import, division, print_function
from typing import Tuple, List, Dict
from collections import namedtuple
from enum import Enum
import math

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

def distance(tplA :Tuple[float,float], tplB :Tuple[float,float]) -> float:
    return math.pow((tplB[0] - tplA[0])**2 + (tplB[1] - tplA[1])**2, .5)

# These are defined with ccw rotation = positive, y-axis inverted (aka totally fucked, fix later if you want)
def angle_degrees(tpl_a :Tuple[float,float], tpl_b :Tuple[float,float]) -> float:
    return math.degrees(math.tanh(tpl_b[0] - tpl_a[0] / tpl_a[1] - tpl_b[1]))

def angle_radians(tpl_a :Tuple[float,float], tpl_b :Tuple[float,float]) -> float:
    return math.tanh(tpl_b[0] - tpl_a[0] / tpl_a[1] - tpl_b[1])

# pygame rects are just ints, unfortunately, which makes any physics difficult
class FloatRect:

    def __init__(self, dblLeft: float, dblRight: float, dblTop: float, dblBottom: float):
        if dblLeft >= dblRight:
            raise Exception("Left >= Right")
        if dblTop >= dblBottom:
            raise Exception("Top >= Bottom")

        self._dblWidth = dblRight - dblLeft
        self._dblHeight = dblBottom - dblTop
        self._dblCenterX = (dblLeft + dblRight) / 2
        self._dblCenterY = (dblTop + dblBottom) / 2
        self._dblLeft = dblLeft
        self._dblRight = dblRight
        self._dblTop = dblTop
        self._dblBottom = dblBottom
        self._dblRotation = 0

        self._dctInitialCornersRelCenter = {  # type: Dict[FloatRect.CornerType, Point]
            FloatRect.CornerType.TOP_LEFT: Point(x=dblLeft - self._dblCenterX, y=dblTop - self._dblCenterY),
            FloatRect.CornerType.TOP_RIGHT: Point(x=dblRight - self._dblCenterX,y=dblTop - self._dblCenterY),
            FloatRect.CornerType.BOTTOM_LEFT: Point(x=dblLeft - self._dblCenterX, y=dblBottom - self._dblCenterY),
            FloatRect.CornerType.BOTTOM_RIGHT: Point(x=dblRight - self._dblCenterX, y=dblBottom - self._dblCenterY)
        }

        self._dctCornersRelCenter = self._dctInitialCornersRelCenter.copy()  # type: Dict[FloatRect.CornerType, Point]

    def _move_linear(self, dblDeltaX :float, dblDeltaY :float):
        self._dblCenterX += dblDeltaX
        self._dblLeft += dblDeltaX
        self._dblRight += dblDeltaX

        self._dblCenterY += dblDeltaY
        self._dblTop += dblDeltaY
        self._dblBottom += dblDeltaY

    def copy(self) -> 'FloatRect':
        rectNew = FloatRect(self.left, self.right, self.top, self.bottom)
        rectNew.rotation = self.rotation
        return rectNew

#region Properties/Getters/Setters

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
        return self._dblCenterX

    @property
    def centery(self) -> float:
        return self._dblCenterY

    @property
    def center(self) -> Tuple[float, float]:
        return (self.centerx, self.centery)

    @property
    def rotation(self) -> float:
        return self._dblRotation

    def corner(self, enmCorner) -> Point:
        return Point(
            x=self.centerx + self._dctCornersRelCenter[enmCorner].x,
            y=self.centery + self._dctCornersRelCenter[enmCorner].y
        )

    @property
    def corners(self) -> List[Point]:
        return list(map(self.corner, FloatRect.CornerType))

    def side(self, enmSide: 'FloatRect.SideType') -> Tuple[Point,Point]:
        if enmSide == FloatRect.SideType.RIGHT:
            return (
                self.corner(FloatRect.CornerType.TOP_RIGHT),
                self.corner(FloatRect.CornerType.BOTTOM_RIGHT)
            )
        elif enmSide == FloatRect.SideType.TOP:
            return (
                self.corner(FloatRect.CornerType.TOP_LEFT),
                self.corner(FloatRect.CornerType.TOP_RIGHT)
            )
        elif enmSide == FloatRect.SideType.LEFT:
            return (
                self.corner(FloatRect.CornerType.BOTTOM_LEFT),
                self.corner(FloatRect.CornerType.TOP_LEFT)
            )
        elif enmSide == FloatRect.SideType.BOTTOM:
            return (
                self.corner(FloatRect.CornerType.BOTTOM_RIGHT),
                self.corner(FloatRect.CornerType.BOTTOM_LEFT)
            )

    @property
    def sides(self) -> List[Point]:
        return list(map(self.side, FloatRect.SideType))

    def side_as_right_triangle(self, enmSide: 'FloatRect.SideType') -> 'RightTriangle':
        tplSide = self.side(enmSide)
        tplA, tplB = tplSide
        # determine "C" (3rd point in triangle)
        if self.left < tplA.x < self.right:
            shpTriangle = RightTriangle(tplA, tplB, Point(x=tplA.x, y=tplB.y))
        else:
            shpTriangle = RightTriangle(tplA, tplB, Point(x=tplB.x, y=tplA.y))
        return shpTriangle

    def sides_as_right_triangles(self) -> List['RightTriangle']:
        return list(map(self.side_as_right_triangle, FloatRect.SideType))



    # Setters
    @top.setter
    def top(self, dblTop: float):
        self._move_linear(0, dblTop - self._dblTop)

    @bottom.setter
    def bottom(self, dblBottom: float):
        self._move_linear(0, dblBottom - self._dblBottom)

    @right.setter
    def right(self, dblRight: float):
        self._move_linear(dblRight - self._dblRight, 0)

    @left.setter
    def left(self, dblLeft: float):
        self._move_linear(dblLeft - self._dblLeft, 0)

    @centerx.setter
    def centerx(self, dblCenterx: float):
        self._move_linear(dblCenterx - self._dblCenterX, 0)

    @centery.setter
    def centery(self, dblCentery: float):
        self._move_linear(0, dblCentery - self._dblCenterY)

    @center.setter
    def center(self, tplCenter: Tuple[float, float]):
        self.centerx, self.centery = tplCenter

    @rotation.setter
    def rotation(self, dblRotation):
        if dblRotation == self._dblRotation:
            return

        self._dblRotation = dblRotation % 360
        dblRotRadians = math.radians(360 - dblRotation)  # because y is flipped
        dblCos = math.cos(dblRotRadians)
        dblSin = math.sin(dblRotRadians)

        # Rotate our corners (relative to center) by the specified angle
        # Rotation matrix on a normal graph is:
        #   [(cos, -sin),
        #    (sin,  cos)]

        self._dctCornersRelCenter = self._dctInitialCornersRelCenter.copy()
        setX = set()
        setY = set()
        for tplKV in self._dctCornersRelCenter.items():
            enmCornerType, tplCorner = tplKV  # unpack
            tplRotatedCorner = Point(
                x=-1 * tplCorner.x * dblCos + tplCorner.y * dblSin,
                y=-1 * tplCorner.x * dblSin - tplCorner.y * dblCos
            )
            self._dctCornersRelCenter[enmCornerType] = tplRotatedCorner
            setX.add(tplRotatedCorner.x)
            setY.add(tplRotatedCorner.y)

        # update l,r,t,b to reflect new rotation
        self._dblLeft = min(setX) + self.centerx
        self._dblRight = max(setX) + self.centerx
        self._dblTop = min(setY) + self.centery
        self._dblBottom = max(setY) + self.centery

#endregion

    class CornerType(Enum):
        TOP_LEFT = 0
        TOP_RIGHT = 1
        BOTTOM_LEFT = 2
        BOTTOM_RIGHT = 3

    class SideType(Enum):
        RIGHT = 0  # side that is at 0 deg from center, relative to rect's rotation
        TOP = 90  # side that is at 90 deg from center, relative to rect's rotation
        LEFT = 180  # side that is at 180 deg from center, relative to rect's rotation
        BOTTOM = 270  # side that is at 270 deg from center, relative to rect's rotation

    def contains_point(self, tplPoint: Tuple[float,float]) -> bool:
        dblX, dblY = tplPoint # unpack
        if (self.left <= dblX <= self.right) and (self.top <= dblY <= self.bottom):
            if round(self.rotation % 90) == 0 or round(self.rotation % 90) == 90:
                return True
            else:
                set_x = set()
                set_y = set()
                for shpTriangle in self.sides_as_right_triangles():
                    set_x.add(shpTriangle.tpl90.x)
                    set_y.add(shpTriangle.tpl90.y)
                    if shpTriangle.contains_point(tplPoint):
                        return True
                # check inner square (if valid) yoyo
                return (min(set_x) <= dblX <= max(set_x)) and (min(set_y) <= dblY <= max(set_y))
        else:
            return False

class RightTriangle():
    def __init__(self, tpl1 :Tuple[float,float], tpl2 :Tuple[float,float], tpl3 :Tuple[float,float]):
        setX = set()
        setY = set()
        setPoints = {Point(*tpl1), Point(*tpl2), Point(*tpl3)}
        for tplPoint in setPoints:
            setX.add(tplPoint.x)
            setY.add(tplPoint.y)

        if len(setX) != 2 or len(setY) != 2 or len(setPoints) != 3:
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
            dblSlopeHyp = Div0(self.tplHyp1.y - self.tplHyp0.y, self.tplHyp1.x - self.tplHyp0.x)
            dblSlopePnt = Div0(dblY - self.tplHyp0.y, dblX - self.tplHyp0.x)
            return dblSlopePnt >= dblSlopeHyp
        else:
            return False









