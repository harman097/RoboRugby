import pygame
from pygame.sprite import Sprite
from typing import List, Tuple
from . import RR_Constants as const
from MyUtils import FloatRect, Point, Div0, distance, get_line_intersection, Line, Vec2D, point_within_line
import math

""" Re-enable for intellisense completion (can't run with it, though, due to load order issues) """
# from . import Robot, Ball, Goal


def robot_in_goal(sprRobot: 'Robot', sprGoal: 'Goal') -> bool:
    for tplCorner in sprRobot.rectDbl.corners:
        if sprGoal.triShape.contains_point(tplCorner):
            return True


def robots_collided(sprRobot1: 'Robot', sprRobot2: 'Robot') -> bool:
    if sprRobot1.rectDbl.contains_point(sprRobot2.rectDbl.center):
        return True

    if sprRobot2.rectDbl.contains_point(sprRobot1.rectDbl.center):
        return True

    for tplCorner in sprRobot1.rectDbl.corners:
        if sprRobot2.rectDbl.contains_point(tplCorner):
            return True

    for tplCorner in sprRobot2.rectDbl.corners:
        if sprRobot1.rectDbl.contains_point(tplCorner):
            return True

    return False


# square whose corners are on the surface of the ball.
# used to find different points on a ball's surface for a given rotation.
_dblHalfRadius_Rad2 = const.BALL_RADIUS * math.pow(2, .5) / 2
_rectBallInner = FloatRect(
    dblLeft=-1 * _dblHalfRadius_Rad2,
    dblRight=_dblHalfRadius_Rad2,
    dblTop=-1 * _dblHalfRadius_Rad2,
    dblBottom=_dblHalfRadius_Rad2
)
_dblBallRadius_Sq = const.BALL_RADIUS * const.BALL_RADIUS


def ball_robot_collided(ball: 'Ball', bot: 'Robot') -> bool:
    """
    If the ball/robot collided, either:
        (1) One of the corners of the bot is inside the ball -OR-
        (2) One of the edges of the bot is intersecting the circle -OR-
        (3) Entire ball is within the bot

    3 should never happen (unless our ball speed goes nuts) and, if it does,
    the collision logic won't work anyways so... ignore for now.
    """
    for corner in bot.rectDbl.corners:
        if distance(corner, ball.rectDbl.center) < const.BALL_RADIUS:
            return True

    # get diameters parallel/perp to the robot sides
    _rectBallInner.center = ball.rectDbl.center
    _rectBallInner.rotation = bot.rectDbl.rotation + 45
    lst_diameters = [
        Line(a=ball.rectDbl.corner(FloatRect.CornerType.TOP_LEFT),
             b=ball.rectDbl.corner(FloatRect.CornerType.BOTTOM_RIGHT)),
        Line(a=ball.rectDbl.corner(FloatRect.CornerType.TOP_RIGHT),
             b=ball.rectDbl.corner(FloatRect.CornerType.BOTTOM_LEFT))
    ]
    # if those lines intersect, ball is colliding
    for side in bot.rectDbl.sides:
        for diameter in lst_diameters:
            tpl_i = get_line_intersection(side, diameter)
            if point_within_line(tpl_i, side) and point_within_line(tpl_i, diameter):
                return True

    return False


def balls_collided(spr1: 'Ball', spr2: 'Ball'):
    return distance(spr1.rectDbl.center, spr2.rectDbl.center) <= const.BALL_RADIUS + const.BALL_RADIUS


def collided_wall(objEntity) -> bool:
    if isinstance(objEntity, pygame.Rect):
        rect = objEntity
    else:
        rect = objEntity.rect

    return rect.left < 0 or \
           rect.right > const.ARENA_WIDTH or \
           rect.top < 0 or \
           rect.bottom > const.ARENA_HEIGHT


def apply_force_to_ball(spr_robot: 'Robot', spr_ball: 'Ball') -> None:
    rect_bot = spr_robot.rectDbl  # type: FloatRect
    rect_bot_prev = spr_robot.rectDblPriorFrame  # type: FloatRect
    rect_bot_half = rect_bot.copy()  # type: FloatRect
    rect_bot_half.rotation = (rect_bot_prev.rotation + rect_bot.rotation) / 2

    # Check pending collision point from prior frame
    # create list of diameter lines that are parallel/perpendicular to the sides of the bot's prior position
    _rectBallInner.rotation = 45 + rect_bot_half.rotation  # fudging this for simplicity's sake
    _rectBallInner.center = spr_ball.rectDbl.center
    lst_diameters = [
        Line(
            a=_rectBallInner.corner(FloatRect.CornerType.BOTTOM_LEFT),
            b=_rectBallInner.corner(FloatRect.CornerType.TOP_RIGHT)
        ),
        Line(
            a=_rectBallInner.corner(FloatRect.CornerType.BOTTOM_RIGHT),
            b=_rectBallInner.corner(FloatRect.CornerType.TOP_LEFT)
        )
    ]

    """ CHECK FOR SURFACE COLLISION """
    # if either of these lines intersect with a side of the bot,
    # treat this as a "surface collision" and not a corner collision
    for enm_side in FloatRect.SideType:
        bot_side = rect_bot.side(enm_side)
        bot_side_prev = rect_bot_prev.side(enm_side)
        for diameter in lst_diameters:

            tpl_i = get_line_intersection(bot_side, diameter)
            tpl_i_prev = get_line_intersection(bot_side_prev, diameter)
            if point_within_line(tpl_i, bot_side) and \
                    point_within_line(tpl_i_prev, bot_side_prev) and \
                    point_within_line(tpl_i, diameter, buffer=2):  #todo buffer of 2 is stupid... resolve your fp issues
                # Calculate minimum distance for ball to leave the robot
                tpl_contact = Vec2D(x=tpl_i[0] - tpl_i_prev[0], y=tpl_i[1] - tpl_i_prev[1])
                buffer_mult = 1.1  # contact point is slightly fudged - give it some buffer
                spr_ball.dbl_force_x += tpl_contact.x * buffer_mult
                spr_ball.dbl_force_y += tpl_contact.y * buffer_mult
                spr_ball.lngFrameMass = max(spr_robot.lngFrameMass, spr_ball.lngFrameMass)
                return  # can only be one pending collision point with this logic

    """ CHECK FOR CORNER COLLISION """
    for enm_corner in FloatRect.CornerType:
        bot_corner = rect_bot.corner(enm_corner)
        dbl_dist_center = distance(bot_corner, spr_ball.rectDbl.center)
        if dbl_dist_center < const.BALL_RADIUS:
            bot_corner_prev = rect_bot_prev.corner(enm_corner)
            # Let's just... fudge this a bit... instead of actually calc'ing (weighted average)
            tpl_contact = Vec2D(
                x=spr_ball.rectDbl.centerx - (bot_corner.x * 3 + bot_corner_prev.x) / 4,
                y=spr_ball.rectDbl.centery - (bot_corner.y * 3 + bot_corner_prev.y) / 4)

            # Fudge this a bit, since we're not calc'ing the exact contact point on the ball
            contact_dist = distance((0, 0), tpl_contact)
            exit_dist = (const.BALL_RADIUS - dbl_dist_center) * 1.2
            spr_ball.dbl_force_x += tpl_contact.x * exit_dist / contact_dist
            spr_ball.dbl_force_y += tpl_contact.y * exit_dist / contact_dist
            spr_ball.lngFrameMass = max(spr_robot.lngFrameMass, spr_ball.lngFrameMass)
            return


def bounce_ball_off_bot(spr_robot: 'Robot', spr_ball: 'Ball'):
    if spr_ball.dbl_velocity_x == spr_ball.dbl_velocity_y == 0:
        return  # no velocity to "bounce"

    rect_bot = spr_robot.rectDbl  # type: FloatRect
    rect_bot_prev = spr_robot.rectDblPriorFrame  # type: FloatRect
    rect_bot_half = rect_bot.copy()  # type: FloatRect
    rect_bot_half.rotation = (rect_bot_prev.rotation + rect_bot.rotation) / 2

    # Check pending collision point from prior frame
    # create list of diameter lines that are parallel/perpendicular to the sides of the bot's prior position
    _rectBallInner.rotation = 45 + rect_bot_half.rotation  # fudging this for simplicity's sake
    _rectBallInner.center = spr_ball.rectDbl.center
    lst_diameters = [
        Line(
            a=_rectBallInner.corner(FloatRect.CornerType.BOTTOM_LEFT),
            b=_rectBallInner.corner(FloatRect.CornerType.TOP_RIGHT)
        ),
        Line(
            a=_rectBallInner.corner(FloatRect.CornerType.BOTTOM_RIGHT),
            b=_rectBallInner.corner(FloatRect.CornerType.TOP_LEFT)
        )
    ]

    """ CHECK FOR SURFACE COLLISION """
    # if either of these lines intersect with a side of the bot,
    # treat this as a "surface collision" and not a corner collision
    for enm_side in FloatRect.SideType:
        bot_side = rect_bot.side(enm_side)
        bot_side_prev = rect_bot_prev.side(enm_side)
        for diameter in lst_diameters:
            tpl_i = get_line_intersection(bot_side, diameter)
            tpl_i_prev = get_line_intersection(bot_side_prev, diameter)
            if point_within_line(tpl_i, bot_side) and \
                    point_within_line(tpl_i_prev, bot_side_prev) and \
                    point_within_line(tpl_i, diameter, buffer=.5):
                dist_a = distance(diameter.a, tpl_i_prev)
                dist_b = distance(diameter.b, tpl_i_prev)
                tpl_cp_ball = diameter.a if dist_a < dist_b else diameter.b
                tpl_cp_bot = Point(*tpl_i)
                tpl_cp_bot_prev = Point(*tpl_i_prev)
                tpl_cp_bot_half = Point(x=(tpl_cp_bot.x + tpl_cp_bot_prev.x) / 2,
                                        y=(tpl_cp_bot.y + tpl_cp_bot_prev.y) / 2)

                tpl_contact = Vec2D(x=tpl_cp_bot_half.x - tpl_cp_ball.x, y=tpl_cp_bot_half.y - tpl_cp_ball.y)
                tpl_ball_vel = Vec2D(x=spr_ball.dbl_velocity_x, y=spr_ball.dbl_velocity_y)

                # Projected components of the ball's velocity
                dbl_contact_dist_sq = tpl_contact.x ** 2 + tpl_contact.y ** 2
                dbl_ball_proj_term = ((tpl_contact.x * tpl_ball_vel.x) + (tpl_contact.y * tpl_ball_vel.y)) \
                                     / dbl_contact_dist_sq
                tpl_ball_proj = Vec2D(x=dbl_ball_proj_term * tpl_contact.x, y=dbl_ball_proj_term * tpl_contact.y)

                if (tpl_ball_proj.x < 0 and tpl_contact.x > 0) or (tpl_ball_proj.x > 0 and tpl_contact.x < 0):
                    spr_ball.dbl_velocity_x = -tpl_ball_proj.x * const.BOUNCE_K_WALL * const.BOUNCE_K_ROBOT
                if (tpl_ball_proj.y < 0 and tpl_contact.y > 0) or (tpl_ball_proj.y > 0 and tpl_contact.y < 0):
                    spr_ball.dbl_velocity_y = -tpl_ball_proj.y * const.BOUNCE_K_WALL * const.BOUNCE_K_ROBOT

                tpl_exit_point = Point(*get_line_intersection(bot_side, (tpl_cp_ball, tpl_cp_bot_half)))
                spr_ball.rectDbl.centerx += (tpl_exit_point.x - tpl_cp_ball.x) * 1.1
                spr_ball.rectDbl.centery += (tpl_exit_point.y - tpl_cp_ball.y) * 1.1
                return

    """ CHECK FOR CORNER COLLISION """
    for enm_corner in FloatRect.CornerType:
        bot_corner = rect_bot.corner(enm_corner)
        dbl_dist_center = distance(bot_corner, spr_ball.rectDbl.center)
        if dbl_dist_center < const.BALL_RADIUS:
            bot_corner_prev = rect_bot_prev.corner(enm_corner)
            # Let's just... fudge this a bit... instead of actually calc'ing (weighted average)
            tpl_cp_bot = Point(x=(bot_corner.x*3 + bot_corner_prev.x)/4, y=(bot_corner.y*3 + bot_corner_prev.y)/4)
            tpl_contact = Vec2D(x=spr_ball.rectDbl.centerx - tpl_cp_bot.x, y=spr_ball.rectDbl.centery - tpl_cp_bot.y)

            # Project the ball vel onto the contact vector
            tpl_ball_vel = Vec2D(x=spr_ball.dbl_velocity_x, y=spr_ball.dbl_velocity_y)
            dbl_contact_dist_sq = tpl_contact.x ** 2 + tpl_contact.y ** 2
            dbl_ball_proj_term = ((tpl_contact.x * tpl_ball_vel.x) + (tpl_contact.y * tpl_ball_vel.y)) \
                                 / dbl_contact_dist_sq
            tpl_ball_proj = Vec2D(x=dbl_ball_proj_term * tpl_contact.x, y=dbl_ball_proj_term * tpl_contact.y)

            # Apply bounce, if applicable
            if (tpl_ball_proj.x < 0 and tpl_contact.x > 0) or (tpl_ball_proj.x > 0 and tpl_contact.x < 0):
                spr_ball.dbl_velocity_x = -tpl_ball_proj.x * const.BOUNCE_K_WALL * const.BOUNCE_K_ROBOT
            if (tpl_ball_proj.y < 0 and tpl_contact.y > 0) or (tpl_ball_proj.y > 0 and tpl_contact.y < 0):
                spr_ball.dbl_velocity_y = -tpl_ball_proj.y * const.BOUNCE_K_WALL * const.BOUNCE_K_ROBOT

            # Resolve collision - contact point is fudged, fudge travel dist accordingly
            dbl_exit_dist = distance(bot_corner_prev, spr_ball.rectDbl.center)
            dbl_contact_dist = dbl_contact_dist_sq ** .5
            spr_ball.rectDbl.centerx += tpl_contact.x * dbl_exit_dist / dbl_contact_dist
            spr_ball.rectDbl.centery += tpl_contact.y * dbl_exit_dist / dbl_contact_dist
            return


def bounce_balls(ball1: 'Ball', ball2: 'Ball') -> None:
    if ball1.rectDbl.center == ball2.rectDbl.center:
        raise Exception("Really tho?? The balls are in the EXACT same spot????")

    # Determine vector between two centers
    vec_1_2 = Vec2D(ball2.rectDbl.centerx - ball1.rectDbl.centerx, ball2.rectDbl.centery - ball1.rectDbl.centery)
    dist_1_2 = distance((0,0), vec_1_2)

    # Create vector in that direction of length radius
    vec_radius = Vec2D(vec_1_2.x * const.BALL_RADIUS / dist_1_2, vec_1_2.y * const.BALL_RADIUS / dist_1_2)

    # Determine point on Ball 1 closest to the center of Ball 2
    tplPnt1 = Point(x=ball1.rectDbl.centerx + vec_radius.x, y=ball1.rectDbl.centery + vec_radius.y)
    # and vice versa
    tplPnt2 = Point(x=ball2.rectDbl.centerx - vec_radius.x, y=ball2.rectDbl.centery - vec_radius.y)

    buffer = 1.1
    dblHalfDeltaX = (tplPnt2.x - tplPnt1.x) / 2
    dblHalfDeltaY = (tplPnt2.y - tplPnt1.y) / 2
    if ball1.lngFrameMass == ball2.lngFrameMass:
        """ Move balls out of each other. """
        ball1.rectDbl.centerx += dblHalfDeltaX * buffer
        ball1.rectDbl.centery += dblHalfDeltaY * buffer
        ball2.rectDbl.centerx -= dblHalfDeltaX * buffer
        ball2.rectDbl.centery -= dblHalfDeltaY * buffer
    else:
        """ Only one ball moves. """
        if ball1.lngFrameMass > ball2.lngFrameMass:
            ball2.rectDbl.centerx += (tplPnt1.x - tplPnt2.x) * buffer
            ball2.rectDbl.centery += (tplPnt1.y - tplPnt2.y) * buffer
            ball2.lngFrameMass = ball1.lngFrameMass  # and now they are one
        else:
            ball1.rectDbl.centerx += (tplPnt2.x - tplPnt1.x) * buffer
            ball1.rectDbl.centery += (tplPnt2.y - tplPnt1.y) * buffer
            ball1.lngFrameMass = ball2.lngFrameMass  # and now they are one

    """ Bounce! """
    # Let's conserve some momentum with vector projection (ya i totally remembered how to do this)
    # projection of 1's velocity onto contact vector:
    tplVect1to2 = Vec2D(x=ball2.rectDbl.centerx - ball1.rectDbl.centerx, y=ball2.rectDbl.centery - ball1.rectDbl.centery)
    tplVect2to1 = Vec2D(x=tplVect1to2.x * -1, y=tplVect1to2.y * -1)
    dblCenterDist_Sq = tplVect1to2.x * tplVect1to2.x + tplVect1to2.y * tplVect1to2.y
    dblProjClcTerm1 = Div0(tplVect1to2.x * ball1.dbl_velocity_x + tplVect1to2.y * ball1.dbl_velocity_y, dblCenterDist_Sq)
    tplVel1 = Vec2D(x=dblProjClcTerm1 * tplVect1to2.x, y=dblProjClcTerm1 * tplVect1to2.y)
    dblProjClcTerm2 = Div0(tplVect2to1.x * ball2.dbl_velocity_x + tplVect2to1.y * ball2.dbl_velocity_y, dblCenterDist_Sq)
    tplVel2 = Vec2D(x=dblProjClcTerm2 * tplVect2to1.x, y=dblProjClcTerm2 * tplVect2to1.y)
    tplDiff = Vec2D(x=tplVel1.x - tplVel2.x, y=tplVel1.y - tplVel2.y)
    ball1.dbl_velocity_x -= tplDiff.x * const.BOUNCE_K_BALL
    ball1.dbl_velocity_y -= tplDiff.y * const.BOUNCE_K_BALL
    ball2.dbl_velocity_x += tplDiff.x * const.BOUNCE_K_BALL
    ball2.dbl_velocity_y += tplDiff.y * const.BOUNCE_K_BALL

    """ ... but also account for force """
    if ball1.dbl_force_x > 0:
        ball1.dbl_velocity_x = max(ball1.dbl_velocity_x, ball1.dbl_force_x)
    elif ball1.dbl_force_x < 0:
        ball1.dbl_velocity_x = min(ball1.dbl_velocity_x, ball1.dbl_force_x)
    if ball1.dbl_force_y > 0:
        ball1.dbl_velocity_y = max(ball1.dbl_velocity_y, ball1.dbl_force_y)
    elif ball1.dbl_force_y < 0:
        ball1.dbl_velocity_y = min(ball1.dbl_velocity_y, ball1.dbl_force_y)
    if ball2.dbl_force_x > 0:
        ball2.dbl_velocity_x = max(ball2.dbl_velocity_x, ball2.dbl_force_x)
    elif ball2.dbl_force_x < 0:
        ball2.dbl_velocity_x = min(ball2.dbl_velocity_x, ball2.dbl_force_x)
    if ball2.dbl_force_y > 0:
        ball2.dbl_velocity_y = max(ball2.dbl_velocity_y, ball2.dbl_force_y)
    elif ball2.dbl_force_y < 0:
        ball2.dbl_velocity_y = min(ball2.dbl_velocity_y, ball2.dbl_force_y)



def bounce_ball_off_wall(ball: 'Ball'):
    # Bounce the ball off the wall
    # "K_Wall" = how much to dampen the bouncew
    if ball.rectDbl.left < 0:
        ball.rectDbl.left *= -1.1
        ball.dbl_velocity_x *= -1 * const.BOUNCE_K_WALL
        ball.lngFrameMass = const.MASS_WALL
    if ball.rectDbl.right > const.ARENA_WIDTH:
        ball.rectDbl.right = const.ARENA_WIDTH - (ball.rectDbl.right - const.ARENA_WIDTH) * 1.1
        ball.dbl_velocity_x *= -1 * const.BOUNCE_K_WALL
        ball.lngFrameMass = const.MASS_WALL
    if ball.rectDbl.top <= 0:
        ball.rectDbl.top *= -1.1
        ball.dbl_velocity_y *= -1 * const.BOUNCE_K_WALL
        ball.lngFrameMass = const.MASS_WALL
    if ball.rectDbl.bottom >= const.ARENA_HEIGHT:
        ball.rectDbl.bottom = const.ARENA_HEIGHT - (ball.rectDbl.bottom - const.ARENA_WIDTH) * 1.1
        ball.dbl_velocity_y *= -1 * const.BOUNCE_K_WALL
        ball.lngFrameMass = const.MASS_WALL


def collision_pairs_self(grpSprites: pygame.sprite.Group,
                         fncCollided=pygame.sprite.collide_rect) -> List[Tuple[Sprite, Sprite]]:
    lstPairs = []
    lstSpr = grpSprites.sprites()
    for i in range(len(lstSpr)):
        for j in range(i + 1, len(lstSpr)):
            if fncCollided(lstSpr[i], lstSpr[j]):
                lstPairs.append((lstSpr[i], lstSpr[j]))
    return lstPairs


def collision_pairs(grpSprites1: pygame.sprite.Group,
                    grpSprites2: pygame.sprite.Group,
                    fncCollided=pygame.sprite.collide_rect) -> List[Tuple[Sprite, Sprite]]:
    lstPairs = []
    lstSpr1 = grpSprites1.sprites()
    lstSpr2 = grpSprites2.sprites()
    for i in range(len(lstSpr1)):
        for j in range(len(lstSpr2)):
            if fncCollided(lstSpr1[i], lstSpr2[j]):
                lstPairs.append((lstSpr1[i], lstSpr2[j]))
    return lstPairs


def two_way_lidar_rect(tpl1: Tuple[float, float],
                       tpl2: Tuple[float, float],
                       lst_rect: List[FloatRect]) -> Tuple[float, float]:
    """
    Returns the distance to the nearest object (FloatRect) along the line formed by the two points. This is "two way" so
    two values are returned for both "forwards" (1->2) and "backwards" (2->1).
    """
    start = Point(*tpl1)
    end = Point(*tpl2)

    lst_front = [float("inf")]
    lst_back = [float("inf")]
    for rect in lst_rect:
        for side in rect.sides:
            intersection = Point(*get_line_intersection(side, (start, end)))
            if intersection.x is None or intersection.y is None:
                pass  # doesn't intersect

            else:
                dist_endpoint = distance(intersection, end)
                dist_startpoint = distance(intersection, start)
                if dist_endpoint <= dist_startpoint:
                    lst_front.append(dist_endpoint)
                if dist_startpoint <= dist_endpoint:
                    lst_back.append(dist_startpoint)

    return min(lst_front), min(lst_back)
