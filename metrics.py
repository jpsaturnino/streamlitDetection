
import math

def distance(x, y, x2, y2):
    """
    Returns the distance between two points.
    """
    return math.sqrt((x2 - x)**2 + (y2 - y)**2)

def hand_relative_position(xw: float, yw: float, xe: float, ye: float):
    """
    Returns the relative position x, y of the hand in relation to the writs and elbow.

    Args:
        xw (float): Position of the wrist in the x axis.
        yw (float): Position of the wrist in the y axis.
        xe (float): Position of the elbow in the x axis.
        ye (float): Position of the elbow in the y axis.
    """
    WE_POS = 2/3
    xh = xw + WE_POS * (xw - xe)^2
    yh = yw + WE_POS * (yw - ye)^2

    return [xh, yh]

