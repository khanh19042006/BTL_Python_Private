import math

def calculate_distance(landmark1, landmark2):
    if landmark1 is None or landmark2 is None:
        return None

    lx = (landmark1.x - landmark2.x) ** 2
    ly = (landmark1.y - landmark2.y) ** 2
    lz = (landmark1.z - landmark2.z) ** 2
    distance = math.sqrt(lx + ly + lz)
    return distance

def calculate_threshold(hand_landmarks):
    index_mcp = hand_landmarks[5]        #Gốc ngón trỏ
    pinky_mcp = hand_landmarks[17]       #Gốc ngón út
    return calculate_distance(index_mcp, pinky_mcp)

