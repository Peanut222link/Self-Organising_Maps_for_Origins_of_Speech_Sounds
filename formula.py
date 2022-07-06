def F1(r, h, p):
    result = ((-392 + 392 * r) * h ** 2 + (596 - 668 * r) * h + (-146 + 166 * r)) * p ** 2 + ((348 - 348 * r) * h ** 2 + (-494 + 606 * r) * h + (141 - 175 * r)) * p + ((340 - 72 * r) * h ** 2 + (-796 + 108 * r) * h + (708 - 38 * r))
    return result

def F2(r, h, p):
    result = ((-1200 + 1208 * r) * h ** 2 + (1320 - 1328 * r) * h + (118 - 158 * r)) * p ** 2 + ((1864 - 1488 * r) * h ** 2 + (-2644 + 1510 * r) * h + (-561 + 221 * r)) * p + ((-670 + 490 * r) * h ** 2 + (1355 - 697 * r) * h + (1517 - 117 * r))
    return result

def F3(r, h, p):
    result = ((604 - 604 * r) * h ** 2 + (1038 - 1178 * r) * h + (246 + 566 * r)) * p ** 2 + ((-1150 + 1262 * r) * h ** 2 + (-1443 + 1313 * r) * h + (-317 - 483 * r)) * p + ((1130 - 836 * r) * h ** 2 + (-315 + 44 * r) * h + (2427 - 127 * r))
    return result

def F4(r, h, p):
    result = ((-1120 + 16 * r) * h ** 2 + (1696 - 180 * r) * h + (500 + 522 * r)) * p ** 2 + ((-140 + 240 * r) * h ** 2 + (-578 + 214 * r) * h + (-692 - 419 * r)) * p + ((1480 - 602 * r) * h ** 2 + (-1220 + 289 * r) * h + (3678 - 178 * r))
    return result

def second_effective_formant(F2, F3, F4):
    # c is a constant in barks
    c = 3.5
    w1 = (c - (F3 -F2)) / c
    w2 = ((F4 - F3) - (F3 - F2)) / (F4 - F2)
    result = 0
    # 4 possible cases
    if ((F3 - F2) > c):
        result = F2
    elif (((F3 - F2) <= c) and ((F4 - F2) >= c)):
        result = ((2 - w1) * F2 + w1 * F3) / 2
    elif (((F4 - F2) <= c) and ((F3 - F2) <= (F4 - F3))):
        result = ((w2 * F2 + (2 - w2) * F3) / 2) - 1
    elif (((F4 - F2) <= c) and ((F3 - F2) >= (F4 - F3))):
        result = (((2 + w2) * F3 - w2 * F4) / 2) - 1
    return result

