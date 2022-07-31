def analyzeHASequence(sequence, targetColor):
    print(sequence)
    print(">>>", targetColor)
    if targetColor != sequence[0]:
        return False
    idx = 0
    p1 = 0
    while idx < len(sequence) and sequence[idx] == targetColor:
        idx += 1
    p2 = idx
    while idx < len(sequence) and sequence[idx] != targetColor:
        idx+=1
    p3 = idx

    D1 = p2 - p1
    D2 = p3 - p2

    if D1 <2:
        return False
    if D2 >=5 or p2 + D2 >= len(sequence):
        return True
    return False


bad = [False, True, True, False, False, True, True, False, False]
good = [True, True, False, False, False, False, False, False, False]
full = [True, True, True, True, True]

print(analyzeHASequence(good, True))
print(analyzeHASequence(bad, True))
print(analyzeHASequence(good, False))
print(analyzeHASequence(bad, False))
print(analyzeHASequence(full, True))

