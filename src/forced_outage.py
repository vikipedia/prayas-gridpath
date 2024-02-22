import pandas as pd
import numpy as np
from collections import namedtuple
import collections


def generate_fo(length, average, maxoutage, minoutage):
    pass


def combine_fo_m(m, f):
    """    
    derate
    1 -> available
    0 -> not available

    rules for combining after moving fo
    r -> min(m,fo)
    """
    moved_f = move_fo(m, f)
    df = pd.DataFrame({"m": m, "newf": moved_f})
    return df.apply(min, axis=1)


def move_fo(m, f):
    """
    m      f
    <1     <1    move fo where m is 1
    """
    ms = steps(m)
    fs = steps(f)
    targets = target_intervals(ms, fs)
    l = possible_swap_locations(m, f)

    return moveall(targets, l, f, ms, fs)


Interval = collections.namedtuple("Interval", ['start', 'end', 'value'])


def steps(seq):
    s = [seq[i] == seq[i+1] for i in range(len(seq)-1)]
    count = s.count(False)
    n = 0
    start = 0
    endpoints = []
    for i in range(count):
        end = start + s.index(False)+1
        s = s[end-start:]
        endpoints.append(Interval(start, end, seq[start]))
        start = end
    endpoints.append(Interval(start, len(seq), seq[start]))
    return endpoints


def divide(seq):
    steps_ = steps(seq)
    return [seq[s:e] for s, e, v in steps_]


def overlap(interval1, interval2):
    s1 = set(range(interval1.start, interval1.end))
    s2 = set(range(interval2.start, interval2.end))
    intersect = s1 & s2
    if intersect:
        return Interval(min(intersect), max(intersect)+1, interval1.value)


def target_intervals(maint, forced):
    """
    Find intervals in forced such that
    maint.value < 1 and forced.value < 1
    """
    return set(f for interval in maint for f in forced if interval.value < 1 and f.value < 1 and overlap(interval, f))


def distance(first, second):
    """
    part of refactored function
    """
    return second.start - first.start


def left(target, ms):
    """
    find left hand side positions
    """
    return [m for m in ms if distance(target, m) <= 0 and target != m]


def is_on_left(target, location):
    return distance(target, location) <= 0 and target != location
    # FIXME:see border case f right!


def right(target, ms):
    return [m for m in ms if distance(target, m) > 0 and target != m]


def possible_swap_locations(maint, forced):
    """
    find intervals in forced such that 
    maint.value = 1 and forces.value =1
    can move with forced.value ==1 if it is overlapping
    """
    x = (maint == 1)  # & (forced==1)
    return [Interval(i.start, i.end, 1) for i in steps(x) if i.value == True]


def len_(interval):
    return interval.end-interval.start


def lendiff(interval1, interval2):
    return len_(interval1) - len_(interval2)


def split_and_move(t, locations, f_, ms, fs):
    print("Trying spliting and moving")
    m_overlapping = [m for m in ms if overlap(t, m) and m.value < 1][0]
    index = ms.index(m_overlapping)
    prev = ms[index-1]
    next_ = ms[index+1]
    pl, nl, tl = len_(prev), len_(next_), len_(t)
    foutages = [ft for ft in fs if ft.value < 1]
    findex = foutages.index(t)
    if pl + nl >= tl:
        if (findex == 0 or not overlap(prev, foutages[findex-1])) and (len(foutages) == findex+1 or not overlap(Interval(next_.start, next_.start+tl-pl, 1), foutages[findex+1])):
            locations.remove(next_)
            remaining = Interval(next_.start+tl-pl, next_.end, next_.value)
            locations.append(remaining)
            f_.iloc[t.start:t.end] = 1
            f_.iloc[prev.start:prev.end] = t.value
            f_.iloc[next_.start:next_.start+tl-pl] = t.value
            o = (overlap(t, m_overlapping))

        elif (len(foutages) == findex+1 or not overlap(next_, foutages[findex+1])) and (findex == 0 or not overlap(Inteval(prev.end-(tl-nl), prev.end, 1), foutages[findex-1])):
            locations.remove(prev)
            remaining = Interval(prev.end-(tl-nl), prev.end, prev.value)
            locations.append(remaining)
            f_.iloc[t.start:t.end] = 1
            f_.iloc[next_.start:next_.end] = t.value
            f_.iloc[prev.end-(tl-nl):prev.end] = t.value
        else:
            print("Giving up for", t)
    else:
        print("Giving up for", t)


def moveall(targets, locations, f_, ms, fs):
    # print("Targets->", targets)
    f = f_.copy()
    used = {}
    for t in sorted(targets, key=len_, reverse=True):  # start with biggest
        # print("Possible swaps->", locations)
        locations_ = sorted([l for l in locations if lendiff(l, t) >= 0],
                            key=lambda x: abs(distance(x, t)))
        if not locations_:
            print("No suitable position found for ", t)
            split_and_move(t, locations, f, ms, fs)
            continue

        n = len_(t)
        for l in locations_:
            locations.remove(l)
            locations.append(Interval(l.start+n, l.end, l.value))
            left = is_on_left(t, l)
            if left:
                # set to one,idealy a value from swap locations.
                f.iloc[t.start:t.end] = 1
                f.iloc[l.end-1:l.end-1-n:-1] = t.value  # set to target value
                break
            else:
                # set to one ,idealy a value from swap locations.
                f.iloc[t.start:t.end] = 1
                f.iloc[l.start:l.start+n] = t.value  # set to target value
                break

    return f


def check_asserts(m, f, nf):
    prints(m)
    prints(f)
    prints(nf)
    prints(combine_fo_m(m, nf))
    assert nf.sum() == f.sum()
    print("="*10)


def check_failed_asserts(m, f, nf):
    prints(m)
    prints(f)
    prints(nf)

    assert nf.sum() == f.sum()


def str_to_series(s):
    return pd.Series([int(c) for c in s])


def prints(s):
    print("".join(str(i) for i in s))


def test_moved_fo():
    m = str_to_series("111100011111111")
    f = str_to_series("111100011111111")
    nf = move_fo(m, f)
    check_asserts(m, f, nf)

    m = str_to_series("111100011111111")
    f = str_to_series("111110001111111")
    nf = move_fo(m, f)
    check_asserts(m, f, nf)

    m = str_to_series("111100011111111")
    f = str_to_series("110001111111111")
    nf = move_fo(m, f)
    check_asserts(m, f, nf)

    m = str_to_series("111100011111111")
    f = str_to_series("110001111111111")
    nf = move_fo(m, f)
    check_asserts(m, f, nf)

    m = str_to_series("111100011111111000")
    f = str_to_series("110001111111100011")
    nf = move_fo(m, f)
    check_asserts(m, f, nf)

    m = str_to_series("111100011111111111")
    f = str_to_series("111110000001100000")
    nf = move_fo(m, f)
    check_asserts(m, f, nf)

    m = str_to_series("111100011111111")
    f = str_to_series("111110000001111")
    nf = move_fo(m, f)
    check_asserts(m, f, nf)

    m = str_to_series("111100011000000")
    f = str_to_series("111110000001111")
    nf = move_fo(m, f)
    check_asserts(m, f, nf)

    m = str_to_series("110001111000000")
    f = str_to_series("111000000111111")
    nf = move_fo(m, f)
    check_asserts(m, f, nf)

    m = str_to_series("1111110000111111")
    f = str_to_series("1100111111111000")
    nf = move_fo(m, f)
    check_asserts(m, f, nf)


# test_moved_fo()
