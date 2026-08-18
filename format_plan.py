"""振り順を「レベル帯ごとの振り分け表」に整形する（シミュレータのログ形式）。

■ 行の意味（再生時の解釈）
    LV a - b : STAT 極振り Nup   a〜b の各レベルで STAT を「振れるだけ振る」。
                                 Nup は結果として上がった量。
    LV n     : STAT              そのレベルで STAT をちょうど 1 回だけ振る
    LV n     : STAT * k          そのレベルで STAT をちょうど k 回だけ振る
    LV n     : A, B * 2          そのレベルで A を1回、B を2回（いずれも正確な回数）
    LV a - b : （空）            その間は何も振らない

「極振り」は指示であって記録ではない。最適解は「振れるのに振らずに貯めて、
後で高いステを買う」手を使うことがあり、その手は極振りでは表現できない。
そこで各レベルについて「素直に振れるだけ振った場合」と実際の手を突き合わせ、
一致するレベルの連続だけを極振りにまとめ、一致しないレベルは回数を明示する。
"""
import unicodedata

import character_parameter as cp

STATUS_KINDS = ['STAB', 'HACK', 'INT', 'DEF', 'MR', 'DEX', 'AGI']
LABEL_W = 13

# シミュレータはシエン名を半角カタカナで持っている（例: ﾃﾞｽｻｲｽﾞ）。
_FULL = ('アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホ'
         'マミムメモヤユヨラリルレロワヲンァィゥェォッャュョー')
_HALF = ('ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎ'
         'ﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝｧｨｩｪｫｯｬｭｮｰ')
_K2H = dict(zip(_FULL, _HALF))

# カタカナの全角/半角で吸収できない差異があればここに書く。
LOG_NAME = {}


def to_log_name(name):
    """シエン名をシミュレータのログ表記（半角カタカナ）に直す。"""
    if name in LOG_NAME:
        return LOG_NAME[name]
    out = []
    for ch in name:
        d = unicodedata.decomposition(ch)
        if d and not d.startswith('<'):          # 濁点・半濁点つきカタカナ
            p = d.split()
            base = chr(int(p[0], 16))
            mark = ''
            if len(p) > 1:
                mark = 'ﾞ' if chr(int(p[1], 16)) == '\u3099' else 'ﾟ'
            out.append(_K2H.get(base, base) + mark)
        else:
            out.append(_K2H.get(ch, ch))
    return ''.join(out)


def _xien_line(name):
    b, r = cp.xien_list[name]
    return (f'{"Xien":<{LABEL_W}}: {to_log_name(name)} '
            f'{STATUS_KINDS[b]}/{STATUS_KINDS[r]}')


def _lv_head(a, b=None):
    if b is None or a == b:
        return f'LV {a:3d}' + ' ' * 7 + ':'
    return f'LV {a:3d} - {b:3d} :'


def _exact(stats):
    """正確な回数で書く（順序は最初に出た順）。"""
    seen = []
    for s in stats:
        if s not in seen:
            seen.append(s)
    return ', '.join(f'{s} * {stats.count(s)}' if stats.count(s) > 1 else s
                     for s in seen)


def build(inst, steps, character, ctype, bonus_prob, initial_xien, next_xien,
          change_xien):
    ml, mul, tp = inst.max_level, inst.mul, inst.total_points

    # ---- レベルごとの手と、そのレベルに入る直前の状態を復元
    per = {L: [] for L in range(1, ml + 1)}
    before = {}
    cost, m, si = 0, [0] * inst.k, 0
    for L in range(2, ml + 1):
        before[L] = (cost, list(m))
        while si < len(steps) and steps[si][0] == L:
            name = steps[si][1]
            i = inst.names.index(name)
            cost += inst.a[i] + ((inst.base[L][i] + m[i]) * mul + L) // 125
            m[i] += 1
            per[L].append(name)
            si += 1
    final_cost = cost

    def greedy_n(L, stat):
        """レベル L で stat を振れるだけ振ったら何回振れるか。"""
        i = inst.names.index(stat)
        c0, m0 = before[L]
        c, mi, n = c0, m0[i], 0
        while True:
            step = inst.a[i] + ((inst.base[L][i] + mi) * mul + L) // 125
            if c + step > tp[L]:
                return n
            c += step
            mi += 1
            n += 1

    def is_greedy(L, stat):
        """そのレベルの実際の手が「stat を振れるだけ振った」と同じか。"""
        got = per[L]
        return all(g == stat for g in got) and len(got) == greedy_n(L, stat)

    # ---- 行の組み立て
    lines = [_lv_head(1), f'{"Bonus":<{LABEL_W}}: {int(round(bonus_prob * 100))}%',
             _xien_line(initial_xien)]
    switch_at = change_xien + 1
    emitted = initial_xien == next_xien or switch_at > ml

    run_stat = run_start = run_last = None
    run_n = 0
    pending = []

    def close():
        nonlocal run_stat, run_start, run_last, run_n
        if run_stat is not None:
            lines.append(f'{_lv_head(run_start, run_last)} '
                         f'{run_stat} 極振り {run_n}up')
            run_stat = run_start = run_last = None
            run_n = 0

    def flush():
        nonlocal pending
        if not pending:
            return
        a = pending[0]
        for i, L in enumerate(pending + [None]):
            if L is None or (i and L != pending[i - 1] + 1):
                lines.append(_lv_head(a, pending[i - 1]) + ' ')
                if L is not None:
                    a = L
        pending = []

    for L in range(2, ml + 1):
        if not emitted and L >= switch_at:
            close()
            flush()
            lines.append(_xien_line(next_xien))
            emitted = True

        got = per[L]

        # 走行中の帯を延長できるか（空レベルも「0回振れる」なら延長可）
        if run_stat is not None and is_greedy(L, run_stat):
            run_n += len(got)
            run_last = L
            continue
        close()

        if not got:
            pending.append(L)
            continue

        stat = got[0]
        if len(set(got)) == 1 and is_greedy(L, stat):
            # 直前の空レベルは、そこで 0 回しか振れないなら帯に含められる
            start = L
            while pending and greedy_n(pending[-1], stat) == 0:
                start = pending.pop()
            flush()
            run_stat, run_start, run_last, run_n = stat, start, L, len(got)
        else:
            flush()
            lines.append(f'{_lv_head(L)} {_exact(got)}')

    close()
    flush()

    # ---- フッタ
    tbl = cp.random_bonus_table[str(int(round(bonus_prob * 100)))]
    tbl = tbl * 15 + tbl[:9]
    lines.append(f'{character}/{ctype} Bonus: {sum(tbl[:ml - 1])} / {ml - 1} '
                 f'Point: {inst.max_cost - final_cost}')

    init_all = cp.character_basic_status[character][ctype][0]
    ix, nx = cp.xien_list[initial_xien], cp.xien_list[next_xien]
    auto = [0] * 7
    for i, b in enumerate(tbl[:ml - 1]):
        lv = i + 2
        src = ix if lv <= change_xien else nx
        auto[src[0]] += 1
        auto[src[1]] += b
    final = [init_all[i] + auto[i] for i in range(7)]
    for p, i in enumerate(inst.idx):
        final[i] += m[p]
    lines.append(f'LV:{ml} ' + ' '.join(f'{STATUS_KINDS[i]} {final[i]}'
                                        for i in range(7)))
    return '\n'.join(lines)
