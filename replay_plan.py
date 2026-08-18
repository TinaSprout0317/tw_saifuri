"""振り分け表を読み直して再生し、元の計画を復元できるか検証する。

format_plan.build() の出力が正しく読み戻せるかは「同じ結果になるか」で確かめる。
2 通りの解釈を試せる:

  strict : 「Nup」を上限として、その帯でそのステを N 回まで上げる（記録の再現）
  greedy : 「極振り」を指示と解釈し、その帯は買えるだけ買う
           → ポイントを貯める手を表現できないので、貯めが要る計画ではずれる

strict で元の計画に一致すれば、表は情報を落としていない。
"""
import re

import character_parameter as cp
from solver_fast import point_gain

STATUS_KINDS = ['STAB', 'HACK', 'INT', 'DEF', 'MR', 'DEX', 'AGI']


def parse(txt):
    """[(開始Lv, 終了Lv, [(ステ名, 回数), ...]), ...] を返す。"""
    out = []
    for ln in txt.split('\n'):
        m = re.match(r'LV\s+(\d+)(?:\s+-\s+(\d+))?\s+:\s*(.*)', ln)
        if not m:
            continue
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        body = m.group(3).strip()
        if not body:
            out.append((lo, hi, []))
            continue
        mm = re.match(r'(\w+) 極振り (\d+)up$', body)
        if mm:
            out.append((lo, hi, [(mm.group(1), None)]))   # None = 極振り
            continue
        items = []
        for p in body.split(','):
            q = re.match(r'(\w+)(?:\s*\*\s*(\d+))?$', p.strip())
            items.append((q.group(1), int(q.group(2) or 1)))
        out.append((lo, hi, items))
    return out


def run(txt, character, ctype, bonus_prob, ixn, nxn, cx, mul=5):
    """再生して (最終7ステ, 残ポイント, 消費コスト) を返す。

    「極振り」の行はその帯の各レベルで振れるだけ振る。
    それ以外の行は書かれた回数だけ正確に振る。
    """
    init = cp.character_basic_status[character][ctype][0]
    a = cp.character_basic_status[character][ctype][1]
    ix, nx = cp.xien_list[ixn], cp.xien_list[nxn]
    tbl = cp.random_bonus_table[str(int(round(bonus_prob * 100)))]
    tbl = tbl * 15 + tbl[:9]

    st, cost, tp = list(init), 0, 0
    for lo, hi, items in parse(txt):
        if hi < 2:
            continue                       # 先頭の "LV 1 :" 行
        left = {s: n for s, n in items}
        for L in range(max(lo, 2), hi + 1):
            src = ix if L <= cx else nx
            st[src[0]] += 1                # シエンによる自動上昇
            st[src[1]] += tbl[L - 2]
            tp += point_gain(L)
            for stat, n in items:          # 書かれた順に振る
                i = STATUS_KINDS.index(stat)
                while True:
                    if n is not None and left[stat] <= 0:
                        break              # 回数指定は使い切ったら終わり
                    c = a[i] + ((st[i] * mul + L) // 125)
                    if cost + c > tp:
                        break              # 極振りは買えなくなるまで
                    cost += c
                    st[i] += 1
                    if n is not None:
                        left[stat] -= 1
    return st, tp - cost, cost
