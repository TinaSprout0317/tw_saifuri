"""切り替えレベル・シエン・タイプの総当たり探索。

人が指定するのは「キャラ」と「目標ステータス」だけ。残り
    タイプ / 初期シエン / 切替先シエン / 切替レベル
はここで全部振って、目標に届く組み合わせを探す。

三段構え:
  1. 列挙 + 下界枝刈り  … O(1)/件。到達不能が確定するものを落とす
  2. 貪欲法スクリーニング … 数十 us/件。順位づけ用の推定値を出す
  3. 厳密 DP            … 数秒/件。上位のみ solver_fast で確定させる

貪欲法は最適ではない（実測で最大 14% 悪い）ので、2 の順位は推定でしかない。
最終結果として「達成」と言い切るのは 3 を通ったものだけ。
"""
import tempfile
import time

import numpy as np

import character_parameter as cp
import xien_availability as xa
from solver_fast import (BYTES_PER_CELL, INF, HAVE_NUMBA, Instance,
                         LayerDP, free_disk, mem_budget, mem_info, njit,
                         point_gain, STATUS_KINDS)


# ---------------------------------------------------------------- 前計算
class Ctx:
    """キャラ非依存の前計算（レベルごとのポイント表など）。"""

    def __init__(self, bonus_prob, max_level=310):
        self.max_level = max_level
        tbl20 = cp.random_bonus_table[str(int(round(bonus_prob * 100)))]
        self.bonus = tbl20 * 15 + tbl20[:9]          # Lv2..310
        # cum[L] = Lv2..L で乱数ステータスが上がった回数
        cum = [0] * (max_level + 1)
        for L in range(2, max_level + 1):
            cum[L] = cum[L - 1] + self.bonus[L - 2]
        self.cum = cum
        tp = [0] * (max_level + 1)
        for L in range(2, max_level + 1):
            tp[L] = tp[L - 1] + point_gain(L)
        self.tp = np.array(tp, dtype=np.int32)
        self.max_cost = tp[max_level]
        lof, L = [0] * (self.max_cost + 2), 1
        for c in range(self.max_cost + 1):
            while L < max_level and tp[L] < c:
                L += 1
            lof[c] = L
        self.lof = np.array(lof, dtype=np.int32)

    def base_table(self, ix, nx, cx, idx, init_all):
        """base[L, p] = idx[p] のステータスの Lv L 時点の値（初期値+自動上昇）。"""
        ml = self.max_level
        L = np.arange(ml + 1, dtype=np.int64)
        cum = np.array(self.cum, dtype=np.int64)
        cxc = np.minimum(cx, L)
        cnt_i = np.maximum(cxc - 1, 0)                    # 初期シエンの基本上昇回数
        bon_i = cum[np.maximum(cxc, 0)]                   # 初期シエンの乱数上昇量
        after = np.maximum(L - max(cx, 1), 0)             # 切替後の基本上昇回数
        bon_n = cum[L] - cum[np.minimum(np.maximum(cx, 1), L)]
        auto = np.zeros((ml + 1, 7), dtype=np.int64)
        np.add.at(auto.T, ix[0], cnt_i)
        np.add.at(auto.T, ix[1], bon_i)
        np.add.at(auto.T, nx[0], after)
        np.add.at(auto.T, nx[1], bon_n)
        base = np.empty((ml + 1, len(idx)), dtype=np.int64)
        for p, s in enumerate(idx):
            base[:, p] = init_all[s] + auto[:, s]
        return base

    def auto_at(self, ix, nx, cx, L):
        """Lv L 時点でシエンにより自動上昇した量（7要素）。O(1)。"""
        a = [0] * 7
        c = min(cx, L)
        if c >= 2:
            a[ix[0]] += c - 1
            a[ix[1]] += self.cum[c]
        if L > cx:
            a[nx[0]] += L - max(cx, 1)
            a[nx[1]] += self.cum[L] - self.cum[max(cx, 1)]
        return a


# ---------------------------------------------------------------- 貪欲法
@njit(cache=True)
def _greedy(delta, a, base, tp, lof, mul, max_level, inf):
    """毎回いちばん安いステータスに振る。(振れた回数, 最終コスト, 到達m) を返す。"""
    k = delta.shape[0]
    m = np.zeros(k, np.int64)
    cost = 0
    done = 0
    total = 0
    for i in range(k):
        total += delta[i]
    for _ in range(total):
        L = lof[cost]
        best = inf
        bi = -1
        for i in range(k):
            if m[i] >= delta[i]:
                continue
            LL = L
            v = inf
            while LL <= max_level:
                nc = cost + a[i] + ((base[LL, i] + m[i]) * mul + LL) // 125
                if nc <= tp[LL]:
                    v = nc
                    break
                LL += 1
            if v < best:
                best = v
                bi = i
        if bi < 0:
            break
        cost = best
        m[bi] += 1
        done += 1
    return done, cost, m


# ---------------------------------------------------------------- 候補
class Combo:
    __slots__ = ('ctype', 'ix_name', 'nx_name', 'cx', 'idx', 'delta', 'N',
                 'box', 'lb', 'g_done', 'g_cost', 'exact_cost', 'exact_level',
                 'frontier', 'exact', 'est_sec', 'oversized', 'spilled')

    def __init__(self, ctype, ix_name, nx_name, cx, idx, delta):
        self.ctype, self.ix_name, self.nx_name, self.cx = ctype, ix_name, nx_name, cx
        self.idx, self.delta = idx, delta
        self.N = sum(delta)
        d = sorted(delta)
        self.box = int(np.prod([x + 1 for x in d[:-1]])) if len(d) > 1 else 1
        self.lb = 0
        self.g_done = self.g_cost = -1
        self.exact_cost = self.exact_level = None
        self.frontier = None
        self.exact = False
        self.est_sec = 0.0
        self.oversized = False
        self.spilled = False

    @property
    def names(self):
        return [STATUS_KINDS[i] for i in self.idx]

    def key(self):
        """数値的に等価な組み合わせをまとめるためのキー。"""
        return (self.ctype, tuple(self.idx), tuple(self.delta))


def enumerate_combos(character, target, ctx, cx_values, types=None,
                     mul=5, progress=None):
    """(タイプ, 初期シエン, 切替先, 切替Lv) を列挙し、下界で枝刈りする。"""
    max_level = ctx.max_level
    out, seen = [], {}
    all_types = types or list(cp.character_basic_status[character])
    for ctype in all_types:
        init_all = cp.character_basic_status[character][ctype][0]
        cost_all = cp.character_basic_status[character][ctype][1]
        names, _ = xa.candidates_for(character, ctype, cp.xien_list)
        # 同じ [基本,乱数] ペアは等価なので代表 1 つに畳む
        pairs = {}
        for n in names:
            pairs.setdefault(tuple(cp.xien_list[n]), n)
        reps = list(pairs.values())
        for ixn in reps:
            ix = cp.xien_list[ixn]
            for nxn in reps:
                nx = cp.xien_list[nxn]
                for cx in cx_values:
                    if ixn == nxn and cx != max_level:
                        continue          # 同じシエンなら切替Lvは意味を持たない
                    auto = ctx.auto_at(ix, nx, cx, max_level)
                    idx, delta = [], []
                    for s in range(7):
                        need = target[s] - (init_all[s] + auto[s])
                        if need > 0:
                            idx.append(s)
                            delta.append(need)
                    if not idx:
                        c = Combo(ctype, ixn, nxn, cx, [], [])
                        c.lb = 0
                        out.append(c)
                        continue
                    c = Combo(ctype, ixn, nxn, cx, idx, delta)
                    if c.N > ctx.max_cost:
                        continue          # 1 回 1 ポイント未満はあり得ない
                    # 下界: すべて Lv1・自動上昇なしの最安条件で評価
                    lb = 0
                    for p, s in enumerate(idx):
                        b0 = init_all[s]
                        for t in range(delta[p]):
                            lb += cost_all[s] + ((b0 + t) * mul + 1) // 125
                    if lb > ctx.max_cost:
                        continue
                    c.lb = lb
                    k = c.key()
                    if k in seen:
                        seen[k].append(c)   # 等価な組み合わせは代表だけ解く
                        continue
                    seen[k] = [c]
                    out.append(c)
        if progress:
            progress(f'列挙中: {ctype}')
    return out, seen


def screen(combos, character, ctx, mul=5, progress=None):
    """貪欲法で順位づけ用の推定値を入れる。"""
    for n, c in enumerate(combos):
        if not c.idx:
            c.g_done, c.g_cost = 0, 0
            continue
        init_all = cp.character_basic_status[character][c.ctype][0]
        cost_all = cp.character_basic_status[character][c.ctype][1]
        ix, nx = cp.xien_list[c.ix_name], cp.xien_list[c.nx_name]
        base = ctx.base_table(ix, nx, c.cx, c.idx, init_all)
        done, cost, _m = _greedy(np.array(c.delta, dtype=np.int64),
                             np.array([cost_all[s] for s in c.idx], dtype=np.int64),
                             base, ctx.tp, ctx.lof, mul, ctx.max_level, int(INF))
        c.g_done, c.g_cost = int(done), int(cost)
        if progress and n % 200 == 0:
            progress(f'スクリーニング {n}/{len(combos)}')
    return combos


def solve_exact(character, target, c, bonus_prob, mul=5, max_level=310,
                spill_dir=None, should_stop=None):
    """1 組み合わせを厳密 DP で解き、結果を c に書き戻す。

    箱が実メモリに収まらない場合は spill_dir（既定は一時フォルダ）に
    memmap で退避して計算する。遅いが計算はできる。
    """
    inst = Instance(character, c.ctype, target, c.ix_name, c.nx_name,
                    c.cx, bonus_prob, max_level=max_level, mul=mul)
    dp = LayerDP(inst)
    spill = (spill_dir or tempfile.gettempdir()) if needs_spill(c) else None
    c.spilled = spill is not None
    try:
        cost, _ = dp.run(progress=False, spill_dir=spill, should_stop=should_stop)
    finally:
        dp.cleanup()
    if cost is None:
        return None                        # 中断された（結果は書き込まない）
    if cost < INF:
        c.exact_cost, c.exact_level = int(cost), inst.level_of_cost[cost]
    else:
        c.exact_cost = None
        if dp.frontier:
            s, cc, m = dp.frontier
            c.frontier = (s, inst.level_of_cost[cc],
                          [(inst.names[i], inst.delta[i] - m[i])
                           for i in range(inst.k) if inst.delta[i] > m[i]])
    return c


# 厳密 DP の実測スループット（セル遷移/秒）。i9-11900KF/16スレッドで約 2.6e8。
CELLS_PER_SEC = 2.6e8

# ディスク退避時の減速の初期見積り。実行中に実測で補正されるので目安でよい。
SPILL_SLOWDOWN = 3.0


def fmt_sec(x):
    """秒を読みやすく。"""
    if x < 90:
        return f'{x:.0f}秒'
    if x < 5400:
        return f'{x / 60:.0f}分'
    return f'{x / 3600:.1f}時間'


_RAM_CELLS = None            # 探索開始時に固定する（実行中の揺れで判定を変えない）


def set_mem_budget(limit_bytes=None):
    """作業配列に使える上限セル数を決めて返す。"""
    global _RAM_CELLS
    _RAM_CELLS = int((limit_bytes or mem_budget()) / BYTES_PER_CELL)
    return _RAM_CELLS


def ram_cells():
    return _RAM_CELLS if _RAM_CELLS is not None else set_mem_budget()


def needs_spill(c):
    return c.box > ram_cells()


def est_seconds(c):
    """厳密 DP にかかる秒数の見積り。仕事量は N x レイヤー箱サイズに比例する。"""
    t = c.N * c.box / CELLS_PER_SEC
    return t * SPILL_SLOWDOWN if needs_spill(c) else t


def search(character, target, bonus_prob=0.50, mul=5, cx_step=10, top_k=25,
           refine=True, progress=None, should_stop=None, max_level=310,
           time_budget=120.0, keep_estimates=150, max_box=8_000_000_000,
           mem_limit_bytes=None, spill_dir=None, types=None):
    """全体のオーケストレーション。結果を良い順に返す。

    time_budget=None で「無制限（推測なし）」モード。枝刈りを通った候補を
    すべて厳密 DP にかける。ただし箱が max_box セルを超えるものは、必要メモリが
    現実的でないので除外する（2 レイヤー x 4 バイトが常時必要）。
    """
    def rep(msg):
        if progress:
            progress(msg)

    def stop():
        return should_stop is not None and should_stop()

    t0 = time.time()
    cells = set_mem_budget(mem_limit_bytes)
    avail, total, commit = mem_info()
    rep(f'メモリ 空き{avail / 1e9:.0f}GB/{total / 1e9:.0f}GB '
        f'コミット残{commit / 1e9:.0f}GB → 作業配列 '
        f'{cells * BYTES_PER_CELL / 1e9:.1f}GB まで実メモリ、超過分はディスク退避')
    ctx = Ctx(bonus_prob, max_level)
    cx_values = sorted(set(list(range(1, max_level + 1, cx_step)) + [max_level]))
    rep(f'切替Lv {len(cx_values)} 点で列挙中...')
    combos, seen = enumerate_combos(character, target, ctx, cx_values, mul=mul,
                                    types=types, progress=progress)
    rep(f'枝刈り後 {len(combos)} 件をスクリーニング中...')
    if stop():
        return [], ctx, {}
    screen(combos, character, ctx, mul=mul, progress=progress)

    # 不足手数の少ない順 → 推定コストの安い順に厳密化する
    combos.sort(key=lambda c: (c.N - c.g_done, c.g_cost))
    oversized = []
    if time_budget is None:
        # 無制限モード: 枝刈りを通ったものは全部厳密化する。
        # 実メモリに載らないものはディスクへ退避するので、退避先の空きも見る。
        sdir = spill_dir or tempfile.gettempdir()
        disk_cells = free_disk(sdir) * 0.8 / BYTES_PER_CELL
        cap = min(max_box, max(ram_cells(), disk_cells))
        todo = [c for c in combos if c.box <= cap]
        oversized = [c for c in combos if c.box > cap]
        total = sum(est_seconds(c) for c in todo)
        rep(f'厳密DP {len(todo)} 件を全件計算します（推定 {fmt_sec(total)}）'
            + (f' / メモリ超過で除外 {len(oversized)} 件' if oversized else ''))
    else:
        todo, budget = [], time_budget
        for c in combos:
            if len(todo) >= top_k:
                break
            est = est_seconds(c)
            if est > budget:
                continue                  # 予算に収まるものだけ拾う（1件目も例外にしない）
            todo.append(c)
            budget -= est
        if not todo:
            cheapest = min(est_seconds(c) for c in combos) if combos else 0.0
            rep(f'厳密DPできる組がありません（最短でも約 {fmt_sec(cheapest)} 必要）。'
                f'貪欲法の推定値のみ表示します')
        else:
            rep(f'厳密DP {len(todo)} 件 (推定 {fmt_sec(time_budget - budget)})')

    # 残り時間は実測で補正する。CPU の速さもディスク退避の減速も機械ごとに
    # 違うので、静的な見積りだけだと外れる。
    left = sum(est_seconds(c) for c in todo)
    spent_est = spent_real = 0.0
    t_dp = time.time()
    for n, c in enumerate(todo):
        if stop():
            break
        ratio = (spent_real / spent_est) if spent_est > 1 else 1.0
        eta = f'  残り約{fmt_sec(left * ratio)}' if left * ratio > 5 else ''
        rep(f'厳密DP {n + 1}/{len(todo)}  経過{fmt_sec(time.time() - t_dp)}{eta}'
            f'  (切替Lv{c.cx}, {c.ix_name}->{c.nx_name}'
            + ('・ディスク' if needs_spill(c) else '') + ')')
        t1 = time.time()
        if solve_exact(character, target, c, bonus_prob, mul=mul,
                       max_level=max_level, spill_dir=spill_dir,
                       should_stop=should_stop) is None:
            break                          # 中断
        c.exact = True                     # 解けたものだけ厳密扱いにする
        e = est_seconds(c)
        spent_est += e
        spent_real += time.time() - t1
        left -= e

    # いちばん良かった組の周辺を切替Lv 1 刻みで詰める
    best = min((c for c in todo if c.exact_cost is not None),
               key=lambda c: c.exact_cost, default=None)
    if (refine and best is not None and cx_step > 1 and not stop()
            and best.ix_name != best.nx_name):   # 同シエンなら切替Lvは無意味
        lo, hi = max(1, best.cx - cx_step), min(max_level, best.cx + cx_step)
        rep(f'切替Lv {lo}..{hi} を 1 刻みで再探索中...')
        fine, _ = enumerate_combos(character, target, ctx, list(range(lo, hi + 1)),
                                   types=[best.ctype], mul=mul)
        fine = [c for c in fine
                if c.ix_name == best.ix_name and c.nx_name == best.nx_name]
        already = {(c.ctype, c.ix_name, c.nx_name, c.cx) for c in todo}
        fine = [c for c in fine
                if (c.ctype, c.ix_name, c.nx_name, c.cx) not in already]
        for n, c in enumerate(fine):
            if stop():
                break
            rep(f'切替Lv詰め {n + 1}/{len(fine)}')
            if solve_exact(character, target, c, bonus_prob, mul=mul,
                           max_level=max_level, spill_dir=spill_dir,
                           should_stop=should_stop) is None:
                break
            c.exact = True
        todo = todo + fine

    # 厳密化したものと、貪欲法の推定しか無いものを両方返す。
    # 推定行も残すのは、6〜7 ステータスに手動振りが要る目標だと箱が爆発して
    # 厳密 DP がそもそも回らないため（そこを黙って落とすと何も出なくなる）。
    # 中止された場合、todo の後半は未計算のまま。exact を立てていないので
    # ここでは推定側に回る（結果が無いのに厳密扱いにすると rank で落ちる）。
    exact_keys = {(c.ctype, c.ix_name, c.nx_name, c.cx) for c in todo if c.exact}
    rest = []
    if time_budget is None:
        keep_estimates = 0
        for c in oversized:
            c.exact = False
            c.est_sec = est_seconds(c)
            c.oversized = True
            rest.append(c)
    for c in combos:
        k = (c.ctype, c.ix_name, c.nx_name, c.cx)
        if k in exact_keys:
            continue
        c.exact = False
        c.est_sec = est_seconds(c)
        rest.append(c)
        if len(rest) >= keep_estimates:
            break

    done, seen_key = [], set()
    for c in todo + rest:
        if c.exact and c.exact_cost is None and c.frontier is None:
            continue                       # 中止で未計算のまま
        k = (c.ctype, c.ix_name, c.nx_name, c.cx)
        if k in seen_key:
            continue
        seen_key.add(k)
        if not hasattr(c, 'est_sec'):
            c.est_sec = est_seconds(c)
        done.append(c)
    # 達成 > 厳密で不可 > 推定 の順、同格ならコスト/不足手数の良い順
    def rank(c):
        if c.exact and c.exact_cost is not None:
            return (0, c.exact_cost, 0)
        if c.exact and c.frontier:
            return (1, c.N - c.frontier[0], c.g_cost)
        return (2, c.N - c.g_done, c.g_cost)
    done.sort(key=rank)
    rep(f'完了 ({time.time() - t0:.1f}s)')
    return done, ctx, seen


if __name__ == '__main__':
    import config
    def pr(m): print(' ', m, flush=True)
    res, ctx, seen = search(config.character, config.target_status,
                            config.bonus_prob, mul=5, cx_step=20, top_k=8,
                            progress=pr)
    print(f'\n{config.character}  目標 {config.target_status}')
    for c in res[:15]:
        if c.exact_cost is not None:
            print(f'  達成 {c.ctype:6s} {c.ix_name}->{c.nx_name} @Lv{c.cx:3d}  '
                  f'cost={c.exact_cost} 到達Lv{c.exact_level}')
        else:
            print(f'  不可 {c.ctype:6s} {c.ix_name}->{c.nx_name} @Lv{c.cx:3d}  '
                  f'{c.frontier[0]}/{c.N}手 不足={c.frontier[2]}')


def quick_check(character, target, bonus_prob=0.50, mul=5, cx_step=30,
                max_level=310, types=None):
    """入力中に出す「だいたい届くか」の速報。

    貪欲法が目標に届いたなら、その手順自体が実行可能な計画なので
    「到達可能」と断言できる（厳密探索ではさらに良くなる可能性がある）。
    逆に届かなかった場合は断言できない。貪欲法は最適ではないので、
    厳密探索なら届くことがある。
    """
    out = {'state': 'none', 'need': 0, 'nstat': 0, 'level': None,
           'short': [], 'best': None}
    types = cp.character_basic_status.get(character)
    if not types:
        return out
    ctx = Ctx(bonus_prob, max_level)
    cxs = sorted(set(list(range(1, max_level + 1, cx_step)) + [max_level]))
    combos, _ = enumerate_combos(character, target, ctx, cxs, mul=mul, types=types)
    if not combos:
        out['state'] = 'auto'          # 自動上昇だけで目標を満たす
        return out
    screen(combos, character, ctx, mul=mul)
    best = min(combos, key=lambda c: (c.N - c.g_done, c.g_cost))
    out['need'] = best.N
    out['nstat'] = len(best.idx)
    out['best'] = best
    if best.g_done == best.N:
        out['state'] = 'ok'
        out['level'] = int(ctx.lof[min(best.g_cost, ctx.max_cost)])
        return out

    out['state'] = 'ng'
    out['done'] = best.g_done
    # 不足の内訳は、その組をもう一度貪欲法で走らせて到達 m を見る
    init_all = cp.character_basic_status[character][best.ctype][0]
    cost_all = cp.character_basic_status[character][best.ctype][1]
    ix, nx = cp.xien_list[best.ix_name], cp.xien_list[best.nx_name]
    base = ctx.base_table(ix, nx, best.cx, best.idx, init_all)
    _d, _c, m = _greedy(np.array(best.delta, dtype=np.int64),
                        np.array([cost_all[s] for s in best.idx], dtype=np.int64),
                        base, ctx.tp, ctx.lof, mul, ctx.max_level, int(INF))
    out['short'] = [(STATUS_KINDS[best.idx[i]], int(best.delta[i] - m[i]))
                    for i in range(len(best.idx)) if best.delta[i] > m[i]]
    return out


def warmup():
    """numba のコンパイルを先に済ませる（初回の速報が数十秒待ちになるのを防ぐ）。"""
    try:
        quick_check('ノクターン', [5, 2, 1, 2, 1, 30, 2], 0.50, cx_step=150)
    except Exception:
        pass


def quick_exact(character, target, bonus_prob=0.50, mul=5, cx_step=20,
                budget=8.0, top=8, max_level=310, should_stop=None, types=None):
    """速報の裏取り。有望な上位数件だけを厳密 DP にかける。

    貪欲法は境界付近で弱く、実際は届く目標を「数不足」と誤判定する。
    ここで 1 件でも到達できれば「到達可能」を確定できる（存在証明）。
    届かなかった場合は「調べた範囲では」としか言えない点に注意。
    """
    out = {'state': 'unknown', 'level': None, 'checked': 0, 'best_short': None}
    if character not in cp.character_basic_status:
        return out
    set_mem_budget()
    ctx = Ctx(bonus_prob, max_level)
    cxs = sorted(set(list(range(1, max_level + 1, cx_step)) + [max_level]))
    combos, _ = enumerate_combos(character, target, ctx, cxs, mul=mul, types=types)
    if not combos:
        out['state'] = 'auto'
        return out
    screen(combos, character, ctx, mul=mul)
    combos.sort(key=lambda c: (c.N - c.g_done, c.g_cost))

    spent = 0.0
    for c in combos[:top]:
        if should_stop is not None and should_stop():
            return out
        e = est_seconds(c)
        if out['checked'] and spent + e > budget:
            break                       # 予算超過（1件目は必ず試す）
        if solve_exact(character, target, c, bonus_prob, mul=mul,
                       max_level=max_level, should_stop=should_stop) is None:
            return out                  # 中断
        spent += e
        out['checked'] += 1
        if c.exact_cost is not None:
            out['state'] = 'ok'
            out['level'] = c.exact_level
            out['combo'] = c
            return out
        if c.frontier and (out['best_short'] is None
                           or c.N - c.frontier[0] < out['best_short']):
            out['best_short'] = c.N - c.frontier[0]
            out['short'] = c.frontier[2]
    out['state'] = 'ng' if out['checked'] else 'unknown'
    return out
