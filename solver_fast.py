"""再振り最適化ソルバ（反対角線レイヤー DP 版）

従来 test_5param.py は k 次元格子 (Δ+1 の総積) を丸ごと共有メモリに確保していたが、
遷移は必ず Σm を +1 するので、必要なのは直前レイヤー 1 枚だけ。
さらに「レベルはコストの純関数」なので、セルに持つ値はコスト 1 本で足りる。

  従来 : 499,905,000 セル x uint16 x 2ch = 約 2.0 GB
  本版 : 3,105,000 セル x int32 x 2枚    = 約  25 MB

結果は従来アルゴリズムと厳密に一致する（近似ではない）。

使い方:
    python solver_fast.py            # config.py の設定で最小コストを求める
    python solver_fast.py --path     # 振り順も復元する（一時ファイル ~1.3GB を使用）
"""
import argparse
import ctypes
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import character_parameter as cp

INF = np.int32(1 << 29)
STATUS_KINDS = ['STAB', 'HACK', 'INT', 'DEF', 'MR', 'DEX', 'AGI']

if getattr(sys, 'frozen', False):
    # PyInstaller で固めた場合、ソース横の __pycache__ に書けないため
    os.environ.setdefault(
        'NUMBA_CACHE_DIR',
        os.path.join(os.environ.get('LOCALAPPDATA') or tempfile.gettempdir(),
                     'tw_saifuri', 'numba_cache'))

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                                   # numba が無ければ numpy 版で動く
    HAVE_NUMBA = False

    def njit(*a, **k):
        return lambda f: f
    prange = range


@njit(cache=True, inline='always')
def _step_scalar(cost, i, m, basecol, tp, lof, a, mul, max_level, inf):
    """コスト cost・stat i を m 個振った状態から +1 したときの新コスト。"""
    L = lof[cost]
    a_i = a[i]
    while L <= max_level:
        ac = a_i + ((basecol[i, L] + m) * mul + L) // 125
        nc = cost + ac
        if nc <= tp[L]:
            return nc
        L += 1
    return inf


@njit(cache=True, inline='always')
def _step_tab(cost, i, m, acost, tp, lof, max_level, inf):
    """_step_scalar と同じだが、割り振りコストを事前計算表から引く。"""
    L = lof[cost]
    while L <= max_level:
        nc = cost + acost[i, L, m]
        if nc <= tp[L]:
            return nc
        L += 1
    return inf


@njit(cache=True, parallel=True)
def _layer_kernel3(prev, cur, arg, s, dbig, dlast, strides, OC, Souter,
                   big, others, acost, tp, lof, max_level, inf):
    """生存セルだけを走査する版。

    最内軸（stride 1、いちばん Δ の大きい軸）を選んでおくと、外側座標を固定した
    ときの生存範囲 s-dbig <= Σ <= s は最内軸上の連続区間になる。よって
    「生きているセルだけ」を過不足なく回せる（従来は 62%が空振りだった）。
    外側の直積（数万要素）を prange で分割するので負荷分散も細かくなる。
    """
    nouter = Souter.shape[0]
    npo = OC.shape[1]                       # 外側軸の本数 = nother-1
    jlast = others[npo]                     # 最内軸に対応する stat
    for o in prange(nouter):
        base = o * dlast
        psum = Souter[o]
        lo = s - dbig - psum
        if lo < 0:
            lo = 0
        hi = s - psum
        if hi > dlast - 1:
            hi = dlast - 1
        # 生存区間の外側を INF で潰す（バッファを使い回すため必須）
        for x in range(0, min(lo, dlast)):
            cur[base + x] = inf
            arg[base + x] = 255
        for x in range(max(hi + 1, 0), dlast):
            cur[base + x] = inf
            arg[base + x] = 255

        for x in range(lo, hi + 1):
            f = base + x
            mbig = s - psum - x
            best = inf
            bestj = 255
            # 暗黙軸
            if mbig >= 1:
                pc = prev[f]
                if pc < inf:
                    v = _step_tab(pc, big, mbig - 1, acost, tp, lof,
                                  max_level, inf)
                    if v < best:
                        best = v
                        bestj = big
            # 最内軸（stride 1）
            if x > 0:
                pc = prev[f - 1]
                if pc < inf:
                    v = _step_tab(pc, jlast, x - 1, acost, tp, lof,
                                  max_level, inf)
                    if v < best:
                        best = v
                        bestj = jlast
            # 外側の各軸
            for p in range(npo):
                xp = OC[o, p]
                if xp == 0:
                    continue
                pc = prev[f - strides[p]]
                if pc >= inf:
                    continue
                j = others[p]
                v = _step_tab(pc, j, xp - 1, acost, tp, lof, max_level, inf)
                if v < best:
                    best = v
                    bestj = j
            cur[f] = best
            arg[f] = bestj


@njit(cache=True, parallel=True)
def _layer_kernel2(prev, cur, arg, s, S, strides, dims, big, others,
                   dbig, acost, tp, lof, max_level, inf):
    """_layer_kernel の最適化版。

    - 座標をオドメータで持ち回り、セルあたり 4 回の整数除算を消す
    - 割り振りコストは (stat, level, m) の事前計算表から 1 回のロードで引く
    - 外側軸を prange で分割（スラブ単位なので各スレッドは連続メモリを触る）
    """
    nother = dims.shape[0]
    st0 = strides[0]
    for i0 in prange(dims[0]):
        coord = np.zeros(nother, np.int64)
        coord[0] = i0
        f0 = i0 * st0
        for t in range(st0):
            f = f0 + t
            mbig = s - S[f]
            if mbig < 0 or mbig > dbig:
                cur[f] = inf
                arg[f] = 255
            else:
                best = inf
                bestj = 255
                if mbig >= 1:
                    pc = prev[f]
                    if pc < inf:
                        v = _step_tab(pc, big, mbig - 1, acost, tp, lof,
                                      max_level, inf)
                        if v < best:
                            best = v
                            bestj = big
                for p in range(nother):
                    if coord[p] == 0:
                        continue
                    pc = prev[f - strides[p]]
                    if pc >= inf:
                        continue
                    j = others[p]
                    v = _step_tab(pc, j, coord[p] - 1, acost, tp, lof,
                                  max_level, inf)
                    if v < best:
                        best = v
                        bestj = j
                cur[f] = best
                arg[f] = bestj
            # オドメータを 1 進める（軸 1..nother-1）
            for p in range(nother - 1, 0, -1):
                coord[p] += 1
                if coord[p] < dims[p]:
                    break
                coord[p] = 0


@njit(cache=True, parallel=True)
def _layer_kernel(prev, cur, arg, s, S, strides, dims, big, others,
                  dbig, basecol, tp, lof, a, mul, max_level, inf):
    """レイヤー s-1 (prev) からレイヤー s (cur/arg) を 1 パスで作る。

    セルごとに独立なので prange で並列化できる。死にセルは比較 1 回で抜ける。
    """
    n_other = others.shape[0]
    for f in prange(prev.shape[0]):
        mbig = s - S[f]
        if mbig < 0 or mbig > dbig:
            cur[f] = inf
            arg[f] = 255
            continue
        best = inf
        bestj = 255

        # 暗黙軸を +1（箱の座標は不変、前レイヤーの m_big = mbig-1）
        if mbig >= 1:
            pc = prev[f]
            if pc < inf:
                v = _step_scalar(pc, big, mbig - 1, basecol, tp, lof,
                                 a, mul, max_level, inf)
                if v < best:
                    best = v
                    bestj = big

        # 箱の各軸を +1（前レイヤーの m_big は mbig のまま）
        for p in range(n_other):
            xp = (f // strides[p]) % dims[p]
            if xp == 0:
                continue
            pc = prev[f - strides[p]]
            if pc >= inf:
                continue
            j = others[p]
            v = _step_scalar(pc, j, xp - 1, basecol, tp, lof,
                             a, mul, max_level, inf)
            if v < best:
                best = v
                bestj = j

        cur[f] = best
        arg[f] = bestj


def point_gain(level):
    """LvUp 時の POINT 上昇値"""
    for lo, hi, p in ((1, 6, 2), (7, 22, 3), (23, 48, 4), (49, 80, 5),
                      (81, 129, 6), (130, 175, 7), (176, 235, 8),
                      (236, 265, 9), (266, 290, 12), (291, 310, 15)):
        if lo <= level <= hi:
            return p
    return 0


def mem_info():
    """(空き物理, 合計物理, コミット残) をバイトで返す。

    コミット残（= コミット上限 - コミット済み）も見るのは、ページファイルを
    小さく固定している環境だと物理メモリが空いていても割り当てに失敗するため。
    """
    try:
        class _MS(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong),
                        ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong),
                        ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong),
                        ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        m = _MS()
        m.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return (int(m.ullAvailPhys), int(m.ullTotalPhys),
                int(m.ullAvailPageFile))
    except Exception:
        pass
    try:                                   # Windows 以外
        av = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        tot = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        return av, tot, av
    except (ValueError, AttributeError, OSError):
        return 2 << 30, 4 << 30, 2 << 30   # 分からなければ控えめに


def avail_bytes():
    return mem_info()[0]


def mem_budget(frac=0.50, headroom=(4 << 30)):
    """作業配列に使ってよいバイト数。

    次の 3 つの小さいほうを採る。
      - 空き物理メモリの frac 倍
      - 搭載メモリ - headroom（他のアプリを圧迫しないための保険）
      - コミット残の frac 倍（ページファイルが小さい環境で確保に失敗しないため）
    """
    avail, total, commit = mem_info()
    return max(256 << 20, min(int(avail * frac),
                              max(0, total - headroom),
                              int(commit * frac)))


def free_disk(path):
    """path のあるドライブの空き容量（バイト）。"""
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


BYTES_PER_CELL = 9          # prev(int32) + cur(int32) + arg(uint8)


def _alloc(n, dtype, spill_dir, tag):
    """spill_dir があればディスク上に確保する（memmap）。"""
    if spill_dir is None:
        return np.empty(n, dtype=dtype), None
    path = os.path.join(spill_dir, f'twdp_{tag}_{os.getpid()}.dat')
    return np.memmap(path, dtype=dtype, mode='w+', shape=(n,)), path


class Instance:
    """1 ケースぶんの探索問題。

    stat i を手動で +1 するコスト:
        a[i] + floor((stat_value * MUL + L) / 125)
        stat_value = base[L][i] + m[i]      (base = 初期値 + シエン自動上昇)
    """

    def __init__(self, character, ctype, target_status, initial_xien, next_xien,
                 change_xien, bonus_prob, max_level=310, mul=5):
        init_all = cp.character_basic_status[character][ctype][0]
        cost_all = cp.character_basic_status[character][ctype][1]
        ix = cp.xien_list[initial_xien]
        nx = cp.xien_list[next_xien]

        tbl20 = cp.random_bonus_table[str(int(bonus_prob * 100))]
        tbl = tbl20 * 15 + tbl20[:9]              # Lv2..310 の乱数ボーナス

        auto = [0] * 7
        auto_all = [list(auto), list(auto)]        # index 1 = Lv1
        for i, bonus in enumerate(tbl):
            lv = i + 2
            src = ix if lv <= change_xien else nx
            auto[src[0]] += 1
            auto[src[1]] += bonus
            auto_all.append(list(auto))

        self.max_level = max_level
        self.mul = mul
        self.idx = [s for s in range(7)
                    if target_status[s] - (init_all[s] + auto_all[max_level][s]) > 0]
        self.k = len(self.idx)
        if self.k == 0:
            raise ValueError('手動で振る必要のあるステータスがありません')
        self.a = [cost_all[s] for s in self.idx]
        self.delta = [target_status[s] - (init_all[s] + auto_all[max_level][s])
                      for s in self.idx]
        self.names = [STATUS_KINDS[s] for s in self.idx]
        self.base = [[init_all[self.idx[i]] + auto_all[min(L, max_level)][self.idx[i]]
                      for i in range(self.k)] for L in range(max_level + 1)]

        tp = [0] * (max_level + 1)
        for L in range(2, max_level + 1):
            tp[L] = tp[L - 1] + point_gain(L)
        self.total_points = tp
        self.max_cost = tp[max_level]

        # level_of_cost[c] = そのコストを賄える最小レベル（レベルはコストの純関数）
        lof = [0] * (self.max_cost + 2)
        L = 1
        for c in range(self.max_cost + 1):
            while L < max_level and tp[L] < c:
                L += 1
            lof[c] = L
        self.level_of_cost = lof

    @classmethod
    def from_config(cls, mul=5):
        import config              # CLI 専用。GUI/exe では読み込まない
        return cls(config.character, config.type, config.target_status,
                   config.initial_xien, config.next_xien, config.change_xien,
                   config.bonus_prob, mul=mul)


class LayerDP:
    def __init__(self, inst):
        self.inst = inst
        self.big = int(np.argmax(inst.delta))          # 暗黙軸（最大の Δ）
        # 最内軸（stride 1）を最大の Δ にすると、生存範囲 s-dbig <= Σ <= s が
        # 最内軸上の長い連続区間になり、死にセルを 1 個も踏まずに走査できる。
        # （最外軸を最大にして stride を縮める案も試したが実測で遅かった）
        self.others = sorted((j for j in range(inst.k) if j != self.big),
                             key=lambda j: inst.delta[j])
        self.box = tuple(inst.delta[j] + 1 for j in self.others)
        self.N = sum(inst.delta)

        S = np.zeros(self.box, dtype=np.int32)          # S[x] = Σx
        for p in range(len(self.box)):
            sh = [1] * len(self.box)
            sh[p] = self.box[p]
            S += np.arange(self.box[p], dtype=np.int32).reshape(sh)
        self.S = S

        self.lof = np.array(inst.level_of_cost, dtype=np.int32)
        self.tp = np.array(inst.total_points, dtype=np.int32)
        self.basecol = np.array([[inst.base[L][i] for L in range(inst.max_level + 1)]
                                 for i in range(inst.k)], dtype=np.int32)

        # --- numba カーネル用（フラット表現）
        self.Sf = S.ravel()
        st = []
        acc = 1
        for d in reversed(self.box):
            st.append(acc)
            acc *= d
        self.strides = np.array(list(reversed(st)), dtype=np.int64)
        self.dims = np.array(self.box, dtype=np.int64)
        self.others_arr = np.array(self.others, dtype=np.int64)
        self.a_arr = np.array(inst.a, dtype=np.int64)

        # acost[i, L, m] = stat i を m 個振った状態から Lv L で +1 するコスト
        mmax = max(inst.delta)
        L = np.arange(inst.max_level + 1, dtype=np.int64)[None, :, None]
        mm = np.arange(mmax + 1, dtype=np.int64)[None, None, :]
        bc = self.basecol.astype(np.int64)[:, :, None]
        self.acost = (np.array(inst.a, dtype=np.int64)[:, None, None]
                      + ((bc + mm) * inst.mul + L) // 125).astype(np.int32)

        # --- kernel3 用: 外側直積の座標と座標和（s に依存しないので 1 度だけ）
        self.dlast = self.box[-1]
        nouter = int(np.prod(self.box[:-1])) if len(self.box) > 1 else 1
        if len(self.box) > 1:
            oc = np.indices(self.box[:-1]).reshape(len(self.box) - 1, -1).T
            self.OC = np.ascontiguousarray(oc.astype(np.int32))
            self.Souter = self.OC.sum(axis=1).astype(np.int32)
        else:
            self.OC = np.zeros((1, 0), dtype=np.int32)
            self.Souter = np.zeros(1, dtype=np.int32)
        assert self.Souter.shape[0] == nouter

    def vstep(self, prev, i, m_prev):
        """コスト配列 prev から stat i を +1 したコスト。到達不能は INF。"""
        inst = self.inst
        alive = prev < INF
        c = np.where(alive, prev, 0)
        L = self.lof[np.clip(c, 0, inst.max_cost)]
        bc, tp = self.basecol[i], self.tp
        a_i, mul = inst.a[i], inst.mul
        out = np.full(prev.shape, INF, dtype=np.int32)
        todo = alive.copy()
        for _ in range(inst.max_level + 2):
            if not todo.any():
                break
            nc = c + a_i + ((bc[L] + m_prev) * mul + L) // 125
            ok = todo & (nc <= tp[L])
            out[ok] = nc[ok]
            todo &= ~ok
            L = np.where(todo, L + 1, L)
            todo &= ~(L > inst.max_level)
            L = np.clip(L, 0, inst.max_level)
        return out

    def _layer_candidates(self, prev, s):
        """レイヤー s の各セルについて (最小コスト, 採用した stat) を返す。"""
        inst, box, S = self.inst, self.box, self.S
        dbig = inst.delta[self.big]
        cur = np.full(box, INF, dtype=np.int32)
        arg = np.full(box, 255, dtype=np.uint8)

        # 暗黙軸を +1（箱の座標は不変）
        mbig_prev = s - 1 - S
        ok = (mbig_prev >= 0) & (mbig_prev < dbig)
        if ok.any():
            cand = self.vstep(np.where(ok, prev, INF), self.big,
                              np.where(ok, mbig_prev, 0))
            better = cand < cur
            cur[better] = cand[better]
            arg[better] = self.big

        # 箱の各軸を +1
        for p, j in enumerate(self.others):
            D = box[p]
            if D < 2:
                continue
            src = [slice(None)] * len(box)
            dst = [slice(None)] * len(box)
            src[p] = slice(0, D - 1)
            dst[p] = slice(1, D)
            src, dst = tuple(src), tuple(dst)
            sub = prev[src]
            sh = [1] * len(box)
            sh[p] = D - 1
            mp = np.broadcast_to(np.arange(D - 1, dtype=np.int32).reshape(sh), sub.shape)
            mbig = s - 1 - S[src]
            valid = (mbig >= 0) & (mbig <= dbig)
            cand = self.vstep(np.where(valid, sub, INF), j, mp)
            # dst は基本スライスなので cur[dst] / arg[dst] はビュー。直接書ける。
            cview, aview = cur[dst], arg[dst]
            better = cand < cview
            cview[better] = cand[better]
            aview[better] = j

        mbig_cur = s - S
        bad = (mbig_cur < 0) | (mbig_cur > dbig)
        cur[bad] = INF
        arg[bad] = 255
        return cur, arg

    def run(self, choices=None, progress=True, backend='auto', spill_dir=None,
            should_stop=None):
        """spill_dir を渡すと作業配列をディスク上（memmap）に確保する。

        箱が実メモリに収まらない場合の逃げ道。I/O 律速になるので遅いが、
        6〜7 ステータスに手動振りが要る目標でも計算できる。

        should_stop を渡すとレイヤーごとに中断要求を見る。1 件の DP が
        何時間もかかることがあるので、これが無いと中止ボタンが効かない。
        戻り値の第 1 要素が None なら中断された。
        """
        inst = self.inst
        use_numba = HAVE_NUMBA if backend == 'auto' else (backend == 'numba')
        B = int(np.prod(np.array(self.box, dtype=np.int64)))
        self.frontier = None          # 目標に届かない場合の最遠到達点

        prev, p1 = _alloc(B, np.int32, spill_dir, 'prev')
        cur, p2 = _alloc(B, np.int32, spill_dir, 'cur')
        arg, p3 = _alloc(B, np.uint8, spill_dir, 'arg')
        self._spill = [p for p in (p1, p2, p3) if p]
        prev[:] = INF
        prev[0] = 0                   # 箱の原点 = flat index 0
        # numba は ndarray のサブクラスを受け取らないのでビューにする
        kprev, kcur, karg = (np.asarray(x) for x in (prev, cur, arg))
        t0 = time.time()

        for s in range(1, self.N + 1):
            if should_stop is not None and s % 8 == 0 and should_stop():
                return None, time.time() - t0
            if use_numba:
                _layer_kernel3(kprev, kcur, karg, s, inst.delta[self.big], self.dlast,
                               self.strides, self.OC, self.Souter, self.big,
                               self.others_arr, self.acost, self.tp, self.lof,
                               inst.max_level, int(INF))
            elif backend == 'numba2':
                _layer_kernel2(kprev, kcur, karg, s, self.Sf, self.strides, self.dims,
                               self.big, self.others_arr, inst.delta[self.big],
                               self.acost, self.tp, self.lof,
                               inst.max_level, int(INF))
            elif backend == 'numba0':
                _layer_kernel(kprev, kcur, karg, s, self.Sf, self.strides, self.dims,
                              self.big, self.others_arr, inst.delta[self.big],
                              self.basecol, self.tp, self.lof, self.a_arr,
                              inst.mul, inst.max_level, int(INF))
            else:
                c2, a2 = self._layer_candidates(prev.reshape(self.box), s)
                cur[:] = c2.ravel()
                arg[:] = a2.ravel()
            prev, cur = cur, prev     # prev が今のレイヤー、cur は次で上書きされる
            kprev, kcur = kcur, kprev

            if choices is not None:
                choices[s - 1] = arg.reshape(self.box)
            alive = int((prev < INF).sum())
            if alive:
                flat = int(np.argmin(prev))
                x = np.unravel_index(flat, self.box)
                m = [0] * inst.k
                for p, j in enumerate(self.others):
                    m[j] = int(x[p])
                m[self.big] = s - int(self.S[x])
                self.frontier = (s, int(prev[flat]), m)
            if progress and (s % 20 == 0 or s == self.N):
                print(f'  s={s}/{self.N}  生存={alive:,}  '
                      f'{time.time() - t0:.1f}s', flush=True)

        goal = tuple(inst.delta[j] for j in self.others)
        out = int(np.asarray(prev).reshape(self.box)[goal]), time.time() - t0
        return out

    def cleanup(self):
        """ディスクに退避した作業ファイルを消す。"""
        for p in getattr(self, '_spill', []):
            try:
                os.remove(p)
            except OSError:
                pass
        self._spill = []

    def reconstruct(self, choices):
        """choices から振り順を復元して [(レベル, stat名, 累積コスト), ...] を返す。"""
        inst = self.inst
        x = [inst.delta[j] for j in self.others]
        m = list(inst.delta)
        steps = []
        for s in range(self.N, 0, -1):
            j = int(choices[s - 1][tuple(x)])
            if j == 255:
                raise RuntimeError(f'レイヤー {s} で経路復元に失敗')
            steps.append(j)
            if j != self.big:
                x[self.others.index(j)] -= 1
            m[j] -= 1
        steps.reverse()

        out, cost, m = [], 0, [0] * inst.k
        for j in steps:
            L = inst.level_of_cost[cost]
            while True:
                ac = inst.a[j] + ((inst.base[L][j] + m[j]) * inst.mul + L) // 125
                if cost + ac <= inst.total_points[L]:
                    break
                L += 1
            cost += ac
            m[j] += 1
            out.append((L, inst.names[j], cost, tuple(m)))
        return out


def main():
    import config
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', action='store_true', help='振り順も復元する')
    ap.add_argument('--mul', type=int, default=5,
                    help='コスト式の係数 (既定 5)。analyzer3/4 と multip() の実装値')
    ap.add_argument('--out', default=None, help='結果の出力先フォルダ')
    args = ap.parse_args()

    inst = Instance.from_config(mul=args.mul)
    dp = LayerDP(inst)
    cells = int(np.prod([d + 1 for d in inst.delta]))
    print(f'キャラ      : {config.character} / {config.type}')
    print(f'手動振り    : {list(zip(inst.names, inst.delta))}  (計 {sum(inst.delta)} 回)')
    print(f'従来の格子  : {cells:,} セル  ({cells * 4 / 1e9:.2f} GB 相当)')
    print(f'本版レイヤー: {dp.box} = {int(np.prod(dp.box)):,} セル '
          f'({np.prod(dp.box) * 4 / 1e6:.1f} MB x 2)')
    print()

    choices = None
    tmp = None
    if args.path:
        tmp = os.path.join(tempfile.gettempdir(), 'tw_choices.dat')
        choices = np.memmap(tmp, dtype=np.uint8, mode='w+',
                            shape=(dp.N,) + dp.box)
        print(f'経路復元用テンポラリ: {tmp} '
              f'({dp.N * np.prod(dp.box) / 1e9:.2f} GB)\n')

    cost, elapsed = dp.run(choices=choices)
    print()
    if cost >= INF:
        print('到達不能: Lv310 までのポイントでは目標ステータスに届きません。')
        if dp.frontier:
            s, c, m = dp.frontier
            print(f'  最遠到達    : {s}/{sum(inst.delta)} 手  (コスト {c}, '
                  f'Lv{inst.level_of_cost[c]})')
            for i, name in enumerate(inst.names):
                short = inst.delta[i] - m[i]
                print(f'    {name:5s} 手動 {m[i]:4d}/{inst.delta[i]:<4d}'
                      + (f'  ← {short} 不足' if short else ''))
        if tmp:
            del choices
            os.remove(tmp)
        return
    print(f'最小コスト  : {cost}')
    print(f'到達レベル  : {inst.level_of_cost[cost]}')
    print(f'計算時間    : {elapsed:.1f}s')

    if args.path:
        steps = dp.reconstruct(choices)
        out_dir = args.out or (config.output_folder + '_fast')
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, 'path.tsv')
        with open(dst, 'w', encoding='utf-8') as f:
            f.write('level\tstat\tcum_cost\t' + '\t'.join(inst.names) + '\n')
            for L, name, c, m in steps:
                f.write(f'{L}\t{name}\t{c}\t' + '\t'.join(map(str, m)) + '\n')
        print(f'振り順      : {dst} ({len(steps)} 手)')
        del choices
        os.remove(tmp)


if __name__ == '__main__':
    main()
