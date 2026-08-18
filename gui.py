"""再振り最適化ツール GUI（Tkinter / 標準ライブラリのみ）

    python gui.py

入力はキャラと目標ステータスだけ。タイプ・初期シエン・切替先シエン・切替レベルは
全部こちらで振って、目標に届く組み合わせを探す。
"""
import os
import queue
import sys
import time
import threading
import tkinter as tk
import traceback
from tkinter import filedialog, font as tkfont, messagebox, ttk

import character_parameter as cp
import xien_availability as xa
import search as se
import format_plan as fp
from solver_fast import INF, Instance, LayerDP, STATUS_KINDS

import numpy as np

CHARACTERS = list(cp.character_basic_status)

# コスト式の係数。ゲーム内実測（Lv75/Lv145 の全7ステ 14点）で 5 に確定している。
#   必要ポイント = 初期コスト + floor((ステ値 * 5 + Lv) / 125)
COST_MUL = 5

UNLIMITED = '無制限（推測なし）'
ALL_TYPES = '全タイプ'

class PlanWindow(tk.Toplevel):
    """振り分け表を表示する別ウィンドウ。

    シミュレータに貼り付けて使うものなので、等幅フォントで桁を揃え、
    全選択してすぐコピーできるようにしてある。
    """

    def __init__(self, parent, combo, args, txt, tsv, n_steps):
        super().__init__(parent)
        self.txt, self.tsv, self.combo, self.args = txt, tsv, combo, args
        self.title(f'振り分け表  {args["character"]}/{combo.ctype}  '
                   f'{combo.ix_name}→{combo.nx_name}  切替Lv{combo.cx}')
        self.geometry('660x780')
        self.minsize(520, 400)

        head = ttk.Frame(self)
        head.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Label(head, text=f'到達Lv {combo.exact_level}   '
                             f'コスト {combo.exact_cost}   {n_steps} 手',
                  font=('', 10, 'bold')).pack(side='left')
        self.mode = tk.StringVar(value='plan')
        ttk.Radiobutton(head, text='振り分け表', value='plan',
                        variable=self.mode, command=self._show).pack(side='right')
        ttk.Radiobutton(head, text='各手の明細', value='tsv',
                        variable=self.mode, command=self._show).pack(side='right',
                                                                     padx=6)

        body = ttk.Frame(self)
        body.pack(fill='both', expand=True, padx=8)
        # 等幅かつ半角カナも揃うフォントを順に試す
        font = ('MS Gothic', 10)
        for cand in ('MS Gothic', 'Consolas', 'Courier New'):
            try:
                tkfont.Font(family=cand, size=10)
                font = (cand, 10)
                break
            except tk.TclError:
                continue
        self.text = tk.Text(body, wrap='none', font=font, undo=False)
        ys = ttk.Scrollbar(body, orient='vertical', command=self.text.yview)
        xs = ttk.Scrollbar(body, orient='horizontal', command=self.text.xview)
        self.text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.text.grid(row=0, column=0, sticky='nsew')
        ys.grid(row=0, column=1, sticky='ns')
        xs.grid(row=1, column=0, sticky='ew')
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        bot = ttk.Frame(self)
        bot.pack(fill='x', padx=8, pady=8)
        ttk.Button(bot, text='全部コピー', command=self._copy).pack(side='left')
        ttk.Button(bot, text='ファイルに保存...',
                   command=self._save).pack(side='left', padx=6)
        ttk.Button(bot, text='閉じる', command=self.destroy).pack(side='right')
        self.hint = ttk.Label(bot, text='Ctrl+A で全選択、Ctrl+C でコピー',
                              foreground='#666')
        self.hint.pack(side='left', padx=12)

        self.text.bind('<Control-a>', self._select_all)
        self.text.bind('<Control-A>', self._select_all)
        self.bind('<Escape>', lambda e: self.destroy())
        self._show()

    def _show(self):
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', self.txt if self.mode.get() == 'plan' else self.tsv)
        self.text.configure(state='disabled')   # 読み取り専用。選択とコピーは可能

    def _select_all(self, _=None):
        self.text.configure(state='normal')
        self.text.tag_add('sel', '1.0', 'end-1c')
        self.text.configure(state='disabled')
        self.text.focus_set()
        return 'break'

    def _copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.txt if self.mode.get() == 'plan' else self.tsv)
        self.hint.config(text='クリップボードにコピーしました', foreground='#2a7a2a')
        self.after(2500, lambda: self.hint.config(
            text='Ctrl+A で全選択、Ctrl+C でコピー', foreground='#666'))

    def _save(self):
        c, plan = self.combo, self.mode.get() == 'plan'
        ext = '.txt' if plan else '.tsv'
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=ext,
            filetypes=[('テキスト', '*' + ext)],
            initialfile=f'plan_{c.ctype}_{c.ix_name}_{c.nx_name}_Lv{c.cx}{ext}')
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.txt + chr(10) if plan else self.tsv)
        self.hint.config(text='保存しました: ' + os.path.basename(path),
                         foreground='#2a7a2a')



class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('再振り最適化')
        self.geometry('1080x720')
        self.minsize(900, 600)

        self.q = queue.Queue()
        self.worker = None
        self.stop_flag = threading.Event()
        self.results = []
        self.row_combo = {}
        self.search_args = None

        self._check_job = None      # 入力が落ち着くまで待つための after id
        self._check_token = 0       # 古い結果を捨てるための世代番号

        self._build()
        self.after(100, self._pump)
        threading.Thread(target=se.warmup, daemon=True).start()
        self._schedule_check()

    # ------------------------------------------------------------ 実現性の速報
    def _schedule_check(self):
        """入力が 400ms 落ち着いたら裏で判定する（打鍵ごとには走らせない）。"""
        if self._check_job is not None:
            self.after_cancel(self._check_job)
        self.feas.config(text='判定中...', foreground='#666')
        self._check_job = self.after(400, self._run_check)

    def _run_check(self):
        self._check_job = None
        try:
            target = [int(self.stat_vars[n].get()) for n in STATUS_KINDS]
        except ValueError:
            self.feas.config(text='目標ステータスは整数で入力してください',
                             foreground='#b25000')
            return
        if any(v < 1 or v > 999 for v in target):
            self.feas.config(text='目標ステータスは 1〜999 の範囲で',
                             foreground='#b25000')
            return
        self._check_token += 1
        tok = self._check_token
        args = (self.char_var.get(), target,
                int(self.bonus_var.get()) / 100.0, COST_MUL)
        kw = {'types': self._sel_types()}
        def work():
            # 1段目: 貪欲法の速報（数十ms〜1秒）
            try:
                r = se.quick_check(*args, **kw)
            except Exception as e:
                self.q.put(('feas', (tok, {'state': 'err', 'msg': str(e)}, False)))
                return
            self.q.put(('feas', (tok, r, False)))
            # 貪欲法が届いたならそれ自体が実行可能な計画なので裏取り不要
            if r.get('state') in ('ok', 'auto', 'none'):
                return
            # 2段目: 上位数件だけ厳密に確認（境界付近で貪欲法は当てにならない）
            stop = lambda: tok != self._check_token
            try:
                e = se.quick_exact(*args, should_stop=stop, **kw)
            except Exception as ex:
                e = {'state': 'err', 'msg': str(ex)}
            if not stop():
                self.q.put(('feas', (tok, e, True)))
        threading.Thread(target=work, daemon=True).start()

    def _show_check(self, tok, r, exact=False):
        """exact=False は貪欲法の速報、True は上位数件の厳密確認の結果。"""
        if tok != self._check_token:
            return                      # 入力が進んでいるので破棄
        st = r.get('state')
        GREEN, RED, GRAY = '#2a7a2a', '#b25000', '#666'
        if st == 'auto':
            self.feas.config(foreground=GREEN,
                             text='シエンの自動上昇だけで目標に届きます（手動振り不要）')
        elif st == 'ok' and exact:
            self.feas.config(foreground=GREEN,
                             text=f'到達可能（確定） — Lv{r["level"]} で到達。'
                                  f'上位{r["checked"]}件を厳密計算して確認')
        elif st == 'ok':
            self.feas.config(
                foreground=GREEN,
                text=f'到達可能 — Lv{r["level"]} までに到達できます'
                     f'（手動振り {r["nstat"]}ステータス / {r["need"]}回）')
        elif st == 'ng' and exact:
            sh = '、'.join(f'{n} {d}不足' for n, d in r.get('short', [])[:4])
            self.feas.config(
                foreground=RED,
                text=f'届く可能性は低いです — 有望な上位{r["checked"]}件を'
                     f'厳密計算しても不足'
                     + (f'（最良で {sh}）' if sh else '')
                     + '。全候補を試したわけではないので、探索すれば'
                       '見つかる場合もあります')
        elif st == 'ng':
            sh = '、'.join(f'{n} {d}不足' for n, d in r.get('short', [])[:4])
            self.feas.config(
                foreground=GRAY,
                text=f'簡易判定では届いていません（{sh}）'
                     f'  手動振り {r["nstat"]}ステータス / {r["need"]}回'
                     f'  — 厳密に確認中...')
        elif st == 'err':
            self.feas.config(foreground=RED,
                             text=f'判定できません: {r.get("msg", "")}')
        elif not exact:
            self.feas.config(foreground=RED,
                             text='候補なし（キャラのシエン候補を確認してください）')

    # ------------------------------------------------------------ 画面
    def _build(self):
        pad = dict(padx=6, pady=4)

        top = ttk.LabelFrame(self, text='入力')
        top.pack(fill='x', **pad)

        row = ttk.Frame(top)
        row.pack(fill='x', padx=8, pady=6)
        ttk.Label(row, text='キャラクター').pack(side='left')
        self.char_var = tk.StringVar(value='ジョシュア')
        cb = ttk.Combobox(row, textvariable=self.char_var, values=CHARACTERS,
                          state='readonly', width=14)
        cb.pack(side='left', padx=(4, 16))
        cb.bind('<<ComboboxSelected>>', lambda e: self._on_character())

        ttk.Label(row, text='タイプ').pack(side='left')
        self.type_var = tk.StringVar(value=ALL_TYPES)
        self.type_cb = ttk.Combobox(row, textvariable=self.type_var, width=12,
                                    state='readonly')
        self.type_cb.pack(side='left', padx=(4, 16))
        self.type_cb.bind('<<ComboboxSelected>>', lambda e: self._on_type())

        ttk.Label(row, text='ボーナス確率').pack(side='left')
        self.bonus_var = tk.StringVar(value='50')
        bcb = ttk.Combobox(row, textvariable=self.bonus_var, width=5,
                           state='readonly',
                           values=[str(v) for v in range(0, 101, 5)])
        bcb.pack(side='left')
        bcb.bind('<<ComboboxSelected>>', lambda e: self._schedule_check())
        ttk.Label(row, text='%').pack(side='left', padx=(2, 16))

        ttk.Label(row, text='計算時間').pack(side='left')
        self.budget_var = tk.StringVar(value='120秒')
        ttk.Combobox(row, textvariable=self.budget_var, width=14,
                     state='readonly',
                     values=['30秒', '60秒', '120秒', '300秒', '600秒',
                             UNLIMITED]).pack(side='left', padx=(4, 0))

        srow = ttk.Frame(top)
        srow.pack(fill='x', padx=8, pady=(0, 4))
        ttk.Label(srow, text='目標ステータス').pack(side='left', padx=(0, 8))
        self.stat_vars = {}
        defaults = [3, 1, 103, 100, 71, 310, 310]
        for i, name in enumerate(STATUS_KINDS):
            ttk.Label(srow, text=name).pack(side='left', padx=(8, 2))
            v = tk.StringVar(value=str(defaults[i]))
            self.stat_vars[name] = v
            ttk.Entry(srow, textvariable=v, width=5).pack(side='left')
            v.trace_add('write', lambda *a: self._schedule_check())
        ttk.Button(srow, text='Lv1に戻す', width=10,
                   command=self._fill_base).pack(side='left', padx=(14, 4))
        self.base_lbl = ttk.Label(srow, text='', foreground='#666')
        self.base_lbl.pack(side='left')

        frow = ttk.Frame(top)
        frow.pack(fill='x', padx=8, pady=(0, 4))
        ttk.Label(frow, text='見込み', width=8).pack(side='left')
        self.feas = ttk.Label(frow, text='', anchor='w')
        self.feas.pack(side='left', fill='x', expand=True)

        brow = ttk.Frame(top)
        brow.pack(fill='x', padx=8, pady=(0, 8))
        self.run_btn = ttk.Button(brow, text='探索開始', command=self._start)
        self.run_btn.pack(side='left')
        self.stop_btn = ttk.Button(brow, text='中止', command=self._stop,
                                   state='disabled')
        self.stop_btn.pack(side='left', padx=6)
        self.note = ttk.Label(brow, text='', foreground='#b25000')
        self.note.pack(side='left', padx=16)

        prow = ttk.Frame(self)
        prow.pack(fill='x', **pad)
        self.pbar = ttk.Progressbar(prow, mode='determinate')
        self.pbar.pack(side='left', fill='x', expand=True)
        self.status = ttk.Label(prow, text='待機中', width=46, anchor='w')
        self.status.pack(side='left', padx=8)

        mid = ttk.LabelFrame(self, text='結果（良い順）')
        mid.pack(fill='both', expand=True, **pad)
        orow = ttk.Frame(mid)
        orow.pack(fill='x', padx=4, pady=(2, 0))
        self.group_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(orow, text='同じ組み合わせは最良の切替Lvだけ表示',
                        variable=self.group_var,
                        command=self._render).pack(side='left')
        self.count_lbl = ttk.Label(orow, text='', foreground='#666')
        self.count_lbl.pack(side='left', padx=12)

        cols = ('res', 'judge', 'type', 'ix', 'nx', 'cx', 'lv', 'cost', 'detail')
        heads = {'res': '結果', 'judge': '判定', 'type': 'タイプ',
                 'ix': '初期シエン', 'nx': '切替先シエン', 'cx': '切替Lv',
                 'lv': '到達Lv', 'cost': 'コスト', 'detail': '内訳 / 不足'}
        widths = {'res': 56, 'judge': 56, 'type': 76, 'ix': 118, 'nx': 118,
                  'cx': 126, 'lv': 58, 'cost': 64, 'detail': 278}
        self.tree = ttk.Treeview(mid, columns=cols, show='headings', height=14)
        self.sort_col = None
        self.sort_rev = False
        for c in cols:
            self.tree.heading(c, text=heads[c],
                              command=lambda cc=c: self._sort_by(cc))
            self.tree.column(c, width=widths[c],
                             anchor='w' if c in ('detail', 'ix', 'nx') else 'center')
        vs = ttk.Scrollbar(mid, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.pack(side='left', fill='both', expand=True)
        vs.pack(side='right', fill='y')
        self.tree.bind('<Double-1>', lambda e: self._detail())

        bot = ttk.Frame(self)
        bot.pack(fill='x', **pad)
        ttk.Button(bot, text='選択行の振り分け表を表示',
                   command=self._detail).pack(side='left')
        ttk.Label(bot, text='行をダブルクリックでも表示できます',
                  foreground='#666').pack(side='left', padx=10)

        self._reload_types()
        self._fill_base()
        self._refresh_note()

    def _sel_types(self):
        """探索対象のタイプ一覧。全タイプなら None（= 制限なし）。"""
        t = self.type_var.get()
        return None if t == ALL_TYPES else [t]

    def _reload_types(self):
        """キャラに合わせてタイプの選択肢を作り直す。"""
        types = list(cp.character_basic_status.get(self.char_var.get(), {}))
        self.type_cb['values'] = [ALL_TYPES] + types
        if self.type_var.get() not in types:
            self.type_var.set(ALL_TYPES)

    def _on_character(self):
        self._reload_types()
        self._fill_base()
        self._refresh_note()
        self._schedule_check()

    def _on_type(self):
        self._fill_base()
        self._refresh_note()
        self._schedule_check()

    def _fill_base(self):
        """目標欄をそのキャラの Lv1 初期ステータスで埋める。

        Lv1 の値はタイプごとに違う（最大 4 差）。全タイプの最大を取ると
        どのタイプにも存在しない架空の値になるので、実在する 1 タイプ分を
        そのまま入れ、どのタイプの値かをラベルに出す。
        """
        ch = self.char_var.get()
        types = cp.character_basic_status.get(ch)
        if not types:
            self.base_lbl.config(text='')
            return
        sel = self._sel_types()
        ctype = sel[0] if sel else next(iter(types))
        base = types[ctype][0]
        for name, v in zip(STATUS_KINDS, base):
            self.stat_vars[name].set(str(v))
        note = f'Lv1初期値「{ctype}」'
        others = [t for t in types if t != ctype]
        if others and not sel:
            same = all(types[t][0] == list(base) for t in others)
            note += '（全タイプ同じ）' if same else f'（他 {"/".join(others)} は別の値）'
        self.base_lbl.config(text=note)

    def _refresh_note(self):
        ch = self.char_var.get()
        types = list(cp.character_basic_status.get(ch, {}))
        if not types:
            self.note.config(text='')
            return
        sel = self._sel_types()
        if sel:
            types = sel
        undef = []
        pairs = set()
        for t in types:
            names, ver = xa.candidates_for(ch, t, cp.xien_list)
            pairs |= {tuple(cp.xien_list[n]) for n in names}
            if not ver:
                undef.append(t)
        if undef:
            self.note.config(foreground='#b25000',
                             text=f'※ {ch} の {"/".join(undef)} はシエン候補が未確認'
                                  f' → 全{len(pairs)}ペアを総当たり（重くなります）')
        else:
            label = f'タイプ{len(types)}' if not sel else f'「{sel[0]}」のみ'
            self.note.config(
                foreground='#2a7a2a',
                text=f'シエン候補: {label} × {len(pairs)}ペア を探索')

    # ------------------------------------------------------------ 実行
    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            target = [int(self.stat_vars[n].get()) for n in STATUS_KINDS]
        except ValueError:
            messagebox.showerror('入力エラー', '目標ステータスは整数で入力してください')
            return
        if any(v < 1 or v > 999 for v in target):
            messagebox.showerror('入力エラー', '目標ステータスは 1〜999 の範囲で')
            return

        raw = self.budget_var.get()
        if raw == UNLIMITED:
            if not messagebox.askokcancel(
                    '無制限モードの確認',
                    '候補をすべて厳密計算します。推測値は一切出ません。' + chr(10) + chr(10)
                    + '・目標によっては数時間〜数十時間かかります' + chr(10)
                    + '・所要時間の見積りは探索開始の数秒後に出ます' + chr(10)
                    + '  （実測で補正しながら残り時間を表示します）' + chr(10)
                    + '・途中の「中止」で、そこまでの確定分を見られます' + chr(10)
                    + '・作業配列はこのPCの空きメモリに合わせて確保し、' + chr(10)
                    + '  収まらない分は一時ファイルに退避します（遅くなります）'
                    + chr(10) + chr(10) + '実行しますか？',
                    icon='warning'):
                return
            budget = None
        else:
            budget = float(raw.rstrip('秒'))

        args = dict(character=self.char_var.get(), target=target,
                    bonus_prob=int(self.bonus_var.get()) / 100.0,
                    mul=COST_MUL, time_budget=budget, types=self._sel_types())
        self.stop_flag.clear()
        self.results = []
        self.tree.delete(*self.tree.get_children())
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.pbar.config(mode='indeterminate')
        self.pbar.start(12)
        self.worker = threading.Thread(target=self._work, args=(args,), daemon=True)
        self.worker.start()

    def _stop(self):
        self.stop_flag.set()
        self.status.config(text='中止しています...')

    def _work(self, args):
        try:
            res, ctx, seen = se.search(
                args['character'], args['target'], args['bonus_prob'],
                mul=args['mul'], cx_step=10, top_k=400,
                time_budget=args['time_budget'], types=args.get('types'),
                progress=lambda m: self.q.put(('status', m)),
                should_stop=self.stop_flag.is_set)
            self.q.put(('done', (res, args)))
        except Exception:
            self.q.put(('error', traceback.format_exc()))

    def _pump(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == 'status':
                    self.status.config(text=payload)
                elif kind == 'feas':
                    self._show_check(*payload)
                elif kind == 'done':
                    self._finish(*payload)
                elif kind == 'error':
                    self.pbar.stop()
                    self.run_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    self.status.config(text='エラー')
                    messagebox.showerror('エラー', payload)
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def _finish(self, res, args):
        self.pbar.stop()
        self.pbar.config(mode='determinate', value=0)
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.results = res
        self.search_args = args
        ok = sum(1 for c in res if c.exact and c.exact_cost is not None)
        nex = sum(1 for c in res if c.exact)
        self.status.config(
            text=f'完了: 厳密{nex}件(達成{ok}件) / 推定{len(res) - nex}件'
                 + ('（中止）' if self.stop_flag.is_set() else ''))
        self._render()
        if not res:
            messagebox.showinfo('結果なし',
                                '候補がありませんでした。目標が高すぎるか、'
                                '自動上昇だけで目標を超えている可能性があります。')
        elif nex == 0 and args.get('time_budget') is not None:
            messagebox.showinfo(
                '推定のみ',
                '手動振りが必要なステータスが多く、厳密DPが時間内に回りませんでした。\n'
                '表は貪欲法の推定値です（最適ではありません）。\n'
                '「計算時間の目安」を伸ばすか、目標を下げてください。')

    def _render(self):
        """結果を表に流し込む。既定では (タイプ,初期,切替先) ごとに最良の1行だけ。"""
        self.tree.delete(*self.tree.get_children())
        self.row_combo = {}
        res = self.results
        if not res:
            self.count_lbl.config(text='')
            return

        if self.group_var.get():
            groups = {}
            for c in res:
                groups.setdefault((c.ctype, c.ix_name, c.nx_name), []).append(c)
            rows = []
            for g in groups.values():
                best = g[0]                       # res は既に良い順
                same = [x for x in g
                        if (x.exact and x.exact_cost is not None)
                        == (best.exact and best.exact_cost is not None)]
                cxs = sorted(x.cx for x in same)
                rows.append((best, len(g), cxs))
            rows.sort(key=lambda t: res.index(t[0]))
            self.count_lbl.config(
                text=f'{len(rows)}組に集約（全{len(res)}行）'
                     '  チェックを外すと全部出ます')
        else:
            rows = [(c, 1, [c.cx]) for c in res]
            self.count_lbl.config(text=f'全{len(res)}行')

        for c, n, cxs in rows:
            mix = ' '.join(f'{s}+{d}' for s, d in zip(c.names, c.delta))
            cx_txt = str(c.cx)
            if n > 1 and len(cxs) > 1:
                cx_txt = f'{c.cx}  (+{len(cxs) - 1}件 {cxs[0]}〜{cxs[-1]})'
            if c.exact and c.exact_cost is not None:
                if getattr(c, 'spilled', False):
                    mix += '  (ディスク退避で計算)'
                vals = ('達成', '厳密', c.ctype, c.ix_name, c.nx_name, cx_txt,
                        c.exact_level, c.exact_cost, mix)
                tag = 'ok'
            elif c.exact:
                s, lv, short = c.frontier
                detail = (f'{s}/{c.N}手  不足: '
                          + ', '.join(f'{x} {d}' for x, d in short))
                vals = ('不可', '厳密', c.ctype, c.ix_name, c.nx_name, cx_txt,
                        lv, '-', detail)
                tag = 'ng'
            elif getattr(c, 'oversized', False):
                need = c.box * se.BYTES_PER_CELL / 1e9
                detail = (f'{c.g_done}/{c.N}手  '
                          f'計算不可: 作業領域 約{need:,.0f}GB が必要')
                vals = ('計算不可', '除外', c.ctype, c.ix_name, c.nx_name,
                        cx_txt, '-', '-', detail)
                tag = 'est'
            else:
                short = c.N - c.g_done
                est = ('<1s' if c.est_sec < 1 else f'{c.est_sec:.0f}s')
                if short == 0:
                    detail = f'{mix}  (厳密化 {est})'
                    vals = ('達成?', '推定', c.ctype, c.ix_name, c.nx_name,
                            cx_txt, '?', c.g_cost, detail)
                else:
                    detail = f'{c.g_done}/{c.N}手  (厳密化 {est})'
                    vals = ('不可?', '推定', c.ctype, c.ix_name, c.nx_name,
                            cx_txt, '?', '-', detail)
                tag = 'est'
            iid = self.tree.insert('', 'end', values=vals, tags=(tag,))
            self.row_combo[iid] = c
        self.tree.tag_configure('ok', background='#eaf6ea')
        self.tree.tag_configure('ng', background='#f7f0ee')
        self.tree.tag_configure('est', foreground='#777')

    def _sort_by(self, col):
        """列見出しクリックで並べ替え。数値列は数値として比較する。"""
        if not self.tree.get_children():
            return
        self.sort_rev = not self.sort_rev if self.sort_col == col else False
        self.sort_col = col

        def key(iid):
            v = self.tree.set(iid, col)
            head = v.split(' ')[0].replace('-', '').replace('?', '')
            try:
                return (0, float(head))
            except ValueError:
                return (1, v)

        for i, iid in enumerate(sorted(self.tree.get_children(), key=key,
                                       reverse=self.sort_rev)):
            self.tree.move(iid, '', i)


    # ------------------------------------------------------------ 出力
    def _selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo('未選択', '結果の行を選んでください')
            return None
        return self.row_combo.get(sel[0])

    def _detail(self):
        c = self._selected()
        if c is None:
            return
        if not c.exact:
            messagebox.showinfo(
                '推定行',
                'この行は貪欲法の推定です。振り分け表を出すには厳密DPが必要で、'
                f'約{c.est_sec:.0f}秒かかります。' + chr(10)
                + '「計算時間の目安」を伸ばして再探索してください。')
            return
        if c.exact_cost is None:
            messagebox.showinfo('到達不能',
                                'この組み合わせは目標に届かないため振り分け表はありません')
            return
        a = self.search_args
        self.status.config(text='振り順を復元中...')
        self.update_idletasks()
        try:
            inst = Instance(a['character'], c.ctype, a['target'], c.ix_name,
                            c.nx_name, c.cx, a['bonus_prob'], mul=a['mul'])
            dp = LayerDP(inst)
            ch = np.zeros((dp.N,) + dp.box, dtype=np.uint8)
            cost, _ = dp.run(choices=ch, progress=False)
            if cost >= INF:
                messagebox.showerror('エラー', '再計算で到達不能になりました')
                return
            steps = dp.reconstruct(ch)
            txt = fp.build(inst, steps, a['character'], c.ctype,
                           a['bonus_prob'], c.ix_name, c.nx_name, c.cx)
            nl, tab = chr(10), chr(9)
            tsv = ('level<TAB>stat<TAB>cum_cost<TAB>'.replace('<TAB>', tab)
                   + tab.join(inst.names) + nl)
            for L, nm, cc, mm in steps:
                tsv += (str(L) + tab + nm + tab + str(cc) + tab
                        + tab.join(map(str, mm)) + nl)
            PlanWindow(self, c, a, txt, tsv, len(steps))
            self.status.config(text=f'振り分け表を表示: 切替Lv{c.cx}')
        except MemoryError:
            messagebox.showerror('メモリ不足',
                                 '経路復元用の一時配列が確保できませんでした')
            self.status.config(text='待機中')


def selftest(out_path=None):
    """GUI を出さずに一通り動かして結果をファイルに書く。

    exe に固めた後で「numba が動くか」「探索・振り分け表の生成が通るか」を
    確かめるための自己診断。--selftest を付けて起動すると走る。
    ウィンドウなしでビルドしているので、結果は標準出力ではなくファイルに出す。
    """
    import numpy as np
    out_path = out_path or os.path.join(
        os.environ.get('LOCALAPPDATA') or os.getcwd(), 'tw_saifuri_selftest.txt')
    L = []
    def say(m):
        L.append(m)
    try:
        import solver_fast as sf
        say(f'numba        : {sf.HAVE_NUMBA}')
        say(f'メモリ空き/合計/コミット残 : '
            + ' / '.join(f'{v/1e9:.1f}GB' for v in sf.mem_info()))
        target = [2, 185, 1, 88, 3, 310, 300]
        t = time.time()
        res, _, _ = se.search('ベンヤ', target, 0.55, mul=COST_MUL, cx_step=20,
                              top_k=20, time_budget=60,
                              progress=lambda m: None, should_stop=lambda: False)
        say(f'探索          : {len(res)}件 / {time.time()-t:.1f}s')
        ok = [c for c in res if c.exact and c.exact_cost is not None]
        assert ok, '達成する組み合わせが見つからない'
        b = ok[0]
        say(f'最良          : {b.ctype} {b.ix_name}->{b.nx_name} 切替Lv{b.cx} '
            f'到達Lv{b.exact_level} コスト{b.exact_cost}')
        assert b.exact_cost == 2270, f'既知の最適値と違う: {b.exact_cost}'

        inst = Instance('ベンヤ', b.ctype, target, b.ix_name, b.nx_name, b.cx,
                        0.55, mul=COST_MUL)
        dp = LayerDP(inst)
        ch = np.zeros((dp.N,) + dp.box, dtype=np.uint8)
        cost, _ = dp.run(choices=ch, progress=False)
        steps = dp.reconstruct(ch)
        txt = fp.build(inst, steps, 'ベンヤ', b.ctype, 0.55, b.ix_name,
                       b.nx_name, b.cx)
        say(f'振り分け表    : {len(txt.splitlines())}行 / {len(steps)}手')
        import replay_plan as rp
        st, left, _ = rp.run(txt, 'ベンヤ', b.ctype, 0.55, b.ix_name, b.nx_name,
                             b.cx, mul=COST_MUL)
        want = [int(x) for x in txt.splitlines()[-1].split()[2::2]]
        assert st == want, f'読み戻しが一致しない {st} != {want}'
        say(f'読み戻し検証  : 一致 ({" ".join(map(str, st))} 残{left})')
        say('')
        say('=== 自己診断 OK ===')
    except Exception:
        import traceback
        say('')
        say('=== 自己診断 失敗 ===')
        say(traceback.format_exc())
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(chr(10).join(L) + chr(10))
    return out_path


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        App().mainloop()
