"""コスト式の係数 mul をゲーム内の実測値から絞り込む。

コスト式（(A) そのステータス単独で決まる、で確定）:

    必要ポイント = 初期コスト[stat] + floor( (ステ値 * mul + Lv) / 125 )

初期コストは character_parameter.py のキャラ×タイプごとの2番目の配列。
残る未知数は mul だけなので、観測を数点入れれば一意に決まる。

使い方:
    python calibrate.py ベンヤ 霊魂 HACK 284:140:12 300:160:13
                        ^キャラ ^タイプ ^ステ  ^Lv:ステ値:必要ポイント

  ・「ステ値」は振る直前の値（装備の補正を含まない素の値）
  ・「必要ポイント」はゲームが表示する、そのステを +1 するのに要るポイント
  ・観測は何個でも並べられる。2〜3点あればだいたい一意になる

引数なしで実行すると対話入力になる。
"""
import sys

import character_parameter as cp

STATUS_KINDS = ['STAB', 'HACK', 'INT', 'DEF', 'MR', 'DEX', 'AGI']
MUL_RANGE = range(1, 16)


def predict(a, s, L, mul):
    return a + (s * mul + L) // 125


def consistent(a, obs, mul):
    return all(predict(a, s, L, mul) == c for L, s, c in obs)


def report(character, ctype, stat, obs):
    try:
        a = cp.character_basic_status[character][ctype][1][STATUS_KINDS.index(stat)]
    except KeyError:
        print(f'キャラ/タイプが見つかりません: {character} / {ctype}')
        print('  指定できるキャラ:', ', '.join(cp.character_basic_status))
        return
    except ValueError:
        print(f'ステータス名が不正です: {stat}  (使えるのは {"/".join(STATUS_KINDS)})')
        return

    print(f'{character} / {ctype} / {stat}   初期コスト a = {a}')
    print()
    print('  観測点:')
    for L, s, c in obs:
        print(f'    Lv{L:3d}  ステ値{s:4d}  →  必要ポイント {c}')
    print()

    ok = [m for m in MUL_RANGE if consistent(a, obs, m)]
    print('  各 mul での予測（*=全観測に一致）:')
    print('    mul |' + ''.join(f' Lv{L}/{s}' for L, s, _ in obs) + '  |')
    for m in MUL_RANGE:
        pred = [predict(a, s, L, m) for L, s, _ in obs]
        mark = ' *' if m in ok else '  '
        hit = ''.join(f'{p:8d}' for p in pred)
        print(f'    {m:3d} |{hit}  |{mark}')
    print()

    if not ok:
        print('  × どの mul でも説明できません。')
        print('    ステ値が装備込みになっていないか、タイプが合っているか確認してください。')
    elif len(ok) == 1:
        print(f'  ◎ mul = {ok[0]} で確定しました。')
        print(f'    GUI の「コスト式係数」を {ok[0]} にしてください。')
        if ok[0] not in (5, 6):
            print(f'    ※ GUI の選択肢は 5/6 のみなので、{ok[0]} を足す必要があります。')
    else:
        print(f'  △ 候補が {ok} に絞れましたが、まだ一意ではありません。')
        print('    観測点を追加してください。ステ値が大きく離れた点ほど効きます。')
        lo, hi = min(ok), max(ok)
        for s in range(20, 320, 10):
            if predict(a, s, obs[0][0], lo) != predict(a, s, obs[0][0], hi):
                print(f'    例: Lv{obs[0][0]} でステ値 {s} 付近を測ると分かれます')
                break


def parse(tok):
    L, s, c = (int(x) for x in tok.split(':'))
    return L, s, c


def main():
    if len(sys.argv) >= 5:
        character, ctype, stat = sys.argv[1], sys.argv[2], sys.argv[3].upper()
        obs = [parse(t) for t in sys.argv[4:]]
    else:
        print('キャラ:', ', '.join(cp.character_basic_status))
        character = input('キャラ名 > ').strip()
        print('タイプ:', ', '.join(cp.character_basic_status.get(character, {})))
        ctype = input('タイプ > ').strip()
        stat = input(f'ステータス ({"/".join(STATUS_KINDS)}) > ').strip().upper()
        obs = []
        print('観測を Lv:ステ値:必要ポイント の形で入力（空行で終了）')
        while True:
            t = input('  > ').strip()
            if not t:
                break
            obs.append(parse(t))
    if not obs:
        print('観測がありません')
        return
    print()
    report(character, ctype, stat, obs)


if __name__ == '__main__':
    main()
