import argparse
import json
import math
import random
import statistics
import time

from langdetect import DetectorFactory, detect_langs, LangDetectException

DetectorFactory.seed = 0

EN = (
    "the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an "
    "will my one all would there their what so up out if about who get which go me when make can like time no just him "
    "know take people into year your good some could them see other than then now look only come its over think also "
    "back after use two how our work first well way even new want because any these give day most us great love really "
    "photo camera lens light game team win season price market stock coin bitcoin health doctor law court vote policy"
).split()
FOREIGN = [
    "Bonjour tout le monde, aujourd'hui il fait très beau et les gens profitent du soleil dans le parc",
    "Hoy es un gran día para salir a caminar con los amigos y disfrutar de la comida en la ciudad",
    "Heute ist das Wetter wirklich schön und wir gehen zusammen im Park spazieren und essen Eis",
    "Сегодня отличная погода, мы с друзьями идём гулять в парк и пить кофе на свежем воздухе",
    "今日はとても天気が良いので、友達と公園に散歩に行ってアイスクリームを食べます",
    "O tempo hoje está muito bom e nós vamos passear no parque com a família e comer sorvete",
]


def make_texts(n, seed=1):
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        if i % 5 == 0:
            base = rnd.choice(FOREIGN)
            out.append((base + " ") * rnd.randint(1, 4))
        else:
            k = int(min(200, max(3, rnd.lognormvariate(math.log(22), 0.9))))
            out.append(" ".join(rnd.choice(EN) for _ in range(k)).capitalize() + ".")
    return out


def top(text):
    try:
        r = detect_langs(text)
    except LangDetectException:
        return None
    return r[0].lang if r else None


def timed(texts, cut):
    ts, langs = [], []
    for t in texts:
        s = t[:cut] if cut else t
        a = time.perf_counter()
        langs.append(top(s))
        ts.append(time.perf_counter() - a)
    return ts, langs


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--texts")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.texts:
        lines = [l.strip() for l in open(args.texts, encoding="utf-8") if len(l.strip()) >= 20]
        texts = [lines[i % len(lines)] for i in range(args.n)]
    else:
        texts = make_texts(args.n)

    t = time.perf_counter()
    top("warm up the language profiles please")
    cold = time.perf_counter() - t
    timed(texts[:200], 0)

    res = {"cold_start_s": cold, "variants": []}
    print(f"langdetect cold start: {cold:.2f}s; текстов: {len(texts)}; средняя длина {statistics.mean(map(len, texts)):.0f} симв.")
    base_ts, base_langs = timed(texts, 0)
    print(f"{'вариант':<14}{'mean мс':>9}{'p50 мс':>9}{'p95 мс':>9}{'items/s (1 ядро)':>18}{'ускорение':>11}{'совпад. языка':>15}")
    base_mean = statistics.mean(base_ts)
    for name, cut in (("полный текст", 0), ("первые 500", 500), ("первые 300", 300), ("первые 200", 200)):
        ts, langs = (base_ts, base_langs) if cut == 0 else timed(texts, cut)
        m = statistics.mean(ts)
        agree = sum(1 for a, b in zip(base_langs, langs) if a == b) / len(texts)
        print(f"{name:<14}{m*1e3:>9.3f}{pct(ts,50)*1e3:>9.3f}{pct(ts,95)*1e3:>9.3f}{1/m:>18.0f}{base_mean/m:>10.2f}x{agree*100:>14.1f}%")
        res["variants"].append({"name": name, "mean_ms": m * 1e3, "p95_ms": pct(ts, 95) * 1e3, "items_per_s": 1 / m,
                                "speedup": base_mean / m, "agreement": agree})

    from langdetect import detector_factory as ldf
    ldf.init_factory()

    def top_n(text, n):
        try:
            d = ldf._factory.create()
            d.n_trial = n
            d.append(text)
            r = d.get_probabilities()
        except LangDetectException:
            return None, 0.0
        return (r[0].lang, float(r[0].prob)) if r else (None, 0.0)

    sample = texts[:1500]
    ref = [top_n(t, 7) for t in sample]
    print("\nLANGDETECT_TRIALS (сравнение с 7 пробами):")
    print(f"{'trials':>7}{'мс/item':>9}{'ускорение':>11}{'совпад. языка':>15}{'prob<0.9':>10}{'prob<0.7':>10}")
    res["trials"] = []
    ref_ms = None
    for n in (7, 5, 4, 3, 2):
        t0 = time.perf_counter()
        out = [top_n(t, n) for t in sample]
        ms = (time.perf_counter() - t0) / len(sample) * 1e3
        ref_ms = ref_ms or ms
        ag = sum(1 for (a, _), (b, _) in zip(ref, out) if a == b) / len(sample)
        l9 = sum(1 for _, p in out if p < 0.9) / len(sample)
        l7 = sum(1 for _, p in out if p < 0.7) / len(sample)
        print(f"{n:>7}{ms:>9.3f}{ref_ms/ms:>10.2f}x{ag*100:>14.1f}%{l9*100:>9.1f}%{l7*100:>9.1f}%")
        res["trials"].append({"n": n, "ms": ms, "speedup": ref_ms / ms, "agreement": ag, "low09": l9, "low07": l7})

    buckets = {"<100": [], "100-300": [], "300-1000": [], ">1000": []}
    for tx, d in zip(texts, base_ts):
        k = len(tx)
        buckets["<100" if k < 100 else "100-300" if k < 300 else "300-1000" if k < 1000 else ">1000"].append(d)
    print("\nпо длине текста (полный текст):")
    res["by_len"] = {}
    for k, v in buckets.items():
        if v:
            print(f"  {k:<9} n={len(v):<5} mean {statistics.mean(v)*1e3:.3f} мс")
            res["by_len"][k] = {"n": len(v), "mean_ms": statistics.mean(v) * 1e3}
    res["items_per_s_full"] = 1 / base_mean
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
