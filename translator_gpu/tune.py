import argparse
import difflib
import gc
import json
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import ctranslate2

import translator_gpu as tg

SEEDS = {
    "de": [
        "Das Wetter ist heute sehr schön und die Sonne scheint über Berlin.",
        "Ich habe gestern einen neuen Laptop gekauft, aber der Akku hält nur drei Stunden.",
        "Die Regierung hat angekündigt, die Steuern im nächsten Jahr zu senken.",
        "Bitcoin ist heute um fünf Prozent gefallen, viele Anleger sind nervös.",
        "Kann mir jemand erklären, warum die Bahn schon wieder Verspätung hat?",
        "Das neue Update bringt viele Verbesserungen, aber auch einige unerwartete Fehler.",
    ],
    "fr": [
        "Il fait très beau aujourd'hui à Paris et les gens profitent du soleil.",
        "J'ai acheté un nouveau téléphone hier, mais la batterie ne dure presque rien.",
        "Le gouvernement a annoncé une nouvelle réforme des retraites pour l'année prochaine.",
        "Le prix du bitcoin a chuté de cinq pour cent ce matin, les investisseurs s'inquiètent.",
        "Quelqu'un peut m'expliquer pourquoi le train a encore du retard ce soir ?",
        "La mise à jour apporte beaucoup d'améliorations mais aussi quelques bugs inattendus.",
    ],
    "es": [
        "Hoy hace un día precioso en Madrid y toda la ciudad está en la calle.",
        "Ayer compré un ordenador nuevo, pero la batería apenas dura tres horas.",
        "El gobierno ha anunciado una bajada de impuestos para el próximo año.",
        "El bitcoin ha caído un cinco por ciento esta mañana y los inversores están nerviosos.",
        "¿Alguien me puede explicar por qué el metro llega siempre tarde?",
        "La nueva actualización trae muchas mejoras, pero también algunos errores inesperados.",
    ],
    "ru": [
        "Сегодня в Москве отличная погода, и все гуляют по паркам.",
        "Вчера я купил новый ноутбук, но батарея держит только три часа.",
        "Правительство объявило о снижении налогов в следующем году.",
        "Биткоин сегодня упал на пять процентов, инвесторы очень нервничают.",
        "Кто-нибудь может объяснить, почему поезд опять опаздывает?",
        "Новое обновление принесло много улучшений, но и несколько неожиданных ошибок.",
    ],
}


def vram() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True,
        )
        return int(out.split()[0])
    except Exception:
        return -1


def build_corpus(args) -> list:
    path = args.corpus or os.path.join(tg.MODELS_DIR, "sample.jsonl")
    items = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    lang = str(d.get("detected_lang") or d.get("lang")).lower()
                    text = d.get("content") or d.get("text")
                    if text and lang:
                        items.append((lang, text))
                except ValueError:
                    pass
        print(f"корпус: {path} ({len(items)} шт.)", flush=True)
    if not items:
        rnd = random.Random(1)
        for lang, seeds in SEEDS.items():
            for _ in range(args.n // len(SEEDS)):
                items.append((lang, " ".join(rnd.choices(seeds, k=rnd.choice([1, 1, 1, 2, 2, 3, 5, 8])))))
        print("корпус: встроенный (de/fr/es/ru); для всех языков накопи sample.jsonl", flush=True)
    return items


def group_items(items: list) -> dict:
    groups = {}
    for lang, text in items:
        cands = tg._candidates(lang)
        if not cands:
            continue
        name, nllb, src = cands[0]
        g = groups.setdefault(name, dict(nllb=nllb, texts=[], srcs=[], langs={}))
        g["texts"].append(text)
        g["srcs"].append(src)
        g["langs"][lang] = g["langs"].get(lang, 0) + 1
    return groups


class Runner:
    def __init__(self, groups: dict):
        self.groups = groups
        self.bes = {}
        self.key = None
        self.total = sum(len(g["texts"]) for g in groups.values())

    def backends(self, compute: str, inflight: int) -> dict:
        if self.key != (compute, inflight):
            self.bes = {}
            gc.collect()
            self.bes = {n: tg.Backend(n, g["nllb"], compute, inflight) for n, g in self.groups.items()}
            self.key = (compute, inflight)
        return self.bes

    def tasks(self, limit: int = 0) -> list:
        out = []
        for n, g in self.groups.items():
            texts, srcs = (g["texts"][:limit], g["srcs"][:limit]) if limit else (g["texts"], g["srcs"])
            for i in range(0, len(texts), tg.BATCH_SIZE):
                out.append((n, texts[i:i + tg.BATCH_SIZE], srcs[i:i + tg.BATCH_SIZE]))
        return out

    def run(self, bes: dict, inflight: int, limit: int = 0) -> tuple:
        tasks = self.tasks(limit)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(inflight) as ex:
            res = list(ex.map(lambda t: bes[t[0]].translate(t[1], t[2]), tasks))
        return time.perf_counter() - t0, [(t[0], o) for t, r in zip(tasks, res) for o in r]

    def measure(self, compute, beam, bs, inflight, ref=None) -> dict:
        tg.BEAM, tg.BATCH_SIZE = beam, bs
        cfg = dict(compute=compute, beam=beam, bs=bs, inflight=inflight)
        try:
            bes = self.backends(compute, inflight)
            self.run(bes, inflight, 16)
            best, outs = None, None
            for _ in range(2):
                dt, o = self.run(bes, inflight)
                if best is None or dt < best:
                    best, outs = dt, o
        except Exception as e:
            return dict(cfg, ips=0.0, sim=0.0, vram=vram(), err=str(e)[:80], outs=[])
        sim = 1.0
        if ref:
            sim = sum(
                difflib.SequenceMatcher(None, a[1], b[1]).ratio() for a, b in zip(ref, outs)
            ) / len(outs)
        return dict(cfg, ips=self.total / best, sim=sim, vram=vram(), err="", outs=outs)


def verify(groups: dict, cur: dict):
    print("\nпо моделям (выбранный конфиг):", flush=True)
    tg.BEAM, tg.BATCH_SIZE = cur["beam"], cur["bs"]
    for name, g in sorted(groups.items(), key=lambda kv: -len(kv[1]["texts"])):
        be = None
        try:
            be = tg.Backend(name, g["nllb"], cur["compute"], cur["inflight"])
            texts, srcs = g["texts"][:128], g["srcs"][:128]
            be.translate(texts[:8], srcs[:8])
            t0 = time.perf_counter()
            be.translate(texts, srcs)
            ips = len(texts) / (time.perf_counter() - t0)
            res = f"{ips:7.1f} it/s"
        except Exception as e:
            res = f"ERR {str(e)[:60]}"
        del be
        gc.collect()
        langs = ",".join(f"{l}:{n}" for l, n in sorted(g["langs"].items(), key=lambda kv: -kv[1]))
        print(f"  {name.split('/')[-1]:<24} {res}  [{langs[:70]}]", flush=True)
    missing = sorted(tg._all_langs() - {l for g in groups.values() for l in g["langs"]})
    if missing:
        print(f"нет в корпусе (не проверены): {','.join(missing)}", flush=True)


def show(r: dict):
    tail = f" ERR {r['err']}" if r["err"] else ""
    print(
        f"  {r['compute']:<13} beam={r['beam']} bs={r['bs']:<4} inflight={r['inflight']}"
        f" | {r['ips']:7.1f} it/s  sim={r['sim']:.3f}  vram={r['vram']}MiB{tail}",
        flush=True,
    )


def best_of(rs: list, min_sim: float) -> dict:
    ok = [r for r in rs if r["ips"] > 0 and r["sim"] >= min_sim]
    if not ok:
        raise SystemExit("нет валидных конфигов (все упали или sim < min-sim)")
    top = max(r["ips"] for r in ok)
    near = [r for r in ok if r["ips"] >= top * 0.97]
    return min(near, key=lambda r: (r["bs"], r["inflight"], r["beam"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--corpus", default="")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--bs", default="16")
    p.add_argument("--min-sim", type=float, default=0.90)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    all_groups = group_items(build_corpus(args))
    top = sorted(all_groups, key=lambda n: -len(all_groups[n]["texts"]))[: args.top]
    r = Runner({n: all_groups[n] for n in top})
    print(f"тюнинг на моделях: {[n.split('/')[-1] for n in top]} items={r.total}", flush=True)

    supported = ctranslate2.get_supported_compute_types("cuda")
    print(f"supported={sorted(supported)} vram_start={vram()}MiB", flush=True)
    computes = [c for c in ("int8_float16", "int8", "int8_float32", "float16", "float32") if c in supported]

    ref_compute = "float32" if "float32" in supported else computes[0]
    print("reference:", flush=True)
    ref = r.measure(ref_compute, 4, 16, 1)
    show(ref)
    ref_outs = ref["outs"]

    print("compute:", flush=True)
    rs = [r.measure(c, 1, 16, 1, ref_outs) for c in computes]
    for x in rs:
        show(x)
    cur = best_of(rs, args.min_sim)

    print("beam:", flush=True)
    rs = [r.measure(cur["compute"], b, 16, 1, ref_outs) for b in (1, 2, 4)]
    for x in rs:
        show(x)
    cur = best_of(rs, args.min_sim)

    print("batch:", flush=True)
    rs = [
        r.measure(cur["compute"], cur["beam"], bs, 1, ref_outs)
        for bs in map(int, args.bs.split(","))
    ]
    for x in rs:
        show(x)
    cur = best_of(rs, args.min_sim)

    print("inflight:", flush=True)
    rs = [r.measure(cur["compute"], cur["beam"], cur["bs"], k, ref_outs) for k in (1, 2, 3)]
    for x in rs:
        show(x)
    cur = best_of(rs, args.min_sim)

    lines = [
        f"GPU_COMPUTE_TYPE={cur['compute']}",
        f"GPU_BEAM_SIZE={cur['beam']}",
        f"GPU_BATCH_SIZE={cur['bs']}",
        f"GPU_INFLIGHT={cur['inflight']}",
    ]
    print(f"\nBEST: {cur['ips']:.1f} it/s (reference {ref['ips']:.1f} it/s, x{cur['ips'] / max(ref['ips'], 1e-9):.1f}), sim={cur['sim']:.3f}")
    print("\n".join(lines))
    del r
    gc.collect()
    if not args.no_verify:
        verify(all_groups, cur)
    path = os.path.join(tg.MODELS_DIR, "tuned.env")
    try:
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"записано: {path}")
    except OSError:
        pass


if __name__ == "__main__":
    main()
