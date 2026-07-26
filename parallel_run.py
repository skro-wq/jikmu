"""직무발명 시뮬레이션 실행기 (설계 v5).

층화: 형식 수준(4) × 보상 관대함(4) × 인지모드(2) = 32셀. 기본 N=512 → 셀당 16건.

v5에서 바뀐 것:
  - 에피소드마다 독립 난수 생성기. v4까지는 8개 스레드가 전역 random을 공유하여
    시드를 고정해도 병렬 실행이 재현되지 않았다(실측: 8워커 동일 시드 2회 실행 불일치).
  - 재시도는 LLM 호출만 다시 한다. v4까지는 재시도가 env.reset()을 다시 불러 시나리오를
    통째로 새로 뽑았으므로, 파싱 실패가 시나리오 내용과 상관되면 그 설계점이 조용히
    교체되는 생존 편향이 있었다(구 로그의 파싱 실패율 7.75%).
  - 드롭·재시도를 기록한다. v4까지는 실패 에피소드가 로그 없이 사라지고 ID가 재부여되어
    16셀 설계가 조용히 불균형해질 수 있었다.
  - 인지모드를 층화 (구설계는 random.choice여서 183/217로 치우쳤다).

실행:
  LLM_BACKEND=openai  python parallel_run.py [N] [WORKERS]
  LLM_BACKEND=claude CLAUDE_SIM_MODEL=claude-sonnet-5 MAX_THINKING_TOKENS=0 python parallel_run.py 64 8
"""
import csv
import os
import sys
import json
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents import EmployeeInventorAgent
from env import JikmuEnv, COMP_RATIO_LEVELS

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
OUT = os.path.join(OUT_DIR, os.environ.get("SIM_OUT", "sim_results_v5.csv"))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BASE_SEED = int(os.environ.get("SIM_SEED", "20250807"))

_DEFAULT_MODEL = {"openai": "gpt-5-2025-08-07", "claude": "claude-sonnet-5"}
MODEL = (os.environ.get("SIM_MODEL")
         or _DEFAULT_MODEL.get(os.environ.get("LLM_BACKEND", "openai"), "gpt-5-2025-08-07"))

FIELDNAMES = [
    "에피소드ID", "산업", "발명중요도",
    "보상규정형식레벨", "보상규정형식설명",
    "보상비율수준", "보상비율라벨", "보상비율실현값",
    "인지모드",
    "실시료율", "기준매출만원", "독점권기여율",
    "출원보너스만원", "등록보너스만원", "실시보너스만원", "실지급액합계만원",
    "충격유형", "충격배율",
    "사용자이익기초만원", "사용자이익충격후만원",
    "충격전정당보상만원", "정당보상추정만원", "격차만원",
    "발명자결정", "감정동인", "발명자이유", "최종결과", "재시도횟수",
]

inventor = EmployeeInventorAgent(MODEL)
_lock = threading.Lock()
_done = [0]
_stats = Counter()


def is_valid_response(value):
    return value is not None and str(value).strip() not in ("", "Error", "Parsing Failed")


def crossed_schedule(n, seed):
    """형식(4) × 관대함(4) × 인지모드(2) = 32셀 균등 배정."""
    cells = [(f, g, m) for f in range(4) for g in range(4) for m in ("S1", "S2")]
    per, rem = divmod(n, len(cells))
    sched = [c for c in cells for _ in range(per)]
    r = random.Random(seed)
    extra = cells[:]
    r.shuffle(extra)          # 나머지를 앞쪽 셀에 몰아주지 않는다
    sched += extra[:rem]
    r.shuffle(sched)
    return sched


def run_episode(idx, cell):
    formality, generosity, mode = cell
    rng = random.Random(BASE_SEED * 100003 + idx)   # 에피소드별 독립·재현 가능
    env = JikmuEnv(rng)
    state = env.reset(formality_level=formality, generosity_level=generosity,
                      cognition_mode=mode)
    env.apply_shock()                                # 시나리오는 재시도해도 고정

    decision = emotion = reason = None
    for attempt in range(3):
        decision, emotion, reason = inventor.decide_action(state, mode)
        if is_valid_response(decision):
            break
        with _lock:
            _stats["retry"] += 1
        print(f"  [retry {attempt + 1}/3] ep{idx:04} — {reason}", flush=True)
    else:
        with _lock:
            _stats["dropped"] += 1
        print(f"  [DROP] ep{idx:04} cell={cell} — 3회 실패", flush=True)
        return None

    fair = env.compute_fair_compensation()
    paid = env.compute_amount_paid()
    row = {
        "에피소드ID": idx, "산업": state["industry"], "발명중요도": state["invention_impact"],
        "보상규정형식레벨": state["formality_level"], "보상규정형식설명": state["formality_desc"],
        "보상비율수준": state["generosity_level"], "보상비율라벨": state["generosity_label"],
        "보상비율실현값": state["generosity_realized"], "인지모드": mode,
        "실시료율": state["royalty_rate"], "기준매출만원": state["base_sales"],
        "독점권기여율": state["excl_rate"],
        "출원보너스만원": state["filing_bonus"], "등록보너스만원": state["registration_bonus"],
        "실시보너스만원": state["implementation_bonus"], "실지급액합계만원": paid,
        "충격유형": state["shock_type"], "충격배율": state["shock_multiplier"],
        "사용자이익기초만원": state["employer_interest_base"],
        "사용자이익충격후만원": state["employer_interest_post"],
        "충격전정당보상만원": state["fair_comp_pre_shock"],
        "정당보상추정만원": fair, "격차만원": round(fair - paid, 1),
        "발명자결정": decision, "감정동인": emotion, "발명자이유": reason,
        "최종결과": env.get_outcome(decision), "재시도횟수": attempt,
    }
    with _lock:
        _done[0] += 1
        print(f"[{_done[0]:4}/{N}] ep{idx:04} F{formality} {state['generosity_label']:<4} "
              f"{mode} {state['shock_type']:<6} 격차{row['격차만원']:>10,.0f} → {decision}",
              flush=True)
    return row


def load_done():
    """이미 저장된 에피소드ID를 읽는다. 중단된 실행을 이어서 채우기 위함."""
    if not os.path.exists(OUT):
        return {}
    with open(OUT, encoding="utf-8-sig") as f:
        return {int(r["에피소드ID"]): r for r in csv.DictReader(f)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sched = crossed_schedule(N, BASE_SEED)

    # 증분 저장 + 재개. v10 본 실행이 크레딧 소진으로 119건에서 죽었을 때 완주 시점에만
    # 쓰는 구조였던 탓에 그 119건을 통째로 잃었다. 이제 완료 즉시 flush하고, 재실행하면
    # 이미 있는 에피소드ID는 건너뛴다. 시나리오는 Random(SEED*100003+idx)로 idx에만
    # 의존하므로 이어 붙여도 설계가 어긋나지 않는다.
    done = load_done()
    todo = [i for i in range(N) if (i + 1) not in done]
    if done:
        print(f"♻️  재개: {len(done)}건 보존, {len(todo)}건 남음", flush=True)
    _done[0] = len(done)

    fresh = not done
    f_out = open(OUT, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
    if fresh:
        writer.writeheader()
        f_out.flush()

    rows = list(done.values())
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(run_episode, i + 1, sched[i]): i for i in todo}
            for fu in as_completed(futs):
                r = fu.result()
                if r:
                    with _lock:
                        writer.writerow(r)
                        f_out.flush()
                    rows.append(r)
    finally:
        f_out.close()

    realized = Counter((int(r["보상규정형식레벨"]), int(r["보상비율수준"]), r["인지모드"])
                       for r in rows)

    manifest = {
        "model": MODEL, "backend": os.environ.get("LLM_BACKEND", "openai"),
        "call_shape": os.environ.get("OPENAI_CALL_SHAPE", "legacy"),
        "n_requested": N, "n_realized": len(rows),
        "seed": BASE_SEED, "workers": WORKERS,
        "retries": _stats["retry"], "dropped": _stats["dropped"],
        "comp_ratio_levels": COMP_RATIO_LEVELS,
        "cells_short_of_target": {f"{k}": v for k, v in realized.items() if v != N // 32},
    }
    with open(OUT.replace(".csv", "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n완료: {len(rows)}/{N}건 (재시도 {_stats['retry']}, 드롭 {_stats['dropped']}) → {OUT}")


if __name__ == "__main__":
    main()
