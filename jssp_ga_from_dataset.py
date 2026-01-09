import random
import math
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

# -----------------------------
# Types
# -----------------------------
Operation = Tuple[int, int]        # (machine, processing_time)
Job = List[Operation]
JSSP = List[Job]
Chromosome = List[int]             # job-based chromosome: list of job IDs


# ============================================================================
# OR-LIBRARY INSTANCE LOADING (BY NAME)
# ============================================================================

KNOWN_PREFIXES = ("ft", "la", "abz", "orb", "swv", "yn")


def _is_comment_or_empty(line: str) -> bool:
    s = line.strip()
    return (not s) or s.startswith("+") or s.startswith("#")


def scan_available_instance_names(filename: str) -> List[str]:
    """Scan the file and list likely instance-name lines."""
    names = []
    seen = set()
    with open(filename, "r") as f:
        for raw in f:
            s = raw.strip().lower()
            if _is_comment_or_empty(s):
                continue

            # Case 1: "instance ft06" style
            if "instance" in s:
                parts = s.split()
                for p in parts:
                    if p.startswith(KNOWN_PREFIXES) and any(ch.isdigit() for ch in p):
                        if p not in seen:
                            names.append(p)
                            seen.add(p)

            # Case 2: "ft06" alone in a line
            if s.startswith(KNOWN_PREFIXES) and any(ch.isdigit() for ch in s) and len(s.split()) == 1:
                if s not in seen:
                    names.append(s)
                    seen.add(s)

    return names


def extract_instance_text(filename: str, instance_name: str) -> str:
    """
    Extract one OR-Library instance by name (e.g., 'la01', 'la16', 'abz7').

    Strategy:
    - Find a line that contains the instance name (either standalone or after the word "instance")
    - From the following non-empty, non-comment line: parse header 'num_jobs num_machines'
    - Then read next num_jobs data lines
    """
    target = instance_name.strip().lower()
    with open(filename, "r") as f:
        lines = f.readlines()

    # Find marker line
    marker_idx = None
    for i, raw in enumerate(lines):
        s = raw.strip().lower()
        if _is_comment_or_empty(s):
            continue

        # "instance la16" or similar
        if "instance" in s and target in s.split():
            marker_idx = i
            break

        # name alone: "la16"
        if s == target:
            marker_idx = i
            break

    if marker_idx is None:
        available = scan_available_instance_names(filename)
        raise ValueError(
            f"Instance '{instance_name}' not found in {filename}.\n"
            f"Some available names: {available[:20]}"
        )

    # Find header line after marker
    j = marker_idx + 1
    while j < len(lines) and _is_comment_or_empty(lines[j]):
        j += 1

    if j >= len(lines):
        raise ValueError(
            f"Found marker for '{instance_name}' but no header after it.")

    header = lines[j].strip()
    parts = header.split()
    if len(parts) < 2:
        raise ValueError(
            f"Header after '{instance_name}' is invalid: '{header}'")

    try:
        num_jobs = int(parts[0])
        num_machines = int(parts[1])
    except ValueError:
        raise ValueError(
            f"Header after '{instance_name}' is not two integers: '{header}'")

    # Collect job lines (skip blank/comment lines)
    job_lines = [header]
    j += 1
    jobs_found = 0

    while j < len(lines) and jobs_found < num_jobs:
        s = lines[j].strip()
        if not _is_comment_or_empty(s):
            nums = re.findall(r"-?\d+", s)
            if len(nums) >= 2:  # at least something numeric
                job_lines.append(s)
                jobs_found += 1
        j += 1

    if jobs_found != num_jobs:
        raise ValueError(
            f"Instance '{instance_name}': expected {num_jobs} job lines, got {jobs_found}."
        )

    return "\n".join(job_lines)


def load_jssp_from_text(text: str) -> Tuple[JSSP, int, int]:
    """
    Parse JSSP instance from text:
    - header line: num_jobs num_machines
    - then num_jobs lines, each containing 2*num_machines integers: (machine, time) pairs
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty text provided")

    header = lines[0].split()
    if len(header) < 2:
        raise ValueError("Missing header (num_jobs num_machines)")

    num_jobs = int(header[0])
    num_machines = int(header[1])

    if len(lines) < 1 + num_jobs:
        raise ValueError(
            f"Need {num_jobs} job lines after header, but only have {len(lines)-1}")

    jobs: JSSP = []
    for j in range(num_jobs):
        nums = list(map(int, re.findall(r"-?\d+", lines[1 + j])))

        # OR-Lib JSSP typically has exactly 2*num_machines ints per job line
        if len(nums) < 2 * num_machines:
            raise ValueError(
                f"Job {j}: expected at least {2*num_machines} integers, got {len(nums)}"
            )

        nums = nums[: 2 * num_machines]  # ignore extra tokens if any
        ops = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
        jobs.append(ops)

    return jobs, num_jobs, num_machines


# ============================================================================
# SCHEDULING / DECODING (JOB-BASED CHROMOSOME)
# ============================================================================

class ScheduleResult:
    def __init__(self, makespan: int, start_times: Dict[Tuple[int, int], int], end_times: Dict[Tuple[int, int], int]):
        self.makespan = makespan
        self.start_times = start_times
        self.end_times = end_times


def decode_chromosome(jobs: JSSP, chromosome: Chromosome) -> ScheduleResult:
    """
    Job-based decoding:
    chromosome = list of job IDs (each job repeated once per operation)
    Each time job j appears, schedule its next unscheduled operation.
    """
    num_jobs = len(jobs)
    next_op = [0] * num_jobs
    machine_avail: Dict[int, int] = {}
    job_avail: Dict[int, int] = {j: 0 for j in range(num_jobs)}

    start_times: Dict[Tuple[int, int], int] = {}
    end_times: Dict[Tuple[int, int], int] = {}

    for job_id in chromosome:
        op_idx = next_op[job_id]
        if op_idx >= len(jobs[job_id]):
            continue  # should not happen if chromosome counts are correct

        machine, proc = jobs[job_id][op_idx]
        start = max(machine_avail.get(machine, 0), job_avail[job_id])
        end = start + proc

        start_times[(job_id, op_idx)] = start
        end_times[(job_id, op_idx)] = end

        machine_avail[machine] = end
        job_avail[job_id] = end
        next_op[job_id] += 1

    makespan = max(end_times.values()) if end_times else 0
    return ScheduleResult(makespan, start_times, end_times)


# ============================================================================
# INITIAL POPULATION (JOB-BASED)
# ============================================================================

def job_counts(jobs: JSSP) -> List[int]:
    return [len(job) for job in jobs]


def generate_initial_population(jobs: JSSP, pop_size: int) -> List[Chromosome]:
    counts = job_counts(jobs)
    base = []
    for j, c in enumerate(counts):
        base.extend([j] * c)

    pop = []
    for _ in range(pop_size):
        chrom = base[:]
        random.shuffle(chrom)
        pop.append(chrom)
    return pop


# ============================================================================
# CROSSOVERS FOR DUPLICATE GENES (JOB-BASED)
# ============================================================================

def segment_fill_crossover(p1: Chromosome, p2: Chromosome, counts: List[int]) -> Chromosome:
    """
    Two-point segment from p1; fill remaining positions from p2 in order,
    respecting job multiplicities (counts).
    """
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b+1] = p1[a:b+1]

    remaining = counts[:]
    for g in child[a:b+1]:
        remaining[g] -= 1

    idx = 0
    for i in range(size):
        if child[i] is not None:
            continue
        while idx < size and remaining[p2[idx]] == 0:
            idx += 1
        child[i] = p2[idx]
        remaining[p2[idx]] -= 1
        idx += 1

    return child  # type: ignore


def position_crossover_dups(p1: Chromosome, p2: Chromosome, counts: List[int]) -> Chromosome:
    """
    Keep a random set of positions from p1; fill the rest from p2 in order,
    respecting job multiplicities.
    """
    size = len(p1)
    keep = set(random.sample(range(size), max(1, size // 3)))
    child = [None] * size

    remaining = counts[:]
    for i in keep:
        g = p1[i]
        child[i] = g
        remaining[g] -= 1

    idx = 0
    for i in range(size):
        if child[i] is not None:
            continue
        while idx < size and remaining[p2[idx]] == 0:
            idx += 1
        child[i] = p2[idx]
        remaining[p2[idx]] -= 1
        idx += 1

    return child  # type: ignore


def uniform_crossover_repair(p1: Chromosome, p2: Chromosome, counts: List[int]) -> Chromosome:
    """
    Uniform crossover with repair:
    - First pass: pick p1/p2 gene with 50/50 chance if still available
    - Second pass: fill remaining slots with leftover genes randomly
    """
    size = len(p1)
    remaining = counts[:]
    child = [None] * size

    for i in range(size):
        cand = p1[i] if random.random() < 0.5 else p2[i]
        if remaining[cand] > 0:
            child[i] = cand
            remaining[cand] -= 1

    leftovers = []
    for job_id, rem in enumerate(remaining):
        leftovers.extend([job_id] * rem)
    random.shuffle(leftovers)

    li = 0
    for i in range(size):
        if child[i] is None:
            child[i] = leftovers[li]
            li += 1

    return child  # type: ignore


# ============================================================================
# MUTATIONS
# ============================================================================

def swap_mutation(chrom: Chromosome, rate: float) -> Chromosome:
    if random.random() < rate:
        chrom = chrom[:]
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


def insert_mutation(chrom: Chromosome, rate: float) -> Chromosome:
    if random.random() < rate:
        chrom = chrom[:]
        i, j = sorted(random.sample(range(len(chrom)), 2))
        gene = chrom.pop(i)
        chrom.insert(j, gene)
    return chrom


def inversion_mutation(chrom: Chromosome, rate: float) -> Chromosome:
    if random.random() < rate:
        chrom = chrom[:]
        i, j = sorted(random.sample(range(len(chrom)), 2))
        chrom[i:j+1] = reversed(chrom[i:j+1])
    return chrom


def scramble_mutation(chrom: Chromosome, rate: float) -> Chromosome:
    if random.random() < rate:
        chrom = chrom[:]
        i, j = sorted(random.sample(range(len(chrom)), 2))
        seg = chrom[i:j+1]
        random.shuffle(seg)
        chrom[i:j+1] = seg
    return chrom


# ============================================================================
# SELECTIONS
# ============================================================================

def tournament_selection(pop: List[Chromosome], fitness_map: Dict[Tuple[int, ...], int], k: int = 3) -> Chromosome:
    comps = random.sample(pop, k)
    return min(comps, key=lambda c: fitness_map[tuple(c)])


def roulette_selection(pop: List[Chromosome], fitness_map: Dict[Tuple[int, ...], int]) -> Chromosome:
    max_fit = max(fitness_map.values())
    adj_fit = [max_fit - fitness_map[tuple(c)] + 1 for c in pop]
    total = sum(adj_fit)
    if total == 0:
        return random.choice(pop)
    r = random.random() * total
    cumsum = 0
    for i, f in enumerate(adj_fit):
        cumsum += f
        if r <= cumsum:
            return pop[i]
    return pop[-1]


def rank_selection(pop: List[Chromosome], fitness_map: Dict[Tuple[int, ...], int]) -> Chromosome:
    sorted_pop = sorted(pop, key=lambda c: fitness_map[tuple(c)])
    total = len(sorted_pop) * (len(sorted_pop) + 1) // 2
    r = random.random() * total
    cumsum = 0
    for i, chrom in enumerate(sorted_pop):
        cumsum += i + 1
        if r <= cumsum:
            return chrom
    return sorted_pop[-1]


# ============================================================================
# GENETIC ALGORITHM (WITH STATIONARY DETECTION)
# ============================================================================

def genetic_algorithm(
    jobs: JSSP,
    pop_size: int = 100,
    gens: int = 200,
    cx_rate: float = 0.9,
    mut_rate: float = 0.2,
    cx_method: str = "segment",
    mut_method: str = "swap",
    sel_method: str = "tournament",
    k: int = 3,
    patience: int = 50,
    verbose: bool = True
):
    counts = job_counts(jobs)

    cross_fn = {
        "segment": segment_fill_crossover,
        "position": position_crossover_dups,
        "uniform": uniform_crossover_repair
    }[cx_method]

    mut_fn = {
        "swap": swap_mutation,
        "insert": insert_mutation,
        "inversion": inversion_mutation,
        "scramble": scramble_mutation
    }[mut_method]

    pop = generate_initial_population(jobs, pop_size)

    best: Optional[Chromosome] = None
    best_ms = math.inf
    history: List[int] = []
    no_improve = 0

    for gen in range(gens):
        evals = [(c, decode_chromosome(jobs, c).makespan) for c in pop]
        evals.sort(key=lambda x: x[1])
        fit_map = {tuple(c): ms for c, ms in evals}

        if evals[0][1] < best_ms:
            best = evals[0][0]
            best_ms = evals[0][1]
            no_improve = 0
        else:
            no_improve += 1

        history.append(best_ms)

        if verbose and (gen % 50 == 0 or gen == gens - 1):
            print(f"  Gen {gen:3d}: best={best_ms} | no_improve={no_improve}")

        # stationary stopping
        if no_improve >= patience:
            if verbose:
                print(
                    f"  Stopped early (stationary): no improvement for {patience} generations.")
            break

        # elitism: keep best in current generation
        new_pop = [evals[0][0]]

        pool = [c for c, _ in evals]
        while len(new_pop) < pop_size:
            if sel_method == "tournament":
                p1 = tournament_selection(pool, fit_map, k)
                p2 = tournament_selection(pool, fit_map, k)
            elif sel_method == "roulette":
                p1 = roulette_selection(pool, fit_map)
                p2 = roulette_selection(pool, fit_map)
            else:
                p1 = rank_selection(pool, fit_map)
                p2 = rank_selection(pool, fit_map)

            if random.random() < cx_rate:
                child = cross_fn(p1, p2, counts)
            else:
                child = p1[:]

            child = mut_fn(child, mut_rate)
            new_pop.append(child)

        pop = new_pop

    if best is None:
        best = pop[0]
    return best, decode_chromosome(jobs, best), history


# ============================================================================
# SIMULATED ANNEALING (OPTIONAL METHOD)
# ============================================================================

def simulated_annealing(jobs: JSSP, init_chrom: Chromosome, T0=1000, Tf=0.1, alpha=0.95, iters=50):
    curr = init_chrom[:]
    curr_ms = decode_chromosome(jobs, curr).makespan
    best, best_ms = curr[:], curr_ms

    T = T0
    history = []

    while T > Tf:
        for _ in range(iters):
            i, j = random.sample(range(len(curr)), 2)
            neigh = curr[:]
            neigh[i], neigh[j] = neigh[j], neigh[i]
            neigh_ms = decode_chromosome(jobs, neigh).makespan

            delta = neigh_ms - curr_ms
            if delta <= 0 or random.random() < math.exp(-delta / T):
                curr, curr_ms = neigh, neigh_ms
                if curr_ms < best_ms:
                    best, best_ms = curr[:], curr_ms

            history.append(best_ms)
        T *= alpha

    return best, decode_chromosome(jobs, best), history


# ============================================================================
# EXPERIMENT CONFIGS
# ============================================================================

@dataclass
class Config:
    name: str
    pop: int
    gens: int
    cx_r: float
    mut_r: float
    cx_m: str
    mut_m: str
    sel_m: str
    k: int = 3
    patience: int = 50


def get_configs() -> List[Config]:
    # 6 different combinations (meets the requirement)
    return [
        Config("C1: Seg+Swap+Tour",    100, 250, 0.90, 0.20,
               "segment",  "swap",     "tournament", 3, 60),
        Config("C2: Seg+Insert+Roul",  120, 250, 0.85, 0.25,
               "segment",  "insert",   "roulette",   3, 60),
        Config("C3: Pos+Inv+Rank",     150, 220, 0.80, 0.30,
               "position", "inversion", "rank",       3, 60),
        Config("C4: Pos+Scram+Tour",   200, 200, 0.90, 0.15,
               "position", "scramble", "tournament", 5, 60),
        Config("C5: Uni+Swap+Roul",    120, 250, 0.85, 0.30,
               "uniform",  "swap",     "roulette",   3, 60),
        Config("C6: Uni+Insert+Rank",  150, 220, 0.88, 0.22,
               "uniform",  "insert",   "rank",       3, 60),
    ]


# ============================================================================
# RUN EXPERIMENTS + PLOTS
# ============================================================================

def run_experiments(instance_name: str, jobs: JSSP, nj: int, nm: int):
    print(f"\n{'='*80}\nDATASET: {instance_name} ({nj}x{nm})\n{'='*80}\n")

    results = []
    for cfg in get_configs():
        print(f"\n--- {cfg.name} ---")
        _, res, hist = genetic_algorithm(
            jobs,
            pop_size=cfg.pop,
            gens=cfg.gens,
            cx_rate=cfg.cx_r,
            mut_rate=cfg.mut_r,
            cx_method=cfg.cx_m,
            mut_method=cfg.mut_m,
            sel_method=cfg.sel_m,
            k=cfg.k,
            patience=cfg.patience,
            verbose=True
        )
        results.append({"cfg": cfg, "ms": res.makespan, "hist": hist})
        print(f"✓ Makespan: {res.makespan}\n")

    # Summary table
    print(f"\n{'='*80}\nSUMMARY - {instance_name}\n{'='*80}")
    print(f"{'Config':<28} {'Pop':<6} {'Gen(max)':<9} {'Stops?':<8} {'Makespan':<10}")
    print("-"*80)
    for r in results:
        cfg = r["cfg"]
        ran = len(r["hist"])
        stopped = "yes" if ran < cfg.gens else "no"
        print(
            f"{cfg.name:<28} {cfg.pop:<6} {cfg.gens:<9} {stopped:<8} {r['ms']:<10}")
    print("="*80)

    best = min(results, key=lambda x: x["ms"])
    print(f"\n🏆 BEST: {best['cfg'].name} - {best['ms']}\n")

    # Plot best evolution
    plt.figure(figsize=(10, 6))
    plt.plot(best["hist"], linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best Makespan")
    plt.title(f"Evolution (Best Config) - {instance_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"evolution_{instance_name}.png", dpi=300)
    plt.close()
    print(f" Saved evolution_{instance_name}.png")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    for r in results:
        plt.plot(r["hist"], lw=1.5, alpha=0.85,
                 label=f"{r['cfg'].name} ({r['ms']})")
    plt.xlabel("Generation")
    plt.ylabel("Best Makespan")
    plt.title(f"Comparison - {instance_name}")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"comparison_{instance_name}.png", dpi=300)
    plt.close()
    print(f"Saved comparison_{instance_name}.png")

    return results, best


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("JOB SHOP SCHEDULING - GA EXPERIMENTS (FIXED VERSION)")
    print("Loading from: jobshop1.txt")
    print("="*80)

    filename = "jobshop1.txt"

    # Required sizes (matches teacher recommendation):
    # - 3-5 machines: la01 (10x5)
    # - ~10 machines: la16 (10x10)
    # - 15+ machines: abz7 (20x15)
    dataset_order = ["la01", "la16", "abz7"]

    all_res = {}

    for name in dataset_order:
        print(f"\n{'='*80}\nProcessing: {name}\n{'='*80}")
        try:
            instance_text = extract_instance_text(filename, name)
            jobs, nj, nm = load_jssp_from_text(instance_text)
            res, best = run_experiments(name, jobs, nj, nm)
            all_res[name] = (res, best, jobs)
        except Exception as e:
            print(f"\n ERROR processing {name}: {e}")
            continue

    if not all_res:
        print("\n No instances were successfully processed!")
        return

    print(f"\n{'='*80}\nSIMULATED ANNEALING (OPTIONAL)\n{'='*80}\n")
    for name in dataset_order:
        if name not in all_res:
            continue
        res, best, jobs = all_res[name]
        init_sol = generate_initial_population(jobs, 1)[0]
        _, sa_res, _ = simulated_annealing(
            jobs, init_sol, T0=1000, Tf=0.1, alpha=0.95, iters=50)

        print(f"--- {name} ---")
        print(f"Best GA: {best['ms']} | SA: {sa_res.makespan}")
        diff = best["ms"] - sa_res.makespan
        print(
            f"{'SA better' if diff > 0 else 'GA better or equal'} (difference: {abs(diff)})\n")

    print(f"{'='*80}\n COMPLETE!\n{'='*80}\n")


if __name__ == "__main__":
    main()
