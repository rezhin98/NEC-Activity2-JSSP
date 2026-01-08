import random
import math
import re
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

# Type aliases
Operation = Tuple[int, int]
Job = List[Operation]
JSSP = List[Job]
Chromosome = List[Tuple[int, int]]

# ============================================================================
# DATA LOADING - FROM FILE ONLY
# ============================================================================

def parse_orlib_file(filename='jobshop1.txt'):
    """
    Parse OR-Library jobshop1.txt file and extract all instances.
    Returns a dictionary of instance_name: instance_text
    """
    instances = {}
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        print(f"Read {len(lines)} lines from {filename}")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for instance markers like "instance ft06" or "ft06"
            if 'instance' in line.lower():
                # Extract instance name
                parts = line.split()
                instance_name = None
                for part in parts:
                    # Look for common instance names
                    if any(part.lower().startswith(prefix) for prefix in ['ft', 'la', 'abz', 'orb', 'yn', 'swv']):
                        instance_name = part.lower()
                        break
                
                if instance_name:
                    print(f"  Found instance: {instance_name}")
                    i += 1
                    
                    # Skip empty lines and comments
                    while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith('+') or lines[i].strip().startswith('#')):
                        i += 1
                    
                    # Now we should be at the header line (num_jobs num_machines)
                    if i < len(lines):
                        header_line = lines[i].strip()
                        parts = header_line.split()
                        
                        if len(parts) >= 2:
                            try:
                                num_jobs = int(parts[0])
                                num_machines = int(parts[1])
                                
                                # Collect this line and the next num_jobs lines
                                instance_lines = [header_line]
                                i += 1
                                
                                job_count = 0
                                while i < len(lines) and job_count < num_jobs:
                                    job_line = lines[i].strip()
                                    # Only add lines with actual data (not empty, not comments)
                                    if job_line and not job_line.startswith('+') and not job_line.startswith('#') and not 'instance' in job_line.lower():
                                        # Check if line has numbers
                                        if re.findall(r'\d+', job_line):
                                            instance_lines.append(job_line)
                                            job_count += 1
                                    i += 1
                                
                                # Only store if we got all jobs
                                if job_count == num_jobs:
                                    instances[instance_name] = '\n'.join(instance_lines)
                                    print(f"    ✓ Stored {instance_name}: {num_jobs}x{num_machines}")
                                else:
                                    print(f"    ⚠ Incomplete data for {instance_name}: got {job_count}/{num_jobs} jobs")
                                continue
                            except ValueError:
                                pass
            
            i += 1
        
        if not instances:
            print("\n⚠ WARNING: No instances found with standard markers.")
            print("Attempting alternative parsing method...")
            instances = parse_orlib_alternative(filename)
        
        return instances if instances else None
    
    except FileNotFoundError:
        print(f"ERROR: Could not find '{filename}' in current directory")
        print("Please make sure jobshop1.txt is in the same folder as this script.")
        return None
    except Exception as e:
        print(f"ERROR parsing file: {e}")
        return None


def parse_orlib_alternative(filename='jobshop1.txt'):
    """
    Alternative parser that looks for numerical patterns.
    """
    instances = {}
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        print("  Using alternative parsing method...")
        
        i = 0
        instance_count = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('+') or line.startswith('#'):
                i += 1
                continue
            
            # Look for lines that might be headers (two integers)
            parts = line.split()
            if len(parts) >= 2:
                try:
                    num_jobs = int(parts[0])
                    num_machines = int(parts[1])
                    
                    # Sanity check
                    if 1 <= num_jobs <= 100 and 1 <= num_machines <= 100:
                        instance_count += 1
                        instance_name = f"instance_{instance_count}"
                        
                        # Check if we can find enough job lines
                        instance_lines = [line]
                        temp_i = i + 1
                        jobs_found = 0
                        
                        while temp_i < len(lines) and jobs_found < num_jobs:
                            job_line = lines[temp_i].strip()
                            if job_line and not job_line.startswith('+'):
                                # Check if line contains numbers
                                nums = re.findall(r'-?\d+', job_line)
                                if len(nums) >= num_machines * 2:
                                    instance_lines.append(job_line)
                                    jobs_found += 1
                            temp_i += 1
                        
                        if jobs_found == num_jobs:
                            instances[instance_name] = '\n'.join(instance_lines)
                            print(f"    ✓ Found instance {instance_count}: {num_jobs}x{num_machines}")
                            i = temp_i
                            continue
                
                except ValueError:
                    pass
            
            i += 1
        
        return instances if instances else None
    
    except Exception as e:
        print(f"  ERROR in alternative parser: {e}")
        return None


def load_jssp_from_text(text: str) -> Tuple[JSSP, int, int]:
    """Parse JSSP instance from text with robust error handling."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    if not lines:
        raise ValueError("Empty text provided")
    
    idx = 0
    found = False
    
    # Find the first line with two integers (num_jobs, num_machines)
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) >= 2:
            try:
                nj = int(parts[0])
                nm = int(parts[1])
                if 1 <= nj <= 1000 and 1 <= nm <= 1000:
                    found = True
                    break
            except ValueError:
                pass
        idx += 1
    
    if not found:
        raise ValueError(f"Could not find instance header (num_jobs num_machines)")
    
    parts = lines[idx].split()
    num_jobs = int(parts[0])
    num_machines = int(parts[1])
    
    print(f"  Loading: {num_jobs} jobs, {num_machines} machines")
    
    # Check if we have enough lines
    available_lines = len(lines) - idx - 1
    if available_lines < num_jobs:
        raise ValueError(
            f"Not enough data lines. Need {num_jobs} job lines, "
            f"but only have {available_lines} lines after header"
        )
    
    jobs = []
    for j in range(num_jobs):
        line_idx = idx + 1 + j
        
        if line_idx >= len(lines):
            raise ValueError(f"Job {j}: line index {line_idx} out of range")
        
        # Extract all integers from the line
        nums = list(map(int, re.findall(r"-?\d+", lines[line_idx])))
        
        if len(nums) < 2:
            raise ValueError(f"Job {j}: insufficient data (got {len(nums)} numbers)")
        
        # Create operation pairs (machine, processing_time)
        ops = [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]
        jobs.append(ops)
    
    print(f"  ✓ Parsed {len(jobs)} jobs successfully")
    return jobs, num_jobs, num_machines

# ============================================================================
# SCHEDULING
# ============================================================================

class ScheduleResult:
    def __init__(self, makespan, start_times, end_times):
        self.makespan = makespan
        self.start_times = start_times
        self.end_times = end_times

def decode_chromosome(jobs, chromosome):
    op_data = {}
    for j, job in enumerate(jobs):
        for o, (machine, proc) in enumerate(job):
            op_data[(j, o)] = (machine, proc)
    
    next_op = [0] * len(jobs)
    machine_avail = {}
    start_times = {}
    end_times = {}
    
    for job_id, op_idx in chromosome:
        if op_idx != next_op[job_id]:
            continue
        machine, proc = op_data[(job_id, op_idx)]
        avail_m = machine_avail.get(machine, 0)
        avail_j = end_times.get((job_id, op_idx - 1), 0) if op_idx > 0 else 0
        start = max(avail_m, avail_j)
        end = start + proc
        start_times[(job_id, op_idx)] = start
        end_times[(job_id, op_idx)] = end
        machine_avail[machine] = end
        next_op[job_id] += 1
    
    remaining = [(j, next_op[j]) for j in range(len(jobs)) if next_op[j] < len(jobs[j])]
    for job_id, op_idx in sorted(remaining):
        machine, proc = op_data[(job_id, op_idx)]
        avail_m = machine_avail.get(machine, 0)
        avail_j = end_times.get((job_id, op_idx - 1), 0) if op_idx > 0 else 0
        start = max(avail_m, avail_j)
        end = start + proc
        start_times[(job_id, op_idx)] = start
        end_times[(job_id, op_idx)] = end
        machine_avail[machine] = end
    
    return ScheduleResult(max(end_times.values()) if end_times else 0, start_times, end_times)

# ============================================================================
# GENETIC OPERATORS
# ============================================================================

def generate_initial_population(jobs, pop_size):
    ops = [(j, o) for j, job in enumerate(jobs) for o in range(len(job))]
    return [random.sample(ops, len(ops)) for _ in range(pop_size)]

def order_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    seg = p1[a:b+1]
    child = [None] * size
    child[a:b+1] = seg
    p2_filtered = [g for g in p2 if g not in seg]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_filtered[idx]
            idx += 1
    return child

def pmx_crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b+1] = p1[a:b+1]
    mapping = {p2[i]: p1[i] for i in range(a, b + 1)}
    for i in range(size):
        if child[i] is None:
            gene = p2[i]
            while gene in child[a:b+1]:
                gene = mapping.get(gene, gene)
                if gene not in mapping:
                    break
            child[i] = gene
    return child

def position_crossover(p1, p2):
    size = len(p1)
    positions = random.sample(range(size), size // 3)
    child = [None] * size
    for pos in positions:
        child[pos] = p1[pos]
    p2_filtered = [g for g in p2 if g not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_filtered[idx]
            idx += 1
    return child

def swap_mutation(chrom, rate):
    if random.random() < rate:
        chrom = chrom[:]
        i, j = random.sample(range(len(chrom)), 2)
        chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom

def insert_mutation(chrom, rate):
    if random.random() < rate:
        i, j = sorted(random.sample(range(len(chrom)), 2))
        gene = chrom[i]
        chrom = chrom[:i] + chrom[i+1:]
        chrom.insert(j, gene)
    return chrom

def inversion_mutation(chrom, rate):
    if random.random() < rate:
        chrom = chrom[:]
        i, j = sorted(random.sample(range(len(chrom)), 2))
        chrom[i:j+1] = reversed(chrom[i:j+1])
    return chrom

def scramble_mutation(chrom, rate):
    if random.random() < rate:
        chrom = chrom[:]
        i, j = sorted(random.sample(range(len(chrom)), 2))
        seg = chrom[i:j+1]
        random.shuffle(seg)
        chrom[i:j+1] = seg
    return chrom

def tournament_selection(pop, fitness_map, k=3):
    comps = random.sample(pop, k)
    return min(comps, key=lambda c: fitness_map[tuple(c)])

def roulette_selection(pop, fitness_map):
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

def rank_selection(pop, fitness_map):
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
# GA
# ============================================================================

def genetic_algorithm(jobs, pop_size=100, gens=200, cx_rate=0.9, mut_rate=0.2,
                     cx_method="order", mut_method="swap", sel_method="tournament", k=3, verbose=True):
    cross_fn = {"order": order_crossover, "pmx": pmx_crossover, "position": position_crossover}[cx_method]
    mut_fn = {"swap": swap_mutation, "insert": insert_mutation, "inversion": inversion_mutation, "scramble": scramble_mutation}[mut_method]
    
    pop = generate_initial_population(jobs, pop_size)
    best, best_ms = None, math.inf
    history = []
    
    for gen in range(gens):
        evals = [(c, decode_chromosome(jobs, c).makespan) for c in pop]
        evals.sort(key=lambda x: x[1])
        fit_map = {tuple(c): ms for c, ms in evals}
        
        if evals[0][1] < best_ms:
            best, best_ms = evals[0][0], evals[0][1]
        history.append(best_ms)
        
        if verbose and gen % 50 == 0:
            print(f"  Gen {gen:3d}: {best_ms}")
        
        new_pop = [evals[0][0]]
        while len(new_pop) < pop_size:
            if sel_method == "tournament":
                p1, p2 = tournament_selection([c for c, _ in evals], fit_map, k), tournament_selection([c for c, _ in evals], fit_map, k)
            elif sel_method == "roulette":
                p1, p2 = roulette_selection([c for c, _ in evals], fit_map), roulette_selection([c for c, _ in evals], fit_map)
            else:
                p1, p2 = rank_selection([c for c, _ in evals], fit_map), rank_selection([c for c, _ in evals], fit_map)
            child = cross_fn(p1, p2) if random.random() < cx_rate else p1[:]
            child = mut_fn(child, mut_rate)
            new_pop.append(child)
        pop = new_pop
    
    if verbose:
        print(f"  Final: {best_ms}")
    return best, decode_chromosome(jobs, best), history

# ============================================================================
# SA
# ============================================================================

def simulated_annealing(jobs, init_chrom, T0=1000, Tf=0.1, alpha=0.95, iters=50):
    curr = init_chrom[:]
    curr_ms = decode_chromosome(jobs, curr).makespan
    best, best_ms = curr, curr_ms
    T = T0
    history = []
    
    while T > Tf:
        for _ in range(iters):
            i, j = random.sample(range(len(curr)), 2)
            neighbor = curr[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neigh_ms = decode_chromosome(jobs, neighbor).makespan
            delta = neigh_ms - curr_ms
            if delta <= 0 or random.random() < math.exp(-delta / T):
                curr, curr_ms = neighbor, neigh_ms
                if neigh_ms < best_ms:
                    best, best_ms = neighbor, neigh_ms
            history.append(best_ms)
        T *= alpha
    return best, decode_chromosome(jobs, best), history

# ============================================================================
# EXPERIMENTS
# ============================================================================

class Config:
    def __init__(self, name, pop, gens, cx_r, mut_r, cx_m, mut_m, sel_m, k=3):
        self.name, self.pop, self.gens, self.cx_r, self.mut_r = name, pop, gens, cx_r, mut_r
        self.cx_m, self.mut_m, self.sel_m, self.k = cx_m, mut_m, sel_m, k

def get_configs():
    return [
        Config("C1: OX+Swap+Tour", 100, 200, 0.9, 0.2, "order", "swap", "tournament", 3),
        Config("C2: PMX+Insert+Roul", 120, 200, 0.85, 0.25, "pmx", "insert", "roulette"),
        Config("C3: Pos+Inv+Rank", 150, 150, 0.8, 0.3, "position", "inversion", "rank"),
        Config("C4: OX+Scram+Tour", 200, 150, 0.9, 0.15, "order", "scramble", "tournament", 5),
        Config("C5: PMX+Swap+Roul", 100, 250, 0.85, 0.4, "pmx", "swap", "roulette"),
        Config("C6: Pos+Insert+Rank", 150, 180, 0.88, 0.22, "position", "insert", "rank")
    ]

def run_experiments(name, jobs, nj, nm):
    print(f"\n{'='*80}\nDATASET: {name} ({nj}x{nm})\n{'='*80}\n")
    results = []
    for cfg in get_configs():
        print(f"\n--- {cfg.name} ---")
        _, res, hist = genetic_algorithm(jobs, cfg.pop, cfg.gens, cfg.cx_r, cfg.mut_r,
                                        cfg.cx_m, cfg.mut_m, cfg.sel_m, cfg.k, True)
        results.append({'cfg': cfg, 'ms': res.makespan, 'hist': hist})
        print(f"✓ Makespan: {res.makespan}\n")
    
    print(f"\n{'='*80}\nSUMMARY - {name}\n{'='*80}")
    print(f"{'Config':<30} {'Pop':<6} {'Gen':<6} {'Makespan':<10}")
    print('-'*80)
    for r in results:
        print(f"{r['cfg'].name:<30} {r['cfg'].pop:<6} {r['cfg'].gens:<6} {r['ms']:<10}")
    print('='*80)
    
    best = min(results, key=lambda x: x['ms'])
    print(f"\n🏆 BEST: {best['cfg'].name} - {best['ms']}\n")
    
    plt.figure(figsize=(10, 6))
    plt.plot(best['hist'], linewidth=2, color='#2E86AB')
    plt.xlabel('Generation')
    plt.ylabel('Best Makespan')
    plt.title(f'Evolution - {name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"evolution_{name}.png", dpi=300)
    print(f"📊 Saved evolution_{name}.png")
    plt.close()
    
    plt.figure(figsize=(12, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    for i, r in enumerate(results):
        plt.plot(r['hist'], label=f"{r['cfg'].name} ({r['ms']})", lw=1.5, color=colors[i], alpha=0.8)
    plt.xlabel('Generation')
    plt.ylabel('Best Makespan')
    plt.title(f'Comparison - {name}')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"comparison_{name}.png", dpi=300)
    print(f"📊 Saved comparison_{name}.png")
    plt.close()
    
    return results, best

def main():
    print("\n" + "="*80)
    print("JOB SHOP SCHEDULING - GA EXPERIMENTS")
    print("Loading from: jobshop1.txt")
    print("="*80)
    
    # Load datasets from file ONLY
    print("\nReading jobshop1.txt file...")
    datasets = parse_orlib_file('jobshop1.txt')
    
    if datasets is None or len(datasets) == 0:
        print("\n❌ ERROR: Cannot proceed without jobshop1.txt file!")
        print("Please ensure jobshop1.txt is in the same directory as this script.")
        print("\nYou can download it from:")
        print("http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/jobshop1.txt")
        return
    
    print(f"\n✓ Successfully loaded {len(datasets)} instances from jobshop1.txt")
    print(f"Available instances: {list(datasets.keys())[:10]}...")
    
    # Select instances: ft06 (6x6), la16 (10x10), abz7 (20x15)
    # OR use first 3 instances if named instances not found
    dataset_order = ['ft06', 'la16', 'abz7']
    
    # Check which instances are available
    available = []
    for name in dataset_order:
        if name in datasets:
            available.append(name)
    
    # If we don't have the expected instances, use what we have
    if len(available) < 3:
        print(f"\n⚠ Warning: Not all expected instances found.")
        print(f"Expected: {dataset_order}")
        print(f"Found: {available}")
        print(f"\nUsing first 3 available instances instead...")
        available = list(datasets.keys())[:3]
    
    if not available:
        print("❌ No valid instances found!")
        return
    
    print(f"\nProcessing instances: {available}\n")
    
    all_res = {}
    
    for name in available:
        try:
            print(f"\n{'='*80}")
            print(f"Processing: {name}")
            print('='*80)
            jobs, nj, nm = load_jssp_from_text(datasets[name])
            res, best = run_experiments(name, jobs, nj, nm)
            all_res[name] = (res, best, jobs)
        except Exception as e:
            print(f"\n❌ ERROR processing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_res:
        print("\n❌ No instances were successfully processed!")
        return
    
    print(f"\n{'='*80}\nSIMULATED ANNEALING\n{'='*80}\n")
    for name in available:
        if name in all_res:
            res, best, jobs = all_res[name]
            print(f"--- {name} ---")
            init_sol = generate_initial_population(jobs, 1)[0]
            _, sa_res, _ = simulated_annealing(jobs, init_sol, 1000, 0.1, 0.95, 50)
            print(f"GA: {best['ms']} | SA: {sa_res.makespan}")
            imp = best['ms'] - sa_res.makespan
            print(f"{'Improved' if imp > 0 else 'No change'}\n")
    
    print(f"{'='*80}\n✅ COMPLETE!\n{'='*80}\n")

if __name__ == "__main__":
    main()