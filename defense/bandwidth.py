# Calculate overhead
def bandwidth(trace):
    total_bytes = sum([abs(p.length) for p in trace])
    return 1.0 * total_bytes / latency(trace)


def latency(trace):
    if len(trace) < 2:
        return 0.0
    return trace[-1].timestamp - trace[0].timestamp

  
def bandwidth_overhead(new, old):
    bw_old = bandwidth(old)
    if bw_old == 0.0:
        return 0.0
    return 1.0 * bandwidth(new) / bw_old


def latency_overhead(new, old):
    lat_old = latency(old)
    if lat_old == 0.0:
        return 0.0
    return 1.0 * latency(new) / lat_old