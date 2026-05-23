FACTS = [
    "Eelgrass meadows can act as nursery habitat for juvenile fish and invertebrates.",
    "Seagrass ecosystems can improve water clarity by trapping suspended sediments.",
    "Healthy seagrass beds can help stabilize coastlines and reduce erosion.",
    "Rapid declines are often linked to light limitation, warming, eutrophication, or physical disturbance.",
    "Remote sensing works best when field observations anchor the spectral signal to real ecological state.",
    "Sentinel-2 revisit frequency makes it useful for tracking coastal vegetation seasonality.",
]

def get_fact(progress: int) -> str:
    return FACTS[(progress // 15) % len(FACTS)]