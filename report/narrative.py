def narrative(R):
    keys = ["abstract", "intro", "dataset", "models", "tokenization", "method", "results", "case_variants", "slices",
            "ambiguous", "per_org", "custom_intro", "custom", "errors", "disagreements", "seen", "noise", "discussion",
            "limitations", "conclusion"]
    return {k: f"[{k} text pending] " * 12 for k in keys}
