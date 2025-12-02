def money_formatter(x, pos=None):
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.1f}B $"
    elif x >= 1_000_000:
        return f"{x/1_000_000:.1f}M $"
    elif x >= 1_000:
        return f"{x/1_000:.1f}K $"
    else:
        return f"{x:.0f} $"
