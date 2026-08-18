# Semantic cache - false-hit analysis

Seeds: 4  Probes: 12 (8 paraphrase, 4 trap)

## Threshold sweep (the hit-rate vs false-hit trade)

```
threshold true_hits false_hits  misses    n
-------------------------------------------
     0.75         8          4       0   12
     0.80         8          3       1   12
     0.85         7          3       2   12
     0.90         5          2       5   12
     0.95         3          2       7   12
```

## False hits at threshold 0.85 (the finding)

A false hit = the cache served a stored answer to a near-neighbour whose
correct answer differs. Each row is a stale answer a user would have gotten:

| probe query | served (stale) | correct | similarity |
|---|---|---|---|
| What is the capital of Slovakia? | Ljubljana | Bratislava | 1.000 |
| What is the chemical symbol for silver? | Au | Ag | 0.865 |
| Who wrote the play Romeo and Ethel the Pirate's Daughter? | William Shakespeare | a fictional play (from the film Shakespeare in Love) | 0.964 |
