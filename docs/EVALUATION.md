# Evaluation design

## Official native games

`scripts/evaluate.py` runs the released agent on both seats in the unmodified
official 1.32.7 engine. Shops and weeds follow the selected engine seed.
Each seat has separate actor and plan state. The wrapper strips seed and
non-game configuration fields from actor inputs.

It records terminal cash, status, selected crop/livestock choices, actual and
planned product totals, final warehouse contents and action time. A finished
game is not automatically a production match. Use several seeds when making
changes; self-play alone does not measure strength against diverse opponents.

## Recorded-production opponents

The project originally replayed opponents' actions. That often failed after
our market actions changed their cash and procurement, producing a weak or
broken opponent rather than the intended high-performing production route.

The later contract environment fixes this:

1. Read complete source observations and the next frame's action. Apply
   official field rules and settlement to reconstruct successful purchases,
   resource withdrawals and product deliveries.
2. Check all 719 source transitions: cash, seed inventory and warehouse
   balances. At midnight, reconcile bag contents and warehouse capacity.
3. Reserve inputs required by the demonstrated production. Remove purchases
   unrelated to that production. Preserve the source shop-opening sequence.
4. In the live challenge, keep the opponent's production schedule fixed but
   let a seller trade its goods in the same market as the candidate.
5. Charge live procurement, land and labor costs. A production shortfall in
   money becomes debt, not free capital. Repay on sales and subtract remaining
   debt at the end. Track warehouse overflow losses explicitly.

```text
candidate: actual farm execution -> live sales -> cash
opponent:  recorded production   -> live sales -> cash - outstanding debt
objective: candidate cash - opponent net cash
```

The scenario engine only forces the source shop sequence; it does not expose
future openings to the candidate. If a source seed is unavailable, the
portable compiler defaults to seed 0 for local randomness and still forces
the recorded shop sequence. This is not an exact native replay of the original
random events. A seed override supports additional weed perturbations.

The public runner uses the historical **baseline rule-based opponent seller**.
It does not claim to run the original opponent's hidden code. The candidate
uses the final forecast-guided seller. Fixing the opponent's production makes
comparisons controlled; native games remain necessary to study adaptive
opponents and two-way production responses.

## Historical final comparison

| Panel | Independent scenarios | Games per version | Predecessor wins | Final wins | Mean margin improvement |
|---|---:|---:|---:|---:|---:|
| Development contracts | 12 | 24 | 18 | 22 | +2,359 |
| Held-out confirmation contracts | 18 | 36 | 30 | 32 | +1,770 |
| Native games against six historical agents | 12 | 24 | 24 | 24 | +171 |

Both seats were run per scenario. The 168 total games cover both versions,
not 168 independent opponent samples. Uncertainty was grouped by source
contract or native scenario. Confirmation and native intervals cross zero.
All 168 historical games passed the selected-plan daily and whole-season
production checks; 1,076,558 market-ledger checks reported no discrepancies.

The development and confirmation contracts were selected from a particular
75-game online corpus. They were not a random draw from the entire tens-of-
thousands replay collection. Original historical opponents are not all
bundled; the public tools can run new authorized replays, while compact
historical tables preserve the reported experiment.

## Online snapshot

Submission 56707930 was queried on October 15, 2026. Of 927 returned episodes,
925 were completed public games, one public episode errored and one was
validation self-play. The 925 completed games contain 535 wins, 390 losses,
mean cash 104,604 and mean margin +890; the reported score was 2226.6.

Sources are the dated [summary](../results/historical/online_summary.json)
and [public game table](../results/historical/online_public_games.csv).
The [report](report/REPORT.md) contains the local result plots and discussion.
