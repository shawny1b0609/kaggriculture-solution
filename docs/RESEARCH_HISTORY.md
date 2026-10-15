# Research path: from isolated optimization to shared-market competition

## 1. Reliable fixed schedules

Early successful submissions used carefully engineered production paths and
rule-based execution. They established a useful floor: workers could buy,
move, care, harvest and deliver reliably. Historical scores around 1911 and
1898 became reference points, not permanent measures of strength.

The limitation was rigidity. Good output on one schedule did not imply a good
product mix against a different opponent or shop sequence. Replacing the
entire schedule frequently traded one rigid strategy for another.

## 2. General executors and macro/micro separation

One direction separated strategic production targets from worker scheduling.
Another tried to switch among pre-built operating strategies. The central
difficulty was execution: a profitable macro target is worthless if feeding,
watering, pickup, delivery, worker placement or input timing breaks.

This led to a more constrained goal: imitate demonstrated production first,
then introduce flexibility only where continuity can be checked.

## 3. Daily decision forests

Public high-scoring replays were reconstructed into daily state transitions.
Routes could join at matching farm states, including joins between different
episodes. An early forest had 263 roots. Daily product-output sequences were
used to remove redundant routes, rather than treating different worker paths
with identical output as meaningful strategic diversity.

The work exposed several execution problems:

| Problem | Useful mechanism |
|---|---|
| Replay actions shifted by one frame | Align `action[t+1]` with transition `t -> t+1` |
| Random weeds block a planned planting | Local dig insertion, spare-worker search and temporary labor |
| Early bulk procurement exhausts working capital | Just-in-time purchases and source-compatible hiring cadence |
| Sale orders crowd out required purchases | Protect farm-order capacity before optional sales |
| Missed inputs propagate into later days | Track and repair local deferred jobs before continuation |
| Source state stops matching after a substitution | Recompile resource needs, deliveries and reference states together |

The useful legacy of the forests is the execution and repair infrastructure.
The number of roots or switching nodes was not a good proxy for competitive
strength: many branches changed little, switched too late, or required
fragile cash flows.

## 4. Node-selection imitation learning and RL

Several experiments learned branch choices, using production profiles, market
features and cash-difference rewards. Inputs were simplified after large
observation vectors made learning difficult. Some selectors concentrated on
one route, and weak route sets imposed a ceiling that a better selector could
not remove.

These experiments are not a trained policy inside the final release. Their
lesson was to check the quality and execution stability of the action space
before investing in longer training. A large decision forest can make the
learning problem larger without offering a profitable response to the rival.

## 5. Fix the evaluation opponent

Replaying an opponent's commands under a changed market could break its
procurement and production. Conversely, giving it free inputs inflated its
money. The later environment fixed demonstrated production and accounted for
costs and debt, while trading in a live shared market. This separated
production capability from cash-flow failures in opponent reconstruction.

Native self-play and real online matches remained separate checks. Local
contract win rates were not treated as estimates of online win rate.

## 6. Market competition becomes the main optimization problem

The project shifted from maximizing own output to maximizing terminal money
difference. Timing a sale changes both agents' opportunities. Public market
inventory, known town demand and visible rival farms contain information
about likely competing supply.

The final learned component predicts future rival sales, not worker actions.
Its outputs support a small scenario search with liquidity and inventory
protections. More elaborate market candidates were not automatically better;
later hurdle-model and additional timing experiments did not justify replacing
the retained predictor. Their unsuccessful results motivated production-side
changes instead of indefinite market-rule patching.

## 7. Compatible flexibility in the final agent

The final solution keeps a reliable single production backbone and changes
small, certified blocks. Synchronized wheat/carrot choices can redirect
compatible short-cycle output. Cow/sheep choices require full-lifetime
simulation because timing and quantities differ even when care actions look
similar. Named cargo transfers and complex dependencies can make a tempting
substitution unsuitable.

Directly replacing long-lived crop industries, such as strawberry with
tomato, was not accepted merely because a local production estimate looked
better. Delivery and continuation compatibility remained necessary. The
result is limited but genuine adaptation rather than a different fixed
recording for every version.

## What remains open

- More compatible production blocks with distributed decision times.
- A general worker scheduler that can satisfy production targets without
  losing delivery, feed or daily maintenance guarantees.
- Better forecasting of bursts and rival responses, calibrated on recent
  data rather than only conditional mean sales.
- Broader paired native evaluation of production changes against responsive
  opponents.

The [full report](report/REPORT.md) gives the measurements and external
references. Selected original implementations under `research/` preserve the
training, certification and contract-evaluation lineage; the supported,
portable command-line interfaces are under `scripts/`.
