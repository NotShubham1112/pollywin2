# Leaderboard and Evaluation

Part of: [[AISEHack 2.0 - Round 2 - MOC]] · Sources: Rules §7, §8 · Data page

---

## Two Leaderboards

| Leaderboard | Based on | Visibility |
|-------------|----------|------------|
| **Public Leaderboard** | Public test set (representative sample of test data) | Visible during the competition |
| **Private Leaderboard** | Hidden **private test set** | Revealed at the end — **this determines official rankings** |

- Which data belongs to which set is **not disclosed** to participants.
- The final leaderboard determines **qualification for the next stage** and winners.

## How Winners Are Determined

1. Each [[Submission Rules|Submission]] is scored by the evaluation metric stated on the competition website.
2. Potential winners are determined **solely by Private Leaderboard ranking**, subject to rule compliance.
3. **Tie-break:** the Submission entered **first** wins.
4. If a potential winner is disqualified → next highest score is chosen (see [[Disqualification and Conduct]]).

## Practical Implications

- Don't overfit the public leaderboard — it's only a subset of test data.
- Favor robust, generalizable models (matches the spirit of [[Competition Overview]]).
- Use [[PI1M.csv]] to improve generalization rather than chasing public LB micro-gains.

See also: [[Disqualification and Conduct#Winner Notification]]

#evaluation #leaderboard #kaggle
