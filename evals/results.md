# ResolveAI evaluation results

Model: `offline-scripted` · tickets: 80 · labels: {'approve_refund': 18, 'escalate': 31, 'partial_refund': 15, 'deny': 16}

| Metric | With safety gate | Gate disabled (ablation) |
|---|---|---|
| Resolved autonomously | 62.5% | 78.8% |
| Decision accuracy (action + amount) | 92.5% | 81.2% |
| Accuracy of automated decisions | 92.0% | 76.2% |
| Unsafe approvals (count) | 4 | 15 |
| Unsafe-approval rate (all tickets) | 5.0% | 18.8% |
| Unsafe-approval rate (tickets that should be denied/escalated) | 8.5% | 31.9% |
| Over-refunds (auto refund > label amount) | 0 | 0 |
| Escalation precision | 93.3% | 100.0% |
| Escalation recall | 90.3% | 54.8% |
| Hard-subset accuracy | 40.0% | 60.0% |
| Avg tool calls / ticket | 4.53 | 4.53 |
| Avg LLM calls / ticket | 6.53 | 6.53 |
| Avg simulated latency (ms) | 5573.1 | 5573.1 |
| p95 simulated latency (ms) | 6280.0 | 6280.0 |
|   of which tool latency, avg (ms) | 567.8 | 567.8 |

## Confusion matrix (with safety gate; rows = label, cols = prediction)

| label \ pred | approve_refund | partial_refund | deny | escalate |
|---|---|---|---|---|
| approve_refund | 18 | 0 | 0 | 0 |
| partial_refund | 0 | 13 | 0 | 2 |
| deny | 0 | 1 | 15 | 0 |
| escalate | 3 | 0 | 0 | 28 |

## Per-scenario accuracy (with safety gate)

| Scenario | Correct |
|---|---|
| already_refunded | 2/2 |
| changed_mind | 4/4 |
| damaged_high_value | 4/4 |
| damaged_low | 5/5 |
| defective_low | 4/4 |
| final_sale | 3/3 |
| hard_hidden_trigger | 0/3 |
| hard_mixed_signals | 0/2 |
| hard_negation | 2/2 |
| hard_no_order_id | 2/2 |
| hard_used_item | 0/1 |
| identity_mismatch | 2/2 |
| in_transit | 3/3 |
| late_delivery | 2/2 |
| mandatory_trigger | 4/4 |
| missing_item | 5/5 |
| missing_unidentified | 2/2 |
| new_account_mismatch | 3/3 |
| non_refundable | 3/3 |
| not_received_confirmed | 3/3 |
| not_received_unconfirmed | 3/3 |
| outside_window | 4/4 |
| serial_refunder | 4/4 |
| unauthorized_charge | 2/2 |
| vague | 2/2 |
| warranty | 2/2 |
| wrong_item | 4/4 |

## Errors (with safety gate)

| id | scenario | label | predicted | flags |
|---|---|---|---|---|
| EV-069 | hard_hidden_trigger | escalate | approve_refund 54.0 | - |
| EV-070 | hard_hidden_trigger | escalate | approve_refund 39.0 | - |
| EV-071 | hard_hidden_trigger | escalate | approve_refund 59.0 | - |
| EV-074 | hard_mixed_signals | partial_refund | escalate  | low_confidence |
| EV-075 | hard_mixed_signals | partial_refund | escalate  | low_confidence |
| EV-078 | hard_used_item | deny | partial_refund 101.15 | - |
