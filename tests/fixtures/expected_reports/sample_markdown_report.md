# Test Report: customer-support-agent

**Configuration:** `agents/customer-support.yaml`
**Generated:** 2025-11-22T10:33:00Z
**HoloDeck Version:** 0.1.0
**Environment:** Python 3.10.0 on Darwin

## Summary

| Metric         | Value           |
| -------------- | --------------- |
| Total Tests    | 3               |
| Passed         | 2               |
| Failed         | 1               |
| Pass Rate      | 66.67%          |
| Total Duration | 14500ms (14.5s) |

### Average Metric Scores

| Metric       | Average Score | Scale |
| ------------ | ------------- | ----- |
| groundedness | 0.78          | 0-1   |
| relevance    | 0.80          | 0-1   |
| coherence    | 0.95          | 0-1   |
| completeness | 0.62          | 0-1   |

---

## Test Results

### Test 1: test_query_resolution_pdf ✅ PASSED

**Input:**

```
User asks about warranty coverage for product XYZ
```

**Execution Time:** 4500ms

#### Processed Files

| File              | Type | Format                  | Processing Time | Status     |
| ----------------- | ---- | ----------------------- | --------------- | ---------- |
| docs/warranty.pdf | pdf  | PDF (5 pages, 245.0 KB) | 2150ms          | ✅ Success |

**File Metadata:**

- **Cached Location:** `/tmp/cache/warranty_abc123.md`
- **Pages:** 5
- **Size:** 245.0 KB

#### Agent Response

> Based on the warranty policy document, product XYZ includes a 12-month manufacturer warranty covering defects in materials and workmanship. The warranty does not cover damage from misuse or normal wear.

#### Tool Usage

**Tools Called:** `search_knowledge_base`, `retrieve_policy`
**Expected Tools:** `search_knowledge_base`, `retrieve_policy`
**Match Status:** ✅ Tools matched expected

#### Evaluation Metrics

| Metric       | Score    | Threshold | Status  | Model       | Eval Time | Retries |
| ------------ | -------- | --------- | ------- | ----------- | --------- | ------- |
| groundedness | 0.92/1.0 | 0.8       | ✅ PASS | gpt-4o-mini | 1250ms    | 0       |
| relevance    | 0.88/1.0 | 0.75      | ✅ PASS | gpt-4o-mini | 980ms     | 0       |
| coherence    | 0.95/1.0 | 0.7       | ✅ PASS | gpt-4o-mini | 850ms     | 1       |

#### Ground Truth Comparison

**Expected:**

```
The product comes with a 12-month manufacturer warranty covering manufacturing defects.
```

---

### Test 2: test_refund_policy_no_match ❌ FAILED

**Input:**

```
Can I get a refund if I'm not satisfied?
```

**Execution Time:** 3200ms

#### Agent Response

> I apologize, but I don't have information about our refund policy in my current knowledge base.

#### Tool Usage

**Tools Called:** `search_knowledge_base`
**Expected Tools:** `search_knowledge_base`, `escalate_to_human`
**Match Status:** ❌ Tools did not match expected (missing: `escalate_to_human`)

#### Evaluation Metrics

| Metric       | Score    | Threshold | Status  | Model       | Eval Time | Retries | Error                                                           |
| ------------ | -------- | --------- | ------- | ----------- | --------- | ------- | --------------------------------------------------------------- |
| groundedness | 0.45/1.0 | 0.8       | ❌ FAIL | gpt-4o-mini | 1100ms    | 0       | —                                                               |
| relevance    | 0.62/1.0 | 0.75      | ❌ FAIL | gpt-4o-mini | 920ms     | 0       | —                                                               |
| completeness | 0.35/1.0 | 0.7       | ❌ FAIL | gpt-4o-mini | 2150ms    | 2       | Incomplete response did not address all aspects of the question |

#### Ground Truth Comparison

**Expected:**

```
Our refund policy allows full refunds within 30 days of purchase for any reason.
```

#### Errors

- ❌ Agent failed to retrieve required policy document
- ❌ Tool mismatch: expected escalate_to_human but not called

---

### Test 3: test_multi_document_analysis ✅ PASSED

**Input:**

```
Compare shipping costs and delivery times across different services
```

**Execution Time:** 6800ms

#### Processed Files

| File                     | Type  | Format                   | Processing Time | Status     |
| ------------------------ | ----- | ------------------------ | --------------- | ---------- |
| docs/shipping_rates.xlsx | excel | Excel (xlsx, 18.5 KB)    | 1200ms          | ✅ Success |
| docs/carrier_info.pdf    | pdf   | PDF (15 pages, 512.0 KB) | 3500ms          | ✅ Success |

**File Metadata:**

- **shipping_rates.xlsx**
  - Sheet: `standard_shipping`
  - Size: 18.5 KB
  - Cached: `/tmp/cache/shipping_rates_xyz789.md`
- **carrier_info.pdf**
  - Pages: 15
  - Size: 512.0 KB
  - Cached: `/tmp/cache/carrier_info_def456.md`

#### Agent Response

> Based on the shipping documentation, we offer Standard shipping at $5.99 (5-7 business days) and Express shipping at $12.99 (2-3 business days). Our carrier partners include FedEx, UPS, and DHL, serving over 220 countries globally.

#### Tool Usage

**Tools Called:** `search_knowledge_base`, `retrieve_document`, `retrieve_document`
**Expected Tools:** `search_knowledge_base`, `retrieve_document`
**Match Status:** ✅ Tools matched expected

#### Evaluation Metrics

| Metric       | Score    | Threshold | Status  | Model  | Eval Time | Retries |
| ------------ | -------- | --------- | ------- | ------ | --------- | ------- |
| groundedness | 0.96/1.0 | 0.8       | ✅ PASS | gpt-4o | 1300ms    | 0       |
| relevance    | 0.91/1.0 | 0.75      | ✅ PASS | gpt-4o | 1050ms    | 0       |
| completeness | 0.89/1.0 | 0.7       | ✅ PASS | gpt-4o | 950ms     | 0       |

#### Ground Truth Comparison

**Expected:**

```
Shipping options include Standard ($5.99, 5-7 days) and Express ($12.99, 2-3 days) through multiple carriers.
```

---

## Report Summary

✅ **2 tests passed** | ❌ **1 test failed** | **Pass Rate: 66.67%**

The customer-support-agent performed well on warranty and shipping inquiries with high metric scores (0.78-0.95), but failed to handle refund policy questions appropriately. Recommended action: add refund policy documentation to knowledge base and implement escalation logic for out-of-scope queries.
