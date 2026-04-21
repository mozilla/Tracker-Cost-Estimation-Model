# Paper Improvement Changelog

## Supervisor Feedback — 2026-04-21

Feedback received from supervisor covering six items (intro framing, path+param features, referrer feature, cross-browser feature discipline, cascading blocking, model size comparison). Triaged into must-do / should-do / nice-to-have. This cycle addresses the two must-do items; the rest are deferred with rationale below.

**Changes made:**

1. **[CONTENT] §1 Introduction: added explicit ML-vs-LUT research question** — The intro flowed from the LUT critique (lines 82–94) directly into "We formulate per-request cost prediction as supervised regression" without ever stating the investigative question. Supervisor noted the framing should emphasize exploring predictive models to address LUT limitations. Inserted two sentences between the LUT critique and the formulation block: "This raises the central question of our work: can a learned model, trained on completed tracker requests, produce more accurate and more compact per-request cost estimates than any lookup table at matched feature availability? We investigate this by comparing XGBoost with Tweedie loss against a hierarchy of lookup tables of increasing granularity, using only features Firefox can observe before a response arrives." The second sentence also quietly addresses the cross-browser feature-discipline point (see deferred item below) by naming the pre-response constraint.

2. **[CONTENT] §9 Limitations: added cascading blocking caveat** — The paper's per-request cost estimates assume each blocked request is independent, but blocking a loader script (tag manager, consent provider) can prevent downstream third-party requests from being issued at all, so true bandwidth savings are strictly larger than our aggregated totals. Supervisor flagged this as a multiplier effect the paper should at least acknowledge. Added a new `\paragraph{Cascading blocking effects.}` between Server-side variation and Ecological validity, framing our aggregated weekly totals as a *lower bound* on true savings and citing Brave's paired-crawl approach as the method that would quantify the multiplier. Does not alter any numerical claims; the abstract and conclusion were already careful to say "cost of blocked requests" (the direct cost), which is consistent with the new caveat.

**PDF archived:** (pending rebuild)

---

**Deferred to a later cycle (with rationale):**

- **Explicit "browser-agnostic features" statement in §3.2 or §4.2.** Supervisor's point is that features must be shared between Chrome and Firefox since training uses Chrome crawls but deployment is in Firefox. The paper already handles this implicitly: §3.2 argues $P(Y|X)$ is server-determined and thus browser-invariant, and the feature list in §4.2 uses only URL + content-policy-type + method + initiator (all shared). The new intro sentence now also frames features as "what Firefox can observe before a response arrives." Adding a dedicated sentence listing *excluded* browser-specific features (User-Agent variants, etc.) is nice-to-have but not must-do. Low priority.

- **Model size comparison (XGBoost ONNX vs path LUT footprint) in appendix or §7.** §5.2 already gives ONNX sizes at 200/300/500 trees, and §3.3 notes the path LUT would extrapolate to ~23M entries at full scale. A clean side-by-side (500KB model vs tens-of-MB full-scale LUT) would strengthen the deployment argument, especially for mobile. Medium priority; defer until other must-do items are settled.

- **Explicit mention that query parameters are tokenized alongside paths (§4.2).** The delimiter list in the TF-IDF tokenization (`/ ? & = . - _`) already includes `? & =`, so query params are captured, but a careful reader has to infer this from the delimiters. Supervisor explicitly called path+tokenized parameters the right approach. One clarifying sentence would make it explicit. Low priority, pure presentation.

- **Referrer as additional feature.** Supervisor explicitly called this "not a high priority" and "marginal improvement might be small" since URL path may already encode the same signal. Defer indefinitely as future-work mention only, not as a feature to add pre-submission. No experimental work warranted at this time.

---

## Cycle 28 — 2026-04-07T16:52Z

**Changes made:**

1. **[CONTENT] §7 line 805: "not significantly correlated" → "only weakly correlated"** — With n ≈ 523,624 test rows, the standard error of r is ~1/√n ≈ 0.0014. An r of 0.05 is therefore ~35 standard errors from zero — overwhelmingly statistically significant (p ≪ 0.001). The original phrasing "not significantly correlated" was therefore technically incorrect: any reasonable significance test would detect this correlation. The intended claim was that the effect size is negligible (r² ≈ 0.0025, explaining only 0.25% of variance in prediction error). This is the same precision error corrected in §3.2 in Cycle 25, where "confirming no systematic feature shift" was changed to "confirming no substantial systematic feature shift." Fixed to "only weakly correlated."

2. **[PRESENTATION] §5.8 line 723: clarified 1,117 is a test-set count, not training-example density** — The calibration table (Table 9) partitions test-set rows by predicted value; the 1,117 in the 500 B–1 KB bin represents test-set rows with predictions in that range, not training examples. The original text said "the predicted range with the lowest training-example density (1,117 rows; the sparsest bin in Table~\ref{tab:calibration})" — conflating a test-set count with training density. Fixed to: "the predicted range with the sparsest test-set coverage (1,117 rows in Table~\ref{tab:calibration}; similarly sparse in training given the random split)."

3. **[PRESENTATION] §7 line 809: "1,117 training examples" → "1,117 test-set rows"** — Same factual error as above, but more explicit: the original said "the 500 B–1 KB predicted bin has only 1,117 training examples," which is wrong — Table 9 counts test rows. The spirit of the argument (this prediction range has sparse gradient signal during training) is correct, since the random 80/20 split means training density is proportionally similar. Fixed to: "the 500 B–1 KB predicted bin has only 1,117 test-set rows, and training density is comparably sparse."

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_28.pdf`

---

**Deferred to Cycle 29:**
- §6.2 figure relocation (pred-vs-actual to §5.1; LaTeX reorganization risk)
- Domain-held-out split (requires retraining)

---

## Cycle 27 — 2026-04-07T16:34Z

**Changes made:**

1. **[PRESENTATION] Table 1 (tab:lut-hierarchy): flipped sign convention and renamed column header** — The "vs. global" column showed −42.3%, −51.7%, −72.2% (negative because MAE decreased relative to global baseline). Every other comparison column in the paper uses positive percentages for improvements (e.g., +47.5% in the loss ablation, +75.9% in the by-type table), and uses negative only when an approach is *worse* than the baseline (e.g., −3.8% for text type, −12.7% for TTFB). A reader familiar with those conventions would momentarily read −42.3% as worsening, not improvement. Fixed by: (1) removing the $-$ prefix from all three values, (2) renaming the column header from "vs. global" to "Improvement", (3) updating the caption to say "percent MAE reduction relative to the global median; higher is better." The numerical values themselves are unchanged.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_27.pdf`

---

**Deferred to Cycle 28:**
- §6.2 figure relocation (pred-vs-actual to §5.1; LaTeX reorganization risk)
- Domain-held-out split (requires retraining)

---

## Cycle 26 — 2026-04-07T16:12Z

**Changes made:**

1. **[CONTENT] §5.2: "at least 17%" → "at least 20%"** — The loss ablation section claimed all Tweedie variants in p ∈ [1.2, 1.8] outperform squared error "by at least 17%." The actual computed improvements are: p=1.2 → 23.0%, p=1.5 → 23.4%, p=1.8 → 20.5%. The actual floor is ~20.5%, so the correct lower bound is "at least 20%." Saying "at least 17%" was technically true but understated the point by ~3.5 percentage points, oddly underselling a result that supports a key claim about Tweedie robustness.

2. **[PRESENTATION] Contribution #3: Added "bytes" unit** — The contributions list at line 107 gave "MAE 1,346 vs. 1,448" without units, while the abstract at line 58 gives "MAE 1,346 vs. 1,448 bytes." In ACM sigconf layout the abstract and body are both on the first page; a reader scanning the contributions sees the numbers without units. Added "bytes" for consistency with the abstract.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_26.pdf`

---

**Deferred to Cycle 27:**
- §6.2 figure relocation (pred-vs-actual to §5.1; LaTeX reorganization risk)
- Domain-held-out split (requires retraining)

---

## Cycle 25 — 2026-04-07T15:57Z

**Changes made:**

1. **[CONTENT] §3.2: "confirming no systematic feature shift" → "confirming no substantial systematic feature shift"** — The classifier achieves AUC = 0.557 vs. chance = 0.5. With ~3M samples, any AUC above 0.5 is statistically distinguishable from chance — so the classifier *can* detect some feature shift between splits, meaning "no systematic feature shift" is technically an overstatement. The shift is small and practically negligible (0.557 is close to chance), so adding "substantial" makes the claim accurate: any shift that exists is too small to matter, but its existence is not denied.

2. **[PRESENTATION] §6.1: Restructure KS test parenthetical** — "are uniformly inexpensive (KS test p < 0.001 on all features between the two populations)" placed the KS test as evidence for "uniformly inexpensive," which doesn't need statistical support (it follows from the Lighthouse threshold criterion). The KS test actually establishes that the two populations are feature-distinct, which supports the next sentence's "The disjointness is structural" claim. Restructured to: "are uniformly inexpensive; a KS test confirms the two populations are feature-distinct across all dimensions (p < 0.001)" — putting the evidence beside the claim it supports.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_25.pdf`

---

**Deferred to Cycle 26:**
- §6.2 figure relocation (pred-vs-actual to §5.1; LaTeX reorganization risk)
- Domain-held-out split (requires retraining)

---

## Cycle 24 — 2026-04-07T15:34Z

**Changes made:**

1. **[PRESENTATION] Abstract: "In aggregate" → "Overall"** — The paper uses "aggregate/aggregation" as a technical term with a specific meaning (weekly per-user cost summation, §5.5). "In aggregate" in the abstract means "overall/across all test rows" — contrasting with the seen-paths-only MAE in the preceding sentence — but a reader who has skimmed the paper's section titles will momentarily read it as referring to the weekly aggregation comparison, which uses the D+T LUT as baseline (not the path LUT cited in this sentence). "Overall" eliminates the ambiguity with no loss of meaning.

2. **[PRESENTATION] Introduction: "approximately 250 million daily active users" → "hundreds of millions of daily active users"** — The specific figure was stated without a citation. Firefox DAU figures vary over time across Mozilla's public communications; asserting a precise number without a source invites a reviewer challenge. The rhetorical purpose is to establish scale, for which "hundreds of millions" is both accurate and defensible without citation.

3. **[PRESENTATION] §7: soften unsourced power-law claim** — "real browsing follows a power-law distribution" is a factual distributional claim without citation. The point of the sentence (uniform-over-domains sampling may over-diversify relative to actual browsing) is valid and worth retaining, but can be expressed without asserting a specific distribution family. Changed to "real browsing tends to be more concentrated, with visits clustering on a small set of frequently-visited sites" — consistent with §5.5's existing language ("users tend to visit a small set of sites repeatedly") and supported by the domain-count sensitivity analysis already in the paper.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_24.pdf`

---

**Deferred to Cycle 25:**
- §6.2 figure relocation (pred-vs-actual to §5.1; LaTeX reorganization risk)
- Domain-held-out split (requires retraining)

---

## Cycle 23 — 2026-04-07T15:14Z

**Changes made:**

1. **[PRESENTATION] Fixed conclusion normative phrase "with sufficient accuracy for user-facing display"** — The conclusion's opening sentence retained the phrase "with sufficient accuracy for user-facing display" which was correctly removed from the abstract in Cycle 20. The same critique applies: this is a normative/requirements judgment the paper does not establish (it would require a user study or explicit design criteria). The parenthetical already contained the empirical metrics (6.0% median weekly aggregation error; 68.5% within 10%); dropping the normative overlay and integrating the metrics directly makes the claim precisely what the results support and removes the inconsistency between the abstract and conclusion.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_23.pdf`

---

**Deferred to Cycle 24:**
- §7 power-law browsing citation (requires bibliography check; claim is hedged with "may")
- §6.2 figure relocation (pred-vs-actual to §5.1; LaTeX reorganization risk)
- Domain-held-out split (requires retraining)

---

## Cycle 22 — 2026-04-07T14:56Z

**Changes made:**

1. **[PRESENTATION] Fixed tab:by-type "Improv." → "vs. D+T LUT"** — Cycle 21 fixed the identical "Improv." column header in tab:multi-target but missed the same header in tab:by-type. The two tables now had inconsistent headers for the same statistic. Changed to "vs. D+T LUT" to match tab:multi-target.

2. **[PRESENTATION] Fixed §5.4 SHAP opening sentence** — The sentence "SHAP values (mean |φ|, ... ; ...) give a substantially different picture" had two nested parenthetical clauses (computation method + in-sample bias caveat) totaling ~60 words before the main predicate "give." Split into two sentences: one stating the computation method and caveat, one making the finding. The main predicate "The result gives a substantially different picture" is now immediately readable.

3. **[PRESENTATION] Fixed data split notation inconsistency** — Section 4.4 said "15% validation (523,624 rows), 15% test (\numtestrows)" — both are the same value (523,624) but expressed differently. Changed to "15% each for validation and test (\numtestrows rows each)" for consistent notation and to remove the superficial asymmetry.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_22.pdf`

---

**Deferred to Cycle 23:**
- §6.2 figure relocation (pred-vs-actual to §5.1; LaTeX reorganization risk)
- Domain-held-out split (requires retraining)

---

## Cycle 21 — 2026-04-07T14:34Z

**Changes made:**

1. **[CONTENT] Removed unsourced "10--20 sites" quantitative claim from §5.5** — "users visit the same 10--20 sites repeatedly" was a specific quantitative claim with no citation. A reviewer would flag this as needing a source. The intended point (browsing is concentrated, not uniform) needs no specific number; the domain-count sensitivity analysis (n_domains ∈ {5, 10, 15, 20, 25}) already provides empirical coverage across this range. Changed to "users tend to visit a small set of sites repeatedly" — defensible without a citation.

2. **[CONTENT] Strengthened §6.2 with evidence citation from Table~\ref{tab:by-type}** — The section "Tweedie's Tail Prioritization Concentrates Gains in Aggregation" claimed Tweedie's per-request gains are concentrated where they matter most for aggregation, but cited no evidence for this. Added one sentence connecting to Table 4 (resource-type breakdown): "Consistent with this, the largest per-request MAE improvements by resource type are scripts (+75.9%) and CSS (+66.5%)---precisely the high-transfer types that dominate weekly bandwidth totals." This makes the analysis section evidence-based rather than purely assertive, and connects two pieces of the paper that were previously unlinked.

3. **[PRESENTATION] Fixed Table 6 column header "Improv." → "vs. D+T LUT"** — The column header "Improv." required inferring the denominator (D+T LUT) from context. Changed to "vs. D+T LUT" for clarity, consistent with how other tables in the paper phrase relative comparisons.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_21.pdf`

---

**Deferred to Cycle 22:**
- §5.4 SHAP parenthetical (long mid-sentence methodology note; low priority)
- §6.2 figure relocation (pred-vs-actual to §5.1 or §5.3; LaTeX reorganization risk)
- Domain-held-out split (requires retraining)

---

## Cycle 20 — 2026-04-07T14:13Z

**Changes made:**

1. **[CONTENT] Replaced abstract's normative "sufficient for user-facing display" with within-10% metric** — The abstract claimed the 6.0% aggregation error is "sufficient for user-facing display in Firefox" — a threshold claim with no defined threshold. A reviewer at IMC/WWW outside the Firefox context has no way to evaluate this assertion. Replaced with the factual within-10% metric ("with 68.5% of weeks falling within 10% of true cost"), which gives readers the data to judge accuracy themselves. The conclusion (§9) already pairs the sufficiency claim with these numbers in context; the abstract should provide the data, not the editorial judgment.

2. **[PRESENTATION] Separated IID-split check from deployment validity argument in §3.2** — The paragraph began "We confirm this empirically: a gradient-boosted classifier...AUC = 0.557, confirming that the train/test split produces similar feature distributions." The phrase "We confirm this empirically" implied the AUC check confirmed deployment validity (Chrome HTTP Archive → Firefox), but it only confirms within-dataset IID split quality. The subsequent sentence ("Deployment validity rests primarily on the server-determinism argument") is the actual deployment validity claim. Changed "We confirm this empirically" to "We verify the random split is IID" (scoping the AUC check correctly), and changed "rests primarily on the server-determinism argument above" to "is a separate argument and rests on server-determinism" (making the logical independence explicit).

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_20.pdf`

---

**Deferred to Cycle 21:**
- §6.2 restructuring (figure relocation + sentence merge into §5.3; low payoff vs. LaTeX reorganization risk)
- Power-law browsing citation in §7 (bibliography check needed)
- Domain-held-out split (requires retraining)

---

## Cycle 19 — 2026-04-07T13:56Z

**Changes made:**

1. **[CONTENT] Fixed factual error in §5.6 calibration note** — Cycle 18 added a sentence claiming the 25–50 KB bin has "the lowest within-25% accuracy outside the beacon bin (37.3%)." This is factually wrong: the 250 KB+ bin has 25.8%, which is lower. The 25–50 KB bin's notable property is not its rank but the combination of mean-calibration (predicted 34 KB ≈ actual 36 KB) with anomalously low interval accuracy. Replaced "the lowest within-25% accuracy outside the beacon bin" with "notably low within-25% accuracy for a mean-calibrated bin" — accurate and correctly frames the paradox.

2. **[PRESENTATION] Retitled Section 6.2** — After Cycle 18 removed the unsupported "even larger aggregation advantage" claim, §6.2 was left with two sentences about Tweedie's tail prioritization concentrating gains in aggregation. The old title "Why Tweedie Loss Outperforms Squared Error" no longer matched: the per-request mechanistic explanation (proportional error weighting, beacon de-emphasis) is in §5.3, not §6.2. Retitled to "Tweedie's Tail Prioritization Concentrates Gains in Aggregation" to accurately describe what the section actually argues.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_19.pdf`

---

**Deferred to Cycle 20:**
- Power-law browsing citation in §7 (low priority; needs bibliography check)
- Domain-held-out split (requires retraining)

---

## Cycle 18 — 2026-04-07T13:37Z

**Changes made:**

1. **[CONTENT] Removed unsupported "even larger aggregation advantage" claim from Section 6.2** — The sentence "The 23% per-request MAE gap (3,466 vs. 4,527) is expected to produce an even larger aggregation advantage" asserted a comparative claim (aggregation advantage > per-request gap) without demonstrating it. The squared-error model's aggregation error was never computed, so the direction is unknowable. Replaced with the mechanistic argument alone: "Beacons (39.5% of requests, near-zero transfer size) contribute negligibly to the weekly bandwidth total, while the high-cost scripts that Tweedie loss prioritizes dominate it. This means Tweedie's per-request gains are concentrated where they matter most for aggregation."

2. **[PRESENTATION] Restructured correlated browsing parenthetical (lines 594–595)** — A single sentence containing a long mid-clause parenthetical ("note that the uniform column in Table X uses an independent simulation...") forced readers to track two ideas simultaneously. Split into two sentences: the primary claim (correlated browsing doubles error, confirming error correlation reduces cancellation) and the methodology note (independent 2,000-trial simulation; difference reflects simulation variance).

3. **[PRESENTATION] Fixed Table 5 caption "deployable alternatives"** — The phrase "within-10% rates are reported for the deployable alternatives only" was ambiguous; the path LUT is technically deployable (just impractical at scale). Changed to "model and D+T LUT only" to be explicit.

4. **[CONTENT] Added 25–50 KB calibration paradox note** — Table 7 shows the 25–50 KB bin is mean-calibrated (predicted 34 KB vs. actual 36 KB, 4.3% error) but has anomalously low within-25% accuracy (37.3%) — the lowest of any bin with accurate mean prediction. This apparent paradox was undiscussed. Added a sentence explaining it: high within-bin variance rather than systematic bias, distinguishing it from the underprediction pattern in the 500B–1KB and 1–5KB bins.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_18.pdf`

---

**Deferred to Cycle 19:**
- Abstract restructuring (path LUT mentioned twice; minor but cleanable)
- Self-training quantification
- Domain-held-out split (requires retraining)

---

## Cycle 17 — 2026-04-07T13:09Z

**Changes made:**

1. **[CONTENT] Fixed internal inconsistency in Limitations calibration paragraph** — Section 7 "Calibration bias" paragraph still said "the region where Tweedie gradient signal is weakest relative to training density" — the old incorrect phrasing that was specifically corrected in Section 5.6 in Cycle 15. The Tweedie gradient is not weak when y >> ŷ (the underprediction case); the correct explanation is training-data sparsity (1,117 examples). Updated Limitations to match Section 5.6: "the region with the lowest training-example density (the 500B–1KB predicted bin has only 1,117 training examples)."

2. **[CONTENT] Fixed CV 0.94 redundancy in Section 5.4** — Cycle 16 added "(CV 0.94 for transfer size and comparably high for download duration)" to line 623, but line 625 already said "transfer size has CV 0.94." The same number appeared in consecutive sentences. Removed the CV citation from line 623 (restoring the sentence to say "URL determines response size, which is the primary driver of download time"). Expanded line 625 to include download_ms: "transfer size has CV 0.94 and download duration has comparably high within-domain variance (Figure 3), both URL-predictable; TTFB has CV 0.38."

3. **[PRESENTATION] Standardized Contribution 3 language** — Intro contribution #3 said "(MAE 1,346 vs. 1,448; bootstrap CIs non-overlapping)" as a bare parenthetical. Changed to "a statistically significant improvement (non-overlapping bootstrap CIs)" to match the abstract's register and make the finding clearer on first read.

4. **[PRESENTATION] Fixed "blocked action" in conclusion** — Final sentence said "labeled examples of the blocked action can be collected from a proxy population." "Blocked action" is ambiguous (intercepted-by-ETP vs. failed request). Changed to "the intercepted resource type."

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_17.pdf`

---

**Deferred to Cycle 18:**
- Abstract restructuring (path LUT mentioned twice; minor but cleanable)
- Self-training quantification
- Domain-held-out split (requires retraining)

---

## Cycle 16 — 2026-04-07T12:58Z

**Pre-cycle incorporation (Cycle 15 PID 59708):** Within-10% aggregation simulation completed. Table 5 restructured into two blocks: (1) median % error (all three approaches), (2) fraction within 10% (model and D+T LUT only, path LUT impractical). Body text updated: 67.2% → 68.5% at N=200; 13.6% → 16.8%; path LUT within-10% comparison removed (not computed).

**Changes made:**

1. **[PRESENTATION] Abstract: replace "(bootstrap CIs non-overlapping)" with plain language** — The parenthetical was correct but too technical for an abstract audience. Changed to: "a statistically significant improvement" — the full CI detail is in Table 3. Abstract readers can verify in the body; the abstract's job is to convey the finding, not the methodology.

2. **[CONTENT] Download_ms: symmetric evidence + removed "hence"** — Line 623 justified transfer_bytes improvement by citing CV 0.94 but gave no CV for download_ms. Updated to cite Figure 3 (which shows both have high within-domain CV) and softened the causal "hence" to "which is the primary driver of download time for large resources" — preserving the content-dependent claim without overstating determinism.

3. **[PRESENTATION] Data split: eliminated \numtestrows for validation** — Line 288 used the same macro for validation and test set sizes (both happen to be 523,624). Using \numtestrows for validation is confusing because the macro is used throughout the paper to refer to the test set. Fixed to write the number explicitly for validation: "15% validation (523,624 rows), 15% test (\numtestrows)."

4. **[CONTENT] Conclusion opening: added headline numbers** — "sufficient accuracy for user-facing display" was quantitatively empty at the conclusion's opening sentence. Readers who have absorbed all the results know the numbers; they deserve to see them restated at the conclusion. Added: "(6.0% median weekly aggregation error over the deployed baseline; 68.5% of weeks within 10% of true cost)."

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_16.pdf`

---

**Deferred to Cycle 17:**
- Self-training null-result quantification
- Domain-held-out split (requires retraining)

---

## Cycle 15 — 2026-04-07T12:39Z

**Pre-cycle incorporation (Cycle 14 PID 48930):** Bootstrap CIs for path decomposition completed. Key finding: seen-path improvement statistically significant (Model CI [1,317–1,384] vs. Path LUT CI [1,409–1,491], non-overlapping); unseen-path improvement NOT significant (CIs overlap). Updated:
- Table 3 (tab:decomposition) restructured to include CI columns with dagger on unseen row
- Abstract: removed "generalizes better to unseen paths" claim; kept statistically valid seen-path claim
- Body text line 369: added CI values and "not statistically significant" qualification
- Figure caption (fig:path-decomp): softened "outperforms on unseen paths"

**Changes made:**

1. **[CONTENT] Contribution 3 in intro: added both error figures** — The contribution claimed "producing estimates with 6.0% median error" after "including under correlated browsing patterns," implying 6.0% applies to both regimes. 6.0% is the uniform-sampling result; under correlated browsing the figure is 12.9%. Split into explicit values: "6.0% under uniform sampling and 12.9% under domain-correlated browsing at N=200."

2. **[CONTENT] Conclusion finding 3: added 12.9% figure and qualified "accurate"** — The conclusion said "enabling accurate cost estimation" without specifying the error rate under correlated browsing. Replaced with the concrete comparison: model 6.0% vs D+T LUT 21.9% (uniform) and 12.9% vs 36.4% (correlated), framing accuracy relative to the deployed baseline rather than in absolute terms.

3. **[CONTENT] Calibration mechanism: corrected "Tweedie gradient weakest" to density explanation** — The paper claimed the 500B–1KB miscalibration occurs where "the Tweedie gradient is weakest." This is mechanistically incorrect: the Tweedie gradient is large when y is large and ŷ is small (the underprediction case). The correct explanation is example density: 1,117 training rows (the sparsest calibration bin) have their gradient signal diluted by the beacon (26,462 rows) and large-bundle clusters. Updated to: "the predicted range with the lowest training-example density."

**Background jobs:** Within-10% aggregation rates (PID 59708, log: `logs/within_10pct_agg_cycle15.log`) — still running at archive time. If results arrive: add "Within 10%" column to Table 5 (tab:aggregation) in Cycle 16.

**PDF archived:** `versions/paper_CYCLE_15.pdf`

---

**Deferred to Cycle 16:**
- Within-10% column for Table 5 — incorporate when PID 59708 completes
- Self-training quantification (requires recovering prior experiment logs)
- Domain-held-out split (requires retraining)

---

## Cycle 14 — 2026-04-07T12:10Z

**Changes made:**

1. **[PRESENTATION] Abstract: added practical deployment improvement metric** — The abstract summarized model performance as "6.0% median error in weekly aggregation" without stating what baseline this compares to. Changed the final sentence to: "reduces weekly aggregation error from 21.9% to 6.0% over the deployed domain+type lookup table." A reader now immediately sees the practical value without reaching Section 5.3.

2. **[CONTENT] Timing metrics aggregation paradox explained** — Section 5.4 stated that the model achieves better aggregation error than the D+T LUT even for TTFB and load_ms where per-request MAE was worse, attributing this to "systematic bias" without explanation. Added 2 sentences: the D+T LUT's training medians reflect HTTP Archive's fixed crawl conditions (Moto G4, cable); deployment conditions vary, introducing a directional shift that compounds in aggregation. The model, despite worse per-request MAE, is less systematically biased because URL features capture request-type variation that correlates with timing across conditions.

3. **[CONTENT] Limitations correlated browsing paragraph updated to reflect Cycle 13 sensitivity** — The existing text said "Real browsing patterns may differ from our 15-domain simulation in ways we cannot fully characterize without per-user telemetry." This was accurate before the sensitivity analysis. Updated to acknowledge what is now resolved (domain count robustness, 5–25 domains) and precisely name what remains unquantified: the shape of the domain-visit distribution (uniform random vs. power-law) and whether the test-set empirical request frequencies match real browsing concentrations.

**Background jobs:** Bootstrap CIs for path decomposition subsets (PID 48930, log: `logs/path_decomp_ci_cycle14.log`) — still running at archive time. If results arrive: add CI columns to Table 3 (tab:decomposition) in Cycle 15.

**PDF archived:** `versions/paper_CYCLE_14.pdf`

---

**Deferred to Cycle 15:**
- Bootstrap CIs for path decomposition subsets — incorporate when PID 48930 completes
- Domain-held-out split (requires retraining)

---

## Cycle 13 — 2026-04-07T12:00Z

**Changes made:**

1. **[CONTENT] Resource-type table: accounted for 2,069 missing test rows** — Table caption for tab:by-type gave no indication it was incomplete. The six listed categories cover 521,555 of 523,624 test rows; the remaining 2,069 rows (<0.4%) span resource types with <1,000 test examples each. Added a note to the caption to prevent a reviewer from flagging the apparent row-count discrepancy.

2. **[CONTENT] Hyperparameter selection methodology added** — Section 4.2 listed exact XGBoost hyperparameters (max depth 8, lr 0.05, etc.) but never stated how they were chosen. Added one sentence: "Hyperparameters were set based on standard XGBoost defaults and prior experience on tabular regression tasks; no automated search was performed." This clarifies that the validation set is not contaminated by hyperparameter search, keeping the validation-set stopping criterion unbiased.

3. **[EXPERIMENT] Domain-count sensitivity for correlated browsing** — Ran `src/domain_sensitivity.py` (n_domains ∈ {5, 10, 15, 20, 25}, N=200, 2,000 trials). Results:

   | n_domains | Model % | D+T LUT % | Gap pp |
   |-----------|---------|-----------|--------|
   | 5         | 16.5    | 34.2      | 17.7   |
   | 10        | 12.5    | 35.6      | 23.1   |
   | 15        | 11.5    | 37.6      | 26.1   |
   | 20        | 11.4    | 36.2      | 24.8   |
   | 25        | 11.9    | 34.7      | 22.7   |

   The model's advantage holds across the entire range (gap 17.7–26.1 pp). Added one sentence to the correlated browsing paragraph (Section 5.5): "The model's advantage is robust to domain count: at N=200, model error ranges from 11.4%–16.5% across n_domains ∈ {5,10,15,20,25} vs. D+T LUT error from 34.2%–37.6%."

**Background jobs:** `src/domain_sensitivity.py` completed (PID 45915); results in `logs/domain_sensitivity_cycle13.json`
**PDF archived:** `versions/paper_CYCLE_13.pdf`

---

**Deferred to Cycle 14:**
- Bootstrap CIs for path decomposition subset (1,346 vs 1,448 MAE; requires code change)
- Domain-held-out split (requires retraining)

---

## Cycle 12 — 2026-04-07T11:30Z

**Changes made:**

1. **[PRESENTATION] "Three findings" → "Four findings" in conclusion** — Cycle 10 inserted a fourth generalized finding (SHAP vs. gain inversion) without updating the lead sentence. The conclusion said "Three findings generalize" but enumerated First, Second, Third, and Fourth. Fixed the count.

2. **[CONTENT] Removed stale AUC=0.557 cross-reference from Limitations** — Cycle 11 clarified in Section 3.2 that AUC=0.557 measures within-HTTP-Archive train/test iid similarity (not Chrome→Firefox deployment shift). But the Limitations paragraph still cited "the mild covariate shift (AUC=0.557; Section~\ref{sec:problem})" as evidence for deployment shift, which was now inconsistent with Section 3.2. Removed the parenthetical; the sentence flows cleanly using only the OOD error correlation (r=0.05), which is the directly relevant metric.

3. **[CONTENT] Domain-level section: explained 4,592 vs. 3,723 domain discrepancy** — Section 6.1 mentioned 4,592 tracker domains while the per-request model uses 3,723 (the \numdomains macro). No explanation was given. Added parenthetical: "a broader set drawn from the full HTTP Archive entity list, prior to the per-request 1% sample filtering." This removes a potential reviewer question.

4. **[CONTENT] Intro: reconciled "75% unique paths unseen" vs. "8.4% rows on unseen paths"** — The intro cited 75% of unique paths as unseen, but Section 5.1 shows 8.4% of rows are on unseen paths. Both are correct (high-frequency paths recur), but the apparent contradiction confuses early readers. Added parenthetical: "(though high-frequency paths recur, so only 8.4% of test rows fall on unseen paths)."

5. **[PRESENTATION] Smoothed LUT: disclosed mean vs. median asymmetry** — Standard LUT baselines use training-set median; smoothed LUT interpolates between training mean and global mean. For heavy-tailed data, mean-based smoothing is more sensitive to outliers. Added "(unlike the median-based standard LUT baselines)" to the smoothed LUT paragraph. The conclusion (smoothing hurts) remains valid; the disclosure prevents a reviewer from flagging the comparison as uncontrolled.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_12.pdf`

---

**Deferred to Cycle 13:**
- Domain-held-out split (requires retraining)

---

## Cycle 11 — 2026-04-07T11:10Z

**Changes made:**

1. **[PRESENTATION] 4 more "LUT" → "D+T LUT" in temporal section** — Cycle 9's sweep missed four instances inside the temporal table and its body text paragraph: (A) table row "LUT med. err" → "D+T LUT med. err" (aggregation sub-block); (B) table row "LUT MAE" → "D+T LUT MAE" (scripts-only sub-block); (C) bold body text "32.8% advantage over the LUT" → "D+T LUT"; (D) "below the LUT's 21.5%" → "D+T LUT's". Paper now fully disambiguated throughout.

2. **[CONTENT] AUC covariate shift test: clarified evaluation sample and scope** — The phrase "held-out evaluation sample" was ambiguous; a reviewer familiar with the deployment problem (Firefox, not HTTP Archive) would reasonably ask what this sample is. Clarified to "held-out HTTP Archive test samples" and added a sentence making explicit that deployment validity (Chrome→Firefox) rests on the server-determinism structural argument, not the AUC test. The AUC=0.557 confirms iid train/test split quality; the theoretical argument carries the deployment validity claim.

3. **[CONTENT] "Even larger aggregation advantage" claim hedged** — Section 6.2 asserted "translates to an even larger aggregation advantage" for Tweedie vs. squared error, but the squared error model's aggregation error is never computed or shown in the paper. Changed to "is expected to produce an even larger aggregation advantage" and added a closing clause explaining the mechanism and acknowledging this is not separately demonstrated.

4. **[PRESENTATION] Domain LUT entry count explained in Table 1 caption** — Table 1 showed 2,606 domain LUT entries against 3,723 total domains with no explanation. Added: "The domain LUT has 2,606 entries (not all domains appear in the training split; the remainder fall back to the global median)." Removes a potential reviewer question.

5. **[PRESENTATION] Spearman ρ relevance added to loss ablation figure caption** — Spearman ρ appeared as a metric without explanation of why ranking quality matters for the aggregation use case. Added: "ranking quality matters for per-domain attribution, which requires correctly ordering high-cost vs. low-cost requests."

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_11.pdf`

---

**Deferred to Cycle 12:**
- Domain-held-out split (requires retraining)

---

## Cycle 10 — 2026-04-07T10:50Z

**Changes made:**

1. **[PRESENTATION] Abstract: "best practical lookup table" → "path-level lookup table (the most accurate non-model baseline)"** — The abstract called the path LUT "best practical" while the body (Section 3.2) explicitly calls it "impractical at deployment scale" (227K → ~23M entries). Direct contradiction that a reviewer would flag. Fix removes the contradiction: "path-level lookup table (the most accurate non-model baseline)" is accurate and consistent with the body argument.

2. **[CONTENT] Temporal table caption: added D+T LUT stability finding** — The caption summarized the model's 30% degradation and retained advantage, but didn't explain *why* the advantage is retained. The key insight — that the model and path LUT degrade similarly (~30%) while the D+T LUT is nearly stable (+1.9%) due to coarser granularity — was buried in body text. Added a sentence to the caption making this mechanistic explanation visible at a glance.

3. **[CONTENT] Conclusion: added 4th generalized finding on SHAP vs. gain inversion** — The three existing findings (Tweedie loss dominance, TF-IDF subsuming regex, aggregation cancellation) omitted the paper's most surprising feature importance result. The SHAP vs. gain inversion (domain identity 48.2% by SHAP vs. 9.8% by gain; TF-IDF 32.5% vs. 52.9%) has direct practical implications for practitioners using target encodings. Added as "Fourth" finding with actionable guidance: treat gain-based importance as unreliable when target encodings coexist with high-dimensional continuous features.

4. **[PRESENTATION] Tree-count ablation: "below the significance threshold" → specific CI comparison** — The vague phrase "below the significance threshold" gave no indication of what threshold was being invoked. Replaced with "within its bootstrap CI [3,623, 3,984] and not distinguishable at 95% confidence" — directly ties to the CIs already presented in Table 2.

5. **[PRESENTATION] Path decomposition caption: "the LUT" → "the path LUT"** — Minor consistency fix. The caption's first sentence says "path LUT" and the second said "the LUT." Changed to "the path LUT" for disambiguation consistency with the rest of the paper.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_10.pdf`

---

**Deferred to Cycle 11:**
- Domain-held-out split (requires retraining)

---

## Cycle 9 — 2026-04-07T10:35Z

**Changes made:**

1. **[PRESENTATION] Remaining "LUT" → "D+T LUT" in body text and table headers (9 locations)** — Cycles 7–8 fixed the aggregation table, correlated browsing table headers, and two figure captions. Cycle 9 completes the sweep: (A) resource-type table column header "LUT MAE" → "D+T LUT MAE"; (B) resource-type body text "the LUT's constant prediction" → "the D+T LUT's constant prediction"; (C) correlated browsing body text lines 589–590 (2 instances); (D–E) multi-target table column headers and caption "LUT MAE"/"LUT agg." → "D+T LUT MAE"/"D+T LUT agg."; (F) multi-target body text "the LUT's systematic bias"; (G) temporal table row label "vs. LUT" → "vs. D+T LUT"; (H) temporal table caption; (I) Limitations temporal paragraph. Paper now uses "D+T LUT" or "Path LUT" consistently throughout — no bare "LUT" ambiguity remains.

2. **[PRESENTATION] Fixed calibration section opening sentence** — The sentence "a well-calibrated model has $\mathbb{E}[y \mid \hat{y} \in \text{bin}] \approx \text{bin midpoint}$" was a mean calibration definition that did not match the stated primary metric ("fraction within 25%", an interval calibration criterion). The table actually reports both: Pred. mean vs Actual mean (mean calibration) and Within 25% (interval calibration). Replaced with: "we report both the actual mean per bin against the predicted mean (testing mean-unbiasedness) and the fraction within 25% of the true value (testing interval accuracy)." This accurately describes what the table shows without conflating the two calibration notions.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_9.pdf`

---

**Deferred to Cycle 10:**
- Domain-held-out split (requires retraining)
- 15-domain correlated browsing sensitivity

---

## Cycle 8 — 2026-04-07T10:10Z

**Changes made:**

1. **[CONTENT] Fixed smoothed LUT arithmetic error** — "4.9% worse than the unsmoothed path LUT (3,797)" was wrong: (4,011−3,797)/3,797 = 5.6%, not 4.9%. The 4.9% figure was accurate only relative to the smoothed LUT script's own unsmoothed baseline (3,823), which differs slightly from the canonical test-set path LUT MAE (3,797). Fixed to "5.6% worse than the path LUT (3,797)" for internal consistency.

2. **[PRESENTATION] Correlated browsing table: "LUT" → "D+T LUT"** — Cycle 7 renamed "LUT" to "D+T LUT" in the aggregation table but left the correlated browsing table unchanged. Updated Table 9 headers (both Uniform and Domain-correlated LUT columns) and caption to say "D+T LUT" for consistency.

3. **[PRESENTATION] Figure captions: "LUT" → "D+T LUT"** — Updated two figure captions for consistency: (a) aggregation CDF figure: "the LUT (red)" → "the D+T LUT (red)"; (b) multi-target aggregation figure: "outperforms the LUT" → "outperforms the D+T LUT."

4. **[CONTENT] Section 6.1: added the simple non-ML solution** — The section said domain-level estimation "does not require machine learning" but never stated what solution DOES work. Added one sentence: "A simple non-ML rule suffices: measured domains receive a cost score directly from their Lighthouse metrics; unmeasured domains are classified as inexpensive regardless of other features." This makes the negative finding concrete rather than leaving readers uncertain about what the alternative is.

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_8.pdf`

---

**Deferred to Cycle 9:**
- Domain-held-out split (requires retraining)
- 15-domain correlated browsing sensitivity

---

## Cycle 7 — 2026-04-07T09:59Z

**Changes made:**

1. **[CONTENT] Aggregation table expanded to include path LUT** — The table previously compared model vs domain+type LUT only, never showing path LUT aggregation. Added path LUT column (6.8% at N=200) from `full_experiment_results.json`. Renamed "LUT" → "D+T LUT". Restructured table: dropped "within 10%" columns to make room for 3-way comparison; moved key within-10% figures into body text. Updated caption and body text to explicitly discuss path LUT's competitive aggregation performance (6.8% vs model 6.0%) and why D+T LUT is the practical baseline.

2. **[CONTENT] SHAP in-sample bias disclosed** — Added parenthetical to Section 5.4: target encodings are recomputed from the same 8K-row sample, introducing slight in-sample bias for domain identity features. This was acknowledged in compute_shap.py comments but was absent from the paper.

3. **[CONTENT] OOD error correlation added to Limitations** — Added sentence to "Training-deployment population shift" paragraph citing `error_ood_correlation = 0.05` from `advanced_analysis_results.json`. Confirms that prediction error is not significantly correlated with domain OOD status, validating the mild covariate shift claim (AUC=0.557).

4. **[PRESENTATION] Tree-count ablation sentence rewritten** — Clarified that `iteration_range` is post-hoc pruning (upper bound on what fresh retraining would achieve), not full retraining. Fixed confusing "exceeds the path LUT baseline" phrasing. Split into two clearer sentences.

5. **[PRESENTATION] Section 6.2 trimmed** — Removed first two sentences (redundant with Methods 4.2 Tweedie explanation). Section 6.2 now starts directly with the aggregation implication: "The 23% per-request MAE gap translates to an even larger aggregation advantage."

**Background jobs:** none
**PDF archived:** `versions/paper_CYCLE_7.pdf`

---

**Deferred to Cycle 8:**
- Domain-held-out split (requires retraining)
- 15-domain correlated browsing sensitivity
- N=200 Firefox telemetry justification

---

## Cycle 6 — 2026-04-07T09:36Z

**Changes made:**

1. **[EXPERIMENT] Tree-count ablation completed** — Fixed `src/tree_count_ablation.py` by replacing `url_embeddings_tfidf.npy` (wrong shape) with `url_embedder.joblib`-based embedding computation (same pattern as `compute_shap.py`). Results: 50 trees MAE 13,384 (+286%), 100 trees 8,275 (+139%), 200 trees 3,908 (+12.8%), 300 trees 3,655 (+5.5%), 500 trees 3,466 (baseline). Key finding: 200 trees (200KB ONNX) exceeds path LUT baseline (3,797), losing the headline advantage; 500 trees necessary. Added 1-sentence justification to Methods (line ~316). Results in `logs/tree_count_ablation.json`.

2. **[CONTENT] Calibration table fixed — added Actual mean column** — The table previously showed only predicted mean but the caption incorrectly said "shows the actual mean." The body text referenced actual mean 2,144B for the 500B–1KB bin, which was invisible in the table. Added "Actual mean" column with values from `advanced_analysis_results.json`. Fixed caption to correctly describe both columns. The 500B–1KB underprediction (predicted 697B, actual 2,144B) is now directly verifiable from the table.

3. **[CONTENT] SHAP vs Gain comparison table added** — Inserted Table in Section 5.4 (Feature Importance) showing SHAP% vs Gain% per feature group. The prose already described the inversion but had no table; now directly verifiable. Domain identity: 48.2% SHAP vs 9.8% gain; TF-IDF: 32.5% SHAP vs 52.9% gain.

4. **[PRESENTATION] Feature ablation caption "2.0%" → "2.3%"** — The caption said "adding regex features on top of them (2.0%)" while the body correctly calculated 2.3% relative MAE reduction. Fixed caption to match.

5. **[PRESENTATION] N=200 "typical browsing load" → "moderate browsing load"** — Removed unsupported "typical" claim.

6. **[PRESENTATION] Validation/test duplicate hardcoded count fixed** — Changed "15% validation (523,624)" to use the `\numtestrows` macro consistently for both validation and test splits.

**Background jobs:** Tree-count ablation (launched as background job b1r5rjc6a, log: `logs/tree_count_ablation_cycle6.log`) — completed successfully, results incorporated.
**PDF archived:** `versions/paper_CYCLE_6.pdf`

---

**Deferred to Cycle 7:**
- Domain-held-out split (requires retraining)
- N=200 Firefox telemetry justification (requires external data)

---

## Cycle 5 — 2026-04-07T09:16Z

**Changes made:**

1. **[EXPERIMENT] SHAP feature importance computed and incorporated** — Replaced gain-based importance paragraph in Section 5.4 with actual SHAP values (8K-sample, XGBoost `pred_contribs`). Key finding: domain identity features dominate at 48.2% of mean |SHAP| (vs only 9.8% by gain), while TF-IDF drops from 52.9% gain to 32.5% SHAP — confirming the gain bias the paper had flagged for 4 cycles. `domain_type_median` alone accounts for 44% of total SHAP (mean |φ| = 3.16), nearly 8× the next feature (`rt_other`: 0.40). Results saved to `logs/shap_results.json`.

2. **[CONTENT] Fixed intro CV numbers** — Line 84: changed "coefficient of variation exceeding 3× at the 75th percentile and 29× at the 90th" to "coefficient of variation of 0.94 at the median domain and 3.0 at the 90th percentile," consistent with Section 3.3.

3. **[CONTENT] Fixed conclusion generalization claim** — Replaced "generalizes to any browser intervention where the cost of the blocked action is unobservable" with hedged version: "the general framework---training on completed requests, deploying on blocked ones---applies to any browser intervention where labeled examples of the blocked action can be collected from a proxy population."

4. **[PRESENTATION] Fixed "17%" → "16.5%"** — Contribution #2: (4251−3548)/4251 = 16.54%; changed to match table.

5. **[PRESENTATION] Restructured neural baselines paragraph** — Now leads with deployment constraint (500KB ONNX) as primary justification; Grinsztajn et al. cited as supporting evidence with accurate scope (tree dominance below ~10K samples, tree competitiveness at 2.4M rows).

**PDF archived:** `versions/paper_CYCLE_5.pdf`

---

**Deferred to Cycle 6:**
- Tree-count ablation (fix split alignment: use `url_embedder.joblib` to recompute test embeddings)
- Domain-held-out split (requires retraining)
- N=200 browsing load justification (requires Firefox telemetry)

---

## Cycle 4 — 2026-04-07T00:30Z

**Changes made:**

1. **[CONTENT] Multi-target table expanded** — Added "LUT agg." column to Table 5 (multi-target results) with LUT aggregation errors: transfer_bytes 21.9%, download_ms 37.1%, load_ms 14.2%, ttfb_ms 13.8%. Previously these numbers appeared only in Figure 10 and body text; now verifiable from the table. Updated body text to explicitly reference the download_ms aggregation result (model 16.6% vs LUT 37.1%), which was previously unmentioned despite being a strong result.

2. **[PRESENTATION] Monthly → quarterly retraining fixed** — Deployment section said "monthly retraining" while temporal analysis (Section 5.6) and Limitations both said "quarterly retraining." Changed deployment section to "quarterly retraining" with note that monthly HTTP Archive crawls make this feasible with a one-crawl lag.

3. **[CONTENT] "Architecture dominance" claim tightened** — Abstract, Contribution #2, and Conclusion all claimed "loss function selection dominates architecture choice" but no architecture comparison table exists in the paper. Replaced with a more defensible claim: "loss function selection is the dominant modeling choice, with the 23% gap from switching losses dwarfing the 0.6% gap from tuning the Tweedie variance parameter (p=1.2 vs p=1.5)." Changed in all three locations.

4. **[STRUCTURAL] Section 6.2 trimmed** — Removed the repeated "100-byte error on a beacon" example that duplicated Section 5.2 body text. Kept only what is unique to Section 6.2: the Tweedie gradient scaling formula (ŷ^(2-p)) and the aggregation connection. Saved ~2 sentences of page space.

5. **[PRESENTATION] Repetitive phrase fixed in correlated browsing paragraph** — The phrase "systematic per-domain bias...concentrates on fewer domains" appeared in two consecutive sentences (artifact of Cycle 3 edit). Merged into one sentence.

6. **[PRESENTATION] Calibration figure caption corrected** — Removed "tracks the diagonal for beacons" (overclaim: pred_mean=20 vs actual_mean=63 is off-diagonal). Replaced with "well-calibrated for the beacon mass (0–100B, 51.3% within 25%)."

**Background jobs:** Tree-count ablation attempted (PID 91898, log: `logs/tree_count_ablation_cycle4.log`) but results were internally inconsistent (200 trees MAE < 500 trees MAE), indicating a split-alignment issue between the saved URL embeddings (`url_embeddings_tfidf.npy`, n=523,624) and the script's random split. Will require fixing the split alignment in a future cycle. Results NOT incorporated into paper.
**PDF archived:** `versions/paper_CYCLE_4.pdf`

---

**Deferred to Cycle 5:**
- Tree-count ablation (fix split alignment: use saved URLEmbedder joblib to recompute test embeddings from scratch rather than loading pre-saved npy)
- Domain-held-out split (requires retraining)
- SHAP feature importance
- N=200 browsing load justification (requires Firefox telemetry)

---

## Cycle 3 — 2026-04-07T00:00Z

**Changes made:**

1. **[PRESENTATION] Correlated browsing claim corrected** — The text and Table 6 caption both claimed "the model's relative advantage *increases* under correlated browsing" and cited ratios 2.8× vs 3.5× as evidence. But 2.8 < 3.5, so the ratios actually show the opposite. Fixed by replacing "relative advantage" with "absolute advantage" and citing the percentage-point gap (15.9 pp uniform → 23.5 pp correlated at N=200), which correctly supports an "increases" framing. Changed in both the table caption and the body text at ~line 573.

2. **[PRESENTATION] Calibration table mislabeled** — The table header said "Actual size range" but the bins are defined by `pd.cut(df['predicted'], ...)` in the source code — they are predicted-value bins, not actual-value bins. Fixed: header changed to "Predicted size range." Caption updated to explain that bins group rows by predicted value and show the actual mean for each group. Body text updated to correctly describe the 500B–1KB stratum (rows the model predicts as 500–1000B have actual mean 2,144B, actual median 563B; the mean/median divergence signals occasional large misclassifications rather than systematic underprediction of mid-range responses). Limitations section updated to match.

3. **[CONTENT] Covariate shift AUC added to Section 3.2** — Added one sentence citing `distribution_shift.auc = 0.557` from `advanced_analysis_results.json`. A classifier trained to distinguish HTTP Archive training samples from evaluation samples achieves AUC=0.557, confirming mild covariate shift. Directly supports the qualitative argument already in the section.

4. **[STRUCTURAL] Removed redundant Figure 4 (per-resource-type bar chart)** — Removed `fig4_per_resource_type.pdf` figure from Section 5.4; the same data is fully covered by Table 4. Generated new calibration figure `fig4_calibration.pdf` (log-log predicted vs actual mean scatter + per-bin accuracy bar chart) and inserted it as Figure 4 in Section 5.7. The calibration section now has its own figure as intended.

5. **[CONTENT] Self-training paragraph strengthened** — Added: (a) explanation that non-measured domains cluster at global-fallback feature values, making disjointness structural; (b) algorithm specifics: pseudo-labeling with confidence threshold 0.8, three iterations; (c) why it failed: high-confidence pseudo-labels concentrated among inexpensive domains already well-served by the LUT.

**Background jobs:** Calibration figure generation completed (PID 88896, log at `logs/calibration_fig_cycle3.log`).
**PDF archived:** `versions/paper_CYCLE_3.pdf`

---

**Deferred to Cycle 4:**
- Domain-held-out split (requires retraining)
- SHAP feature importance (requires inference rerun)
- Temporal path churn characterization (which path types change most June→September?)
- The `error_ood_correlation = 0.0496` metric is not yet cited in the paper

---

## Cycle 2 — 2026-04-06

**Changes made:**

1. **[STRUCTURAL] Abstract rewritten** — Now leads with the key measurement finding (within-domain variance predictable from URL structure) and foregrounds the stronger result (model beats path LUT on matched paths: 1,346 vs 1,448). Added explicit note that the 8.7% aggregate improvement understates the structural advantage since the model wins on both path subsets. Previously the abstract led with the deployment context.

2. **[EXPERIMENT] Smoothed LUT comparison added** — Ran `src/model/smoothed_lut.py` with Laplace add-k smoothing (k∈{0.5,...,100}). Key finding: best smoothed LUT (k=0.5, MAE 4,011) is 4.9% *worse* than unsmoothed (3,797) and 15.7% worse than the model (3,466). Smoothing hurts because it degrades the 91.6% of matched paths. Added paragraph to Section 5.1 baseline comparison. Results in `models/per_request/smoothed_lut_results.json`.

3. **[CONTENT] Calibration subsection added** — New Section 5.7 (Calibration Analysis) with Table showing per-bin prediction accuracy. Key finding: model is well-calibrated for beacons (0–100B: 51.3% within 25%) and JS bundles (50–100KB: 86.6%), but systematically underpredicts in 500B–5KB range (predicted mean 697 vs actual mean 2,144 in 500B–1KB bin). Added corresponding limitation paragraph.

4. **[PRESENTATION] Rounding inconsistency fixed** — Line 526: "5.0%" → "5.1%" and "23.0%" → "23.2%" to match Table 5. Source: `full_experiment_results.json` at N=500.

5. **[PRESENTATION] Tweedie p optimality language fixed** — Added sentence noting that the 20-byte difference between p=1.2 (MAE 3,486) and p=1.5 (MAE 3,466) is within bootstrap variance; any p∈[1.2, 1.5] is effectively equivalent.

6. **[PRESENTATION] Table 1 denominator clarified** — Added note to caption explaining that "vs. global" uses global median (13,661) as denominator, while Section 5 improvement figures use domain+type LUT (6,597) as reference.

7. **[PRESENTATION] Simulation seed note added** — Brief parenthetical explaining why the uniform column in the correlated browsing table (Table 6) shows 6.3% at N=200 vs 6.0% in Table 5 (independent simulation runs).

8. **[CONTENT] Limitations: calibration bias paragraph added** — New paragraph disclosing mid-range calibration gap (500B–5KB) and its implications for per-request vs. aggregate use cases.

**Background jobs:** Smoothed LUT completed (PID logged, output at `logs/smoothed_lut_cycle2.log`).  
**PDF archived:** `versions/paper_CYCLE_2.pdf`

---

**Deferred to Cycle 3:**
- Domain-held-out split (requires retraining)
- SHAP feature importance (requires model inference rerun)
- Remove redundancy between Figure 4 and Table 4 (resource-type breakdown)
- Self-training KS test statistic (D value not in existing JSON, would require rerunning analysis)


