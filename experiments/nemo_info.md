# Structure

Layer map (52 layers total)

  ┌─────────────┬──────────────────────────────────────────────────────────────────┬───────┐
  │    Type     │                             Indices                              │ Count │
  ├─────────────┼──────────────────────────────────────────────────────────────────┼───────┤
  │ Mamba-2 SSM │ 0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50  │ 23    │
  ├─────────────┼──────────────────────────────────────────────────────────────────┼───────┤
  │ MoE FFN     │ 1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51 │ 23    │
  ├─────────────┼──────────────────────────────────────────────────────────────────┼───────┤
  │ Attention   │ 5, 12, 19, 26, 33, 42                                            │ 6     │
  └─────────────┴──────────────────────────────────────────────────────────────────┴───────┘

  The repeating unit is roughly Mamba → MoE → Mamba → MoE → Mamba → Attention → MoE (7-layer blocks), with attention every ~7 layers. 
  **Each MoE layer has 128 experts with up_proj/down_proj**

  https://arxiv.org/pdf/2501.01386
  https://developer.nvidia.com/blog/tutorial/building-large-language-models-with-nemotron-nemo-3/#4




 ---
  # CoT Analysis: expert_inference_log.txt

  ## Environment

  - GPU: RTX PRO 6000 Blackwell, 102 GB VRAM
  - Model: NemotronH 30B-A3B, BF16 full precision — 32.46B params as loaded
  - LoRA: lora3-checkpt800-1188, checkpoint-1188
  - VRAM at rest: 66.73 GB | Peak per inference call: ~70.2 GB (~3.5 GB overhead)

  ## Overall Score: 6/14 passed (+3 unknown)
  
  ┌────────┬───────┬──────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Result │ Count │                                              Tasks                                               │
  ├────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ PASS   │ 6     │ numeral_system, unit_conversion, symbol_transform, text_cipher, physics_gravity, cipher—Caesar+7 │
  ├────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ FAIL   │ 8     │ unit_conversion—non-round, bit_manipulation ×3, bit_manip ×2, symbol ×2, text_cipher_bijection   │
  ├────────┼───────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ UNKN   │ 3     │ bit_manipulation_v2/v3, text_cipher_v2                                                           │
  └────────┴───────┴──────────────────────────────────────────────────────────────────────────────────────────────────┘

  The UNKN cases returned non-binary outputs (-7, 0, 4) that didn't match the expected format — likely scoring script failures rather than model
  failures, worth investigating separately.

  ---
  ## Failure Pattern Analysis
  
 ### Pattern 1 — Correct reasoning, wrong final step (unit_conversion non-round)

  The model correctly derived the conversion factor as 0.6635 via least-squares regression across 5 examples, correctly computed 30.0 × 0.6635 = 
  19.905, was in the process of rounding to 19.91... then boxed 0.663 instead. The CoT was over 460 lines. This is a context truncation failure — the
   reasoning chain hit a generation limit before the final answer token was written. The factor, not the result, ended up in the box. This is
  recoverable with a max_new_tokens increase or a CoT compression prompt.

  ### Pattern 2 — Hypothesis explosion without commitment (bit manipulation)

  All four bit_manipulation failures show the same pathology: the model correctly identifies a partial rule (e.g., bitwise reversal works for
  examples 1–3, fails for example 4), recognizes the inconsistency, then explores GF(2) linear algebra, S-box hypotheses, ARX structures, and
  nibble-level decompositions — without converging. The boxed outputs are truncated or wrong because the model never exits the exploration phase. The
   CoT for case 5 (XOR-then-swap-nibbles) alone exceeded 150 lines.

  This is a systematic weakness: the model reasons well locally but lacks a meta-strategy for hypothesis pruning. It doesn't know when to commit to a
   best guess under ambiguity.

  ### Pattern 3 — Missing the final application step (symbol cycle)

  Symbol shift/cycle failures: the model identifies the symbol set and shift rules but loses the state when applying them across positions. These
  require maintaining an ordered state machine, which taxes the Mamba SSM state harder than a single-step calculation.

  ---
  ## CoT Quality by Task Type

  ┌─────────────────────────────┬────────────────┬───────────────────────────────────────────┬───────────────────────────┐
  │         Task family         │   CoT length   │                  Quality                  │       Failure mode        │
  ├─────────────────────────────┼────────────────┼───────────────────────────────────────────┼───────────────────────────┤
  │ Numeral/unit (simple)       │ 3–10 lines     │ Excellent — immediate pattern recognition │ —                         │
  ├─────────────────────────────┼────────────────┼───────────────────────────────────────────┼───────────────────────────┤
  │ Physics                     │ ~5 lines       │ Clean formula retrieval                   │ —                         │
  ├─────────────────────────────┼────────────────┼───────────────────────────────────────────┼───────────────────────────┤
  │ Text cipher                 │ ~20 lines      │ Good substitution tracking                │ Bijection case fails      │
  ├─────────────────────────────┼────────────────┼───────────────────────────────────────────┼───────────────────────────┤
  │ Bit manipulation            │ 150–600+ lines │ Thorough but non-convergent               │ Hypothesis overgeneration │
  ├─────────────────────────────┼────────────────┼───────────────────────────────────────────┼───────────────────────────┤
  │ Unit conversion (non-round) │ 460 lines      │ Correct reasoning, wrong box              │ Context truncation        │
  └─────────────────────────────┴────────────────┴───────────────────────────────────────────┴───────────────────────────┘

  ---
  ## The LoRA Params = 0 Issue

  Adapter loaded.
  LoRA params  : 0  (0.000%)
  
  This means the adapter was almost certainly merged into the base weights via merge_and_unload() before the parameter count was taken. Merged LoRA
  parameters become indistinguishable from base model weights — the adapter ceases to exist as a separate module. The 6/14 performance is consistent
  with a working adapter. To verify: check whether model is still a PeftModel instance (hasattr(model, 'peft_config')). If it's a plain
  AutoModelForCausalLM, the weights were merged.

  ---
  # Expert Routing Analysis
  
 ## Architecture note — 7 active experts, not 6

  The header "Top 6 Routed + 1 Shared" reveals a shared expert design (same pattern as DeepSeekMoE). One expert always fires regardless of routing.
  The 6 specialists shown are the routed component; the shared expert's contribution is implicit in every forward pass. This shared expert is the
  model's "always-on" backbone.
  
 ## Utilization across the test suite

  ┌───────┬────────────────┬────────────┬───────┬──────────────────────┐
  │ Layer │ Active experts │ Top expert │ Share │      Character       │
  ├───────┼────────────────┼────────────┼───────┼──────────────────────┤
  │ 1     │ 23/128         │ E12        │ 1.9%  │ Broad, diffuse       │
  ├───────┼────────────────┼────────────┼───────┼──────────────────────┤
  │ 10    │ 22/128         │ E48        │ 11.3% │ Highly concentrated  │
  ├───────┼────────────────┼────────────┼───────┼──────────────────────┤
  │ 15    │ 30/128         │ E17        │ 9.2%  │ Mid-stack generalist │
  ├───────┼────────────────┼────────────┼───────┼──────────────────────┤
  │ 24    │ 22/128         │ E20        │ 11.1% │ Highly concentrated  │
  ├───────┼────────────────┼────────────┼───────┼──────────────────────┤
  │ 31    │ 28/128         │ E75        │ 8.3%  │                      │
  ├───────┼────────────────┼────────────┼───────┼──────────────────────┤
  │ 34    │ 34/128         │ E4         │ 9.9%  │ Most diverse         │
  ├───────┼────────────────┼────────────┼───────┼──────────────────────┤
  │ 40    │ 30/128         │ E112       │ 10.1% │                      │
  ├───────┼────────────────┼────────────┼───────┼──────────────────────┤
  │ 47    │ 35/128         │ E116       │ 3.6%  │ Most diverse         │
  └───────┴────────────────┴────────────┴───────┴──────────────────────┘
  
 ### Only 22–35 of 128 experts ever fired across this entire test suite — roughly 75% of expert capacity was untouched. 

 ## Domain-specific routing fingerprints

  Cross-referencing per-task expert reports against task types reveals clear domain specialists:

  - E4 (layers 8, 13, 34): fires almost exclusively on bit manipulation tasks — likely a bitwise/logical operations specialist
  - E112 (layer 40): dominant for bit manipulation, barely appears elsewhere — dedicated low-level symbol transformation expert
  - E48 (layer 10): appears across numeral, bit_manipulation, and unit_conversion — general mathematical operations
  - E69 (layer 15): strongly dominant for unit_conversion tasks; lighter on others — measurement/scaling specialist
  - E75 (layer 31): bit manipulation and symbolic reasoning — abstract rule application
  - E20 (layer 24): broad activation across multiple task types — likely a domain-general mid-network expert

 ### The 100× activation magnitude anomaly

  Cases 3–6 (non-round conversion, complex bit manipulations) show activation magnitudes 100–300× higher than cases 1–2. This correlates almost
  perfectly with FAIL outcomes. **Hypothesis: magnitude inflation is a proxy for model uncertainty** — when the router can't confidently commit to a
  small expert subset, it activates more experts at higher weights as the model "struggles." This could be a real-time failure predictor. Cases with
  peak layer activations > 10⁶ all failed; cases below 10⁴ all passed

  ---
  # Key Takeaways
  
  1. The 75% unused expert capacity is significant — your surrealist/associative data will very likely route through a largely distinct expert
  neighborhood from what's shown here, which is the best-case scenario for preserving analytical capability while adding personality.
  2. Context truncation is a real problem — the non-round conversion failure was a correct solution that got cut off. Set max_new_tokens high enough
  for your intended CoT depth; 2048 minimum, 4096 if possible for complex chains.
  3. Bit manipulation is a structural weakness — the model over-explores without pruning. If your Kaggle training data contained many
  bit-manipulation-style problems, that weakness propagated. Consider adding a meta-reasoning instruction in your system prompt: "when uncertain,
  commit to the best hypothesis after exploring three alternatives."
  4. The magnitude signal is worth instrumenting — if you add activation monitoring to your inference script, per-layer max activation magnitude
  could give you a real-time quality signal without needing ground truth labels.