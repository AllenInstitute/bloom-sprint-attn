# Test plan

Test descriptions organized by module, with spec section references.

Run all tests: `pytest tests/ -v`

---

## Genome Codebook (`tests/test_genome_codebook.py`)

Tests for `GenomeCodebook` (spec §5-§6).

| Test | What it verifies |
|------|-----------------|
| `test_codebook_deterministic` | Same seed produces identical u and r buffers |
| `test_codebook_r_is_pm_one` | All r entries are exactly {-1, +1} |
| `test_codebook_u_is_unit_norm` | All u rows have unit L2 norm |
| `test_codebook_forward_indexing` | forward(g) output matches buffer rows |
| `test_codebook_no_parameters` | Zero learnable parameters |
| `test_codebook_different_seed` | Different seed → different codes |
| `test_codebook_approximate_orthogonality` | u_i · u_j ≈ 0 for i ≠ j |

---

## Positional Encoder (`tests/test_pos_encoder_v2.py`)

Tests for `PosEncoder` (spec §4-§6).

| Test | What it verifies |
|------|-----------------|
| `test_encoder_output_shapes` | Correct u and r_phi tensor shapes |
| `test_encoder_no_parameters` | Zero learnable parameters |
| `test_encoder_same_chrom_sign_cancellation` | r_phi_i · r_phi_j = phi_i · phi_j when same g |
| `test_encoder_cross_chrom_decorrelation` | Mean cross-chrom dot ≈ 0 |
| `test_encoder_deterministic` | Same inputs always produce same outputs |

---

## Sequence Heads (`tests/test_sequence_heads.py`)

Tests for `SequenceHeads` (spec §3.1).

| Test | What it verifies |
|------|-----------------|
| `test_output_shape` | Output shape is [B, L, H*d_v] |
| `test_permutation_equivariance` | Permuting L gives permuted output |
| `test_gradients_flow` | W_Q, W_K, W_V all receive gradients |
| `test_single_head` | Works with H=1 |
| `test_different_d_seq_d_s` | Q/K and V can have different input dims |

---

## Position Heads (`tests/test_position_heads.py`)

Tests for `PositionHeads` (spec §3.2, §4.2).

| Test | What it verifies |
|------|-----------------|
| `test_output_shape` | Output shape is [B, L, H*d_v] |
| `test_logit_matches_formula` | Q·K^T = τ * (β² u·u^T + (w·r_phi)·(w·r_phi)^T) |
| `test_pad_v_parity` | Padded vs unpadded give identical results |
| `test_w_tiled_sin_cos_sharing` | Adjacent sin/cos pairs share weight |
| `test_head_scale_ordering` | Larger target_scale → wider kernel |
| `test_learn_tau_creates_parameter` | learn_tau=True makes log_tau a Parameter |
| `test_learn_tau_false_creates_buffer` | learn_tau=False makes log_tau a buffer |
| `test_learn_w_creates_parameter` | learn_w=True makes log_w a Parameter |
| `test_learn_w_false_creates_buffer` | learn_w=False makes log_w a buffer |
| `test_gradients_flow` | All learnable params (log_w, beta, log_tau, W_V) get gradients |

---

## Standard Heads (`tests/test_standard_heads.py`)

Tests for `StandardHeads` (spec §3, head type 3).

| Test | What it verifies |
|------|-----------------|
| `test_output_shape` | Output shape is [B, L, H*d_v] |
| `test_permutation_equivariance` | Permuting L gives permuted output |
| `test_gradients_flow` | W_Q, W_K, W_V all receive gradients |
| `test_single_head` | Works with H=1 |
| `test_different_dims` | Handles various d_in values |

---

## Integration (`tests/test_bloom_block.py`)

End-to-end tests for `BloomAttentionBlock`.

| Test | What it verifies |
|------|-----------------|
| `test_output_shape` | Output is [B, L, d_out] with default d_out |
| `test_output_shape_custom_d_out` | Output respects custom d_out |
| `test_all_head_types_run` | No errors with all three head counts > 0 |
| `test_zero_sequence_heads` | Works with n_sequence_heads=0 |
| `test_zero_position_heads` | Works with n_position_heads=0 |
| `test_zero_standard_heads` | Works with n_standard_heads=0 |
| `test_permutation_equivariance` | Permuting L for all inputs permutes output |
| `test_same_chrom_nearby_preference` | Position heads prefer nearby same-chrom features |
| `test_all_params_receive_gradients` | Full backward reaches all learnable parameters |
| `test_integer_g_contract` | g must be LongTensor (integer indices) |
| `test_bf16_cuda_smoke` | No NaN/Inf in bf16 on CUDA (skipped if no GPU) |
