# Changelog

All changes we make to the assignment code or PDF will be documented in this file.


## [1.0.1] - 2024-04-17
### Added
- code: Include tests for logsumexp in flash forward implementation
- handout: Clarify interface for flash autograd function
- code: Test causal=True for forward as well as backward
- code: Simplify BasicsTransformerLM (remove flags and logic for A1 ablations)
- code: In BasicsTransformerLM, broadcast causal mask over all heads

## [1.0.0] - 2024-04-16

### Added
- handout/code: add FlashAttention2
- handout: add proper profiling with Nsight Systems
- handout: add communication accounting
- code: tests for additional content

### Changed
- handout: greatly improve demo example for Triton
- handout: remove busywork for communication

### Fixed

- handout: clarify that `ddp_bucketed_benchmarking` doesn't require the full
  grid of runs.

## [0.0.4] - 2024-04-23

### Added

### Changed

- code: remove try-finally blocks in DDP tests.

### Fixed

- handout: remove outdated mention of a problem that doesn't exist on the assignment
- handout: fix Slurm environment variables in examples.
- handout: clarify assumptions in `ddp_bucketed_benchmarking` (b).

## [0.0.3] - 2024-04-21

### Added

### Changed

- code: remove `humanfriendly` from requirements.txt, add `matplotlib`
- handout: modify problem `distributed_communication_multi_node` to specify that
  multinode measurements should be 2x1, 2x2, and 2x3.
- handout: clarify that `torch.cuda.synchronize()` is necessary for timing
  collective communication ops, even when they are called with `async_op=False`.

### Fixed

- handout: fixed cut off text in problem memory_profiling (a)
- handout: fixed mismatch between slurm config and description text in section 3.2
- code: fix `ToyModelWithTiedWeights` to actually tie weights.
- handout: fix typo in bucketed DDP test command, should be `pytest tests/test_ddp.py` 
- handout: fix deliverable of `ddp_overlap_individual_parameters_benchmarking`
  (a) to not ask for communication time, only end-to-end step time.
- handout: clarify analysis in `optimizer_state_sharding_accounting` (a).

## [0.0.1] - 2024-04-17

### Added

- handout: added a short question about variability on problem benchmarking_script

### Changed

### Fixed

- handout: fixed typo in problem `triton_rmsnorm_forward`. The adapters should
  return the classes, not the `.apply` attribute.
- code: added `-e` flag to `./cs336-systems/'[test]'`
- handout: clarified recommendation about the timeit module
- handout: clarified question about kernel with highest CUDA total

## [0.0.0] - 2024-04-16

Initial release.
