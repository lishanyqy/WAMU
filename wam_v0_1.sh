Loading python3/3.12.13
  Loading requirement: intel-mkl/2025.3.1

===== RANK 0 EXCEPTION (pre-destroy) =====

===== RANK 3 EXCEPTION (pre-destroy) =====

===== RANK 1 EXCEPTION (pre-destroy) =====

===== RANK 2 EXCEPTION (pre-destroy) =====
Traceback (most recent call last):
  File "/g/data/zv32/rz6525/WAMU/scripts/train.py", line 167, in main
    _train(cfg)
  File "/g/data/zv32/rz6525/WAMU/scripts/train.py", line 113, in _train
    _train_openwam(cfg)
  File "/g/data/zv32/rz6525/WAMU/scripts/train.py", line 133, in _train_openwam
    accelerator = _build_accelerator(cfg)
                  ^^^^^^^^^^^^^^^^^^^^^^^
  File "/g/data/zv32/rz6525/WAMU/scripts/train.py", line 86, in _build_accelerator
    return accelerate.Accelerator(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/accelerate/accelerator.py", line 463, in __init__
    self.state = AcceleratorState(
                 ^^^^^^^^^^^^^^^^^
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/accelerate/state.py", line 978, in __init__
    deepspeed_plugin.select(_from_accelerator_state=True)
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/accelerate/utils/dataclasses.py", line 1562, in select
    self.set_deepspeed_weakref()
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/accelerate/utils/dataclasses.py", line 1482, in set_deepspeed_weakref
    from transformers.integrations import (
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/transformers/__init__.py", line 30, in <module>
    from . import dependency_versions_check
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/transformers/dependency_versions_check.py", line 56, in <module>
    require_version_core(deps[pkg])
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/transformers/utils/versions.py", line 116, in require_version_core
    return require_version(requirement, hint)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/transformers/utils/versions.py", line 110, in require_version
    _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/transformers/utils/versions.py", line 43, in _compare_versions
    raise ImportError(
ImportError: huggingface-hub>=1.5.0,<2.0 is required for a normal functioning of this module, but found huggingface-hub==2.0.0.
Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main
Error executing job with overrides: ['model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B']
Error executing job with overrides: ['model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B']
Error executing job with overrides: ['model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B']
Error executing job with overrides: ['model.video_backbone.model_path=assets/video_backbone_ckpt/Wan2.2-TI2V-5B']

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
Traceback (most recent call last):
  File "/g/data/zv32/rz6525/WAMU/scripts/train.py", line 167, in main
    _train(cfg)
  File "/g/data/zv32/rz6525/WAMU/scripts/train.py", line 113, in _train
    _train_openwam(cfg)
  File "/g/data/zv32/rz6525/WAMU/scripts/train.py", line 133, in _train_openwam
    accelerator = _build_accelerator(cfg)
                  ^^^^^^^^^^^^^^^^^^^^^^^
  File "/g/data/zv32/rz6525/WAMU/scripts/train.py", line 86, in _build_accelerator
    return accelerate.Accelerator(
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/accelerate/accelerator.py", line 463, in __init__
    self.state = AcceleratorState(
                 ^^^^^^^^^^^^^^^^^
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/accelerate/state.py", line 978, in __init__
    deepspeed_plugin.select(_from_accelerator_state=True)
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/accelerate/utils/dataclasses.py", line 1562, in select
    self.set_deepspeed_weakref()
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/accelerate/utils/dataclasses.py", line 1482, in set_deepspeed_weakref
    from transformers.integrations import (
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/transformers/__init__.py", line 30, in <module>
    from . import dependency_versions_check
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/transformers/dependency_versions_check.py", line 56, in <module>
    require_version_core(deps[pkg])
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/transformers/utils/versions.py", line 116, in require_version_core
    return require_version(requirement, hint)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/transformers/utils/versions.py", line 110, in require_version
    _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/transformers/utils/versions.py", line 43, in _compare_versions
    raise ImportError(
ImportError: huggingface-hub>=1.5.0,<2.0 is required for a normal functioning of this module, but found huggingface-hub==2.0.0.
Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
W1001 13:56:03.227000 1660592 site-packages/torch/distributed/elastic/multiprocessing/api.py:1028] Sending process 1660619 closing signal SIGTERM
W1001 13:56:03.229000 1660592 site-packages/torch/distributed/elastic/multiprocessing/api.py:1028] Sending process 1660621 closing signal SIGTERM
E1001 13:56:03.344000 1660592 site-packages/torch/distributed/elastic/multiprocessing/api.py:1002] failed (exitcode: 1) local_rank: 0 (pid: 1660618) of binary: /scratch/ou29/rz6525/conda3/envs/openwam/bin/python
Traceback (most recent call last):
  File "/scratch/ou29/rz6525/conda3/envs/openwam/bin/torchrun", line 6, in <module>
    sys.exit(main())
             ^^^^^^
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 380, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/torch/distributed/run.py", line 1138, in main
    run(args)
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/torch/distributed/run.py", line 1129, in run
    elastic_launch(
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 197, in __call__
    return launch_agent(
           ^^^^^^^^^^^^^
  File "/scratch/ou29/rz6525/conda3/envs/openwam/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 386, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
scripts/train.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2026-10-01_13:56:03
  host      : gadi-gpu-h200-0018.gadi.nci.org.au
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 1660620) 
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2026-10-01_13:56:03
  host      : gadi-gpu-h200-0018.gadi.nci.org.au
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 1660619) 
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2026-10-01_13:56:03
  host      : gadi-gpu-h200-0018.gadi.nci.org.au
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 1660621) 
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-10-01_13:56:03
  host      : gadi-gpu-h200-0018.gadi.nci.org.au
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 1660618) 
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
