EXP_NAME = paper
include make/*.mk

DS_DATA = ds-data
BASE_DIR = .
PYTHON = python
_RUN_CPU = PYTHONPATH=$(BASE_DIR)

setup:
	git clone https://github.com/matsengrp/ds-data
	$(PYTHON) setup.py build_ext --inplace

_run_gp_ds%:
	$(eval ds = ds$*)
	$(eval suf = )
	$(eval exp_name = $(EXP_NAME))
	$(eval input_path = $(DS_DATA)/$(ds)/DS$*.nex)
	$(eval out_prefix = results/$(ds)/$(exp_name)/$(ds).gp$(suf))
	$(eval eval_opts = -ns 1000 -ii 5)
	$(_RUN_CPU) $(PYTHON) $(BASE_DIR)/scripts/run_gp.py -v -ip $(input_path) -op $(out_prefix) -c $(BASE_DIR)/config/default.yaml $(opts)
	$(_RUN_CPU) $(PYTHON) $(BASE_DIR)/scripts/eval_gp.py eval_history -v -sp $(out_prefix) -o $(out_prefix).eval.txt $(eval_opts)
	$(_RUN_CPU) $(PYTHON) $(BASE_DIR)/scripts/eval_gp.py eval_state -v -sp $(out_prefix) -o $(out_prefix).eval.latest.txt -ns 1000

example:
	make _run_gp_ds1 exp_name=test opts="-ua -es euclid  -edt diag  -ed 2 -eqs 1e-1 -ast 100_000 -mcs 1 -ci 1000 -ms 1_000 -ul"
	#make _run_gp_ds1 exp_name=test opts="-ua -es lorentz -edt full -ed 4 -eqs 1e-1 -ast 100_000 -mcs 1 -ci 1000 -ms 1_000 -ul"
