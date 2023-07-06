
_bench_model_cv :
	$(eval exp_name = $(EXP_NAME))
	$(eval mcs_opts = -ast 100_000 -mcs 1 -ci 1000 -ms 1_000_000) $(eval mcs_suf = )  #  single
	$(eval opts = $(base_opts) $(mcs_opts) -ul)           $(eval suf = $(base_suf)$(mcs_suf).lax)
	make _run_gp_ds1 exp_name=$(exp_name) opts="$(opts)" suf=$(suf)
	$(eval mcs_opts = -ast 50_000 -mcs 2 -ci 500 -ms 500_000) $(eval mcs_suf = .mcs2)  #  2 samples
	$(eval opts = $(base_opts) $(mcs_opts) -loo         ) $(eval suf = $(base_suf)$(mcs_suf).loo)
	make _run_gp_ds1 exp_name=$(exp_name) opts="$(opts)" suf=$(suf)
	$(eval opts = $(base_opts) $(mcs_opts) -loo -ul    )  $(eval suf = $(base_suf)$(mcs_suf).loo.lax)
	make _run_gp_ds1 exp_name=$(exp_name) opts="$(opts)" suf=$(suf)
	$(eval mcs_opts = -ast 33_333 -mcs 3 -ci 333  -ms 333_333) $(eval mcs_suf = .mcs3)  #  3 samples   # 100000/3
	$(eval opts = $(base_opts) $(mcs_opts) -loo         ) $(eval suf = $(base_suf)$(mcs_suf).loo)
	make _run_gp_ds1 exp_name=$(exp_name) opts="$(opts)" suf=$(suf)
	$(eval opts = $(base_opts) $(mcs_opts) -loo -ul    )  $(eval suf = $(base_suf)$(mcs_suf).loo.lax)
	make _run_gp_ds1 exp_name=$(exp_name) opts="$(opts)" suf=$(suf)


bench_model_cv_all1:
	$(eval seeds = 0 1 2 3 4)
	$(eval emb_dims = 2 3 4)
	$(eval emb_qscales = 1e-1)
	$(eval emb_q_opts = -es euclid  -edt diag ) $(eval emb_q_suf = .n)
	$(eval base_opts = $(emb_q_opts)) $(eval base_suf = $(emb_q_suf))
	$(foreach seed, $(seeds), \
		$(foreach emb_dim, $(emb_dims), \
			$(foreach emb_qscale, $(emb_qscales), \
				$(eval _base_opts = -s $(seed) $(base_opts) -ed $(emb_dim) -eqs $(emb_qscale)) $(eval _base_suf = .s$(seed)$(base_suf)$(emb_dim).eqs$(emb_qscale)) \
				make _bench_model_cv seed=$(seed) base_opts="$(_base_opts)" base_suf=$(_base_suf); \
			)))

bench_model_cv_all:
	$(eval base_opts = ) $(eval base_suf = )
	$(eval base_opts = -ua) $(eval base_suf = .ua)
	$(eval _base_opts = $(base_opts) -es euclid  -edt diag ) $(eval _base_suf = $(base_suf).n)
	make bench_model_cv_all1 base_opts="$(_base_opts)" base_suf=$(_base_suf)
	$(eval _base_opts = $(base_opts) -es euclid  -edt full ) $(eval _base_suf = $(base_suf).mvn)
	make bench_model_cv_all1 base_opts="$(_base_opts)" base_suf=$(_base_suf)
	$(eval _base_opts = $(base_opts) -es lorentz -edt diag ) $(eval _base_suf = $(base_suf).wn)
	make bench_model_cv_all1 base_opts="$(_base_opts)" base_suf=$(_base_suf)
	$(eval _base_opts = $(base_opts) -es lorentz -edt full) $(eval _base_suf = $(base_suf).wmvn)
	make bench_model_cv_all1 base_opts="$(_base_opts)" base_suf=$(_base_suf)
