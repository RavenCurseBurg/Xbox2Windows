s_mov_b32 s12, s7
s_mov_b32 s13, 0
s_load_dwordx8 s[40:47], s[12:13], 0
s_waitcnt lgkmcnt(0)
s_mov_b32 s9, 0
v_lshlrev_b32_e32 v4, 2, v0
buffer_load_dword v5, v4, s[40:43], 0 offen
s_waitcnt vmcnt(0)
v_and_b32_e32 v5, 3, v5
v_mov_b32_e32 v7, 13
s_mov_b32 s26, exec_lo
v_cmpx_gt_u32_e32 s4, v0
s_mov_b32 s21, exec_lo
v_readfirstlane_b32 s22, v5
v_cmpx_eq_i32_e32 s22, v5
s_lshl_b32 s23, s22, 5
s_load_dwordx4 s[28:31], s[8:9], s23
s_waitcnt lgkmcnt(0)
buffer_load_dword v7, v4, s[28:31], 0 offen
s_waitcnt vmcnt(0)
s_andn2_wrexec_b32 s21, s21
s_cbranch_scc1 -11
s_mov_b32 exec_lo, s26
buffer_store_dword v7, v4, s[44:47], 0 offen
s_endpgm
