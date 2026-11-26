s_mov_b32 s12, s7
s_mov_b32 s13, 0
s_load_dwordx8 s[40:47], s[12:13], 0
s_waitcnt lgkmcnt(0)
s_mov_b32 s9, 0
s_lshl_b32 s23, s6, 5
s_load_dwordx4 s[28:31], s[8:9], s23
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v4, 2, v0
buffer_load_dword v7, v4, s[28:31], 0 offen
s_waitcnt vmcnt(0)
buffer_store_dword v7, v4, s[44:47], 0 offen
s_endpgm
