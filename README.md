This is a program that takes any number and finds the partitions for it below are the steps to using it ;
1,Press Enter at any prompt to accept the default.

2,Mode 3 (exact) stores full big integers — p(10,000) already has ~110 digits, so restrict this to n ≲ 10,000 on a normal machine.

3,Mode 4 (Numba) requires pip install numba numpy; it's ~50× faster than pure Python and easily handles n = 10,000,000.

4,After the main computation, the script optionally lets you dump a range of p(i) values

5, Errors (bad integer, unknown mode, missing Numba, out-of-memory) are handled gracefully with sensible fallbacks.