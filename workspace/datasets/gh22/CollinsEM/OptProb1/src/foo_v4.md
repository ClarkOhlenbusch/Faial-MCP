## CollinsEM/OptProb1/src/foo_v4.cu
#### Analysis:

- `block1D_reduce_sum` 1 read within a for-loop. 1 write within a conditional.
    synchronizations appear correctly.

- `computeY` contains multiple reads and writes. 4 synchronizations.
