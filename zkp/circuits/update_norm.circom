pragma circom 2.0.0;

template UpdateNorm(n) {
    signal input values[n];
    signal input threshold;

    signal squares[n];
    signal partial[n + 1];

    partial[0] <== 0;

    for (var i = 0; i < n; i++) {
        squares[i] <== values[i] * values[i];
        partial[i + 1] <== partial[i] + squares[i];
    }

    // public output: sum of squares
    signal output sum;
    sum <== partial[n];

    /*
      NOTE:
      This circuit computes sum(values[i]^2).
      The actual <= threshold check is not fully enforced yet.
      Next we will add LessThan comparator properly.
    */
}

component main = UpdateNorm(10);