# Assigned 07/23/2022

## shamitlal/CoCoNets/src/cuda_ops/corr3d/Correlation_Module/correlation_cuda_kernel.cu

### Kernels:

1. correlation_cuda_forward_kernel
2. correlation_cuda_backward_kernel_input1
3. correlation_cuda_backward_kernel_input2

There were three kernels in the CUDA file, and all of them were analyzed by `c-ast`.

### Warnings:

None

### 1. correlation_cuda_forward_kernel

<hr>

### Reads/Writes

There are 3 reads and 3 writes in this kernel:

```cuda
// Reads
scalar_t v1 = rInput1[i1][j1][k1][n][c];
scalar_t v2 = rInput2[i2][j2][k2][n][c];
reduce_sum += prod_sum[index];

// Writes
prod_sum[thread] = 0;
prod_sum[thread] += v1 * v2;
output[n][pd][ph][pw][d][h][w] = reduce_sum;
```

However, only the reads and writes to `prod_sum` were captured by `c-ast`. The other reads/writes are missing.

```
rw prod_sum[thread];
ro prod_sum[thread];
rw prod_sum[thread];
```

### Loops/Conditionals

There are 9 loops and 4 conditionals in this kernel:

```cuda
// Loops
for(int n = 0; n < N; ++n){
for(int pd = 0; pd < patchD; ++pd){
for(int ph = 0; ph < patchH; ++ph){
for(int pw = 0; pw < patchW; ++pw){
for (int i=0; i<kD; ++i){
for (int j=0; j<kH; ++j){
for (int k=0; k<kW; ++k){
for (int c=thread; c<C; c += THREADS_FORWARD){
for (int index = 0; index < THREADS_FORWARD; ++index) {

// Conditionals
if WITHIN_BOUNDS2(i1, i2, iD, iD){
if WITHIN_BOUNDS2(j1, j2, iH, iH){
if WITHIN_BOUNDS2(k1, k2, iW, iW){
if (thread == 0) {
```

All of them were captured by `c-ast`.

### 2. correlation_cuda_backward_kernel_input1

<hr>

### Reads/Writes

There are 3 reads and 3 writes in this kernel:

```cuda
// Reads
scalar_t val = input2[n][c][i1][j1][k1];
reduce_sum += prod_sum[pd][ph][pw];

// Writes
prod_sum[pd_off][ph_off][pw_off] = 0;
gradInput1[n][c][d][h][w] = reduce_sum;

// Both
prod_sum[pd_off][ph_off][pw_off] += gradOutput[n][pd][ph][pw][i2][j2][k2] * val;
```

However, only one read and one write to `prod_sum` was captured by `c-ast`:

```
rw prod_sum[pd_off, ph_off, pw_off];
ro prod_sum[pd, ph, pw];
```

### Loops/Conditionals

There are 10 loops and 3 conditionals in this kernel:

```cuda
// Loops
for(int c = 0; c < C; c++){
for (int pd = pd_off; pd < patchD; pd += THREADS_BACKWARD){
for (int ph = ph_off; ph < patchH; ph += THREADS_BACKWARD) {
for (int pw = pw_off; pw < patchW; pw += THREADS_BACKWARD) {
for(int tmp1 = d_off, i = 0; tmp1 < kD; tmp1 += dD, ++i) {
for(int tmp2 = h_off, j = 0; tmp2 < kH; tmp2 += dH, ++j) {
for(int tmp3 = w_off, k = 0; tmp3 < kW; tmp3 += dW, ++k) {
for (int pd = 0; pd < THREADS_BACKWARD; ++pd){
for (int ph = 0; ph < THREADS_BACKWARD; ++ph){
for (int pw = 0; pw < THREADS_BACKWARD; ++pw){

// Conditionals
if WITHIN_BOUNDS(i1, j1, k1, iD, iH, iW) {
if WITHIN_BOUNDS(i2, j2, k2, D, H, W) 
if (pd_off == 0 && ph_off == 0 && pw_off == 0){
```

All of the loops were captured by `c-ast`, but the following conditional went missing:

```cuda
if WITHIN_BOUNDS(i2, j2, k2, D, H, W) 
```

### 3. correlation_cuda_backward_kernel_input2

<hr>

### Reads/Writes

There are 3 reads and 3 writes in this kernel.

```cuda
// Reads
scalar_t val = input1[n][c][i1][j1][k1];
reduce_sum += prod_sum[pd][ph][pw];

// Writes
prod_sum[pd_off][ph_off][pw_off] = 0;
gradInput2[n][c][d][h][w] = reduce_sum;

// Both
prod_sum[pd_off][ph_off][pw_off] += gradOutput[n][pd][ph][pw][i2][j2][k2] * val;
```

However, only one read and one write to `prod_sum` was captured by `c-ast`:

```
rw prod_sum[pd_off, ph_off, pw_off];
ro prod_sum[pd, ph, pw];
```

### Loops/Conditionals

There are 10 loops and 3 conditionals in this kernel:

```cuda
// Loops
for(int c = 0; c < C; c++){
for (int pd = pd_off; pd < patchD; pd += THREADS_BACKWARD){
for (int ph = ph_off; ph < patchH; ph += THREADS_BACKWARD) {
for (int pw = pw_off; pw < patchW; pw += THREADS_BACKWARD) {
for(int tmp1 = d_off, i = 0; tmp1 < kD; tmp1 += dD, ++i) {
for(int tmp2 = h_off, j = 0; tmp2 < kH; tmp2 += dH, ++j) {
for(int tmp3 = w_off, k = 0; tmp3 < kW; tmp3 += dW, ++k) {
for (int pd = 0; pd < THREADS_BACKWARD; ++pd){
for (int ph = 0; ph < THREADS_BACKWARD; ++ph){
for (int pw = 0; pw < THREADS_BACKWARD; ++pw){

// Conditionals
if WITHIN_BOUNDS(i1, j1, k1, iD, iH, iW) {
if WITHIN_BOUNDS(i2, j2, k2, D, H, W) 
if (pd_off == 0 && ph_off == 0 && pw_off == 0){
```

All of the loops were captured by `c-ast`, but the following conditional went missing:

```cuda
if WITHIN_BOUNDS(i2, j2, k2, D, H, W) 
```
