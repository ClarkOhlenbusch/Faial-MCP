## shyamshankarhr/image-captioning/fairseq/modules/lightconv_layer/lightconv_cuda_kernel.cu
#### Analysis:

- `lightconv_forward_kernel` missing a sync from line 112 in .cu?

- `lightconv_grad_wrt_input_kernel` ~~read index is incorrect~~?
    This is not being recognized:
    `const scalar_t* inputFilter = &filters[filterIdx * FS];`

    MAPS:
        `ro inputFilter[i];`
    CUDA:
        `filter[i] = inputFilter[FS - i - 1];`

- `lightconv_grad_wrt_weights_firstpass_short_kernel` MAPs correctly inferred syncs within nested for-loops.

- `lightconv_grad_wrt_weights_secondpass_short_kernel` correct.

- `lightconv_grad_wrt_weights_firstpass_kernel` MAPs correctly inferred syncs within nested for-loops.

- `lightconv_grad_wrt_weights_secondpass_kernel` correct.
