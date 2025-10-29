## Aliounelo/faceRecognition-alioune/opencv-3.4/modules/cudaimgproc/src/cuda/build_point_list.cu
#### Analysis:

- `buildPointList`
    CLang could not recognize one of the conditions `if (y < src.rows)`.
    2d write access is missing `s_queues[threadIdx.y][qidx] = val;`
