#include "../include/blackhole.cuh"

__device__ int Lookup(vector2 vuv)
{
    vector2 uvarr;
    uvarr.x = Clip<float>(vuv.x, 0.0f, 0.999f); // wszystkie wartoœci wiêksze ni¿ 0.999 i mniejsze ni¿ 0.0 bêd¹ przycinane do tych wartoœci. 2 -> 0.999; -1 -> 0.0
    uvarr.y = Clip<float>(vuv.y, 0.0f, 0.999f); // wszystkie wartoœci wiêksze ni¿ 0.999 i mniejsze ni¿ 0.0 bêd¹ przycinane do tych wartoœci. 2 -> 0.999; -1 -> 0.0

    uvarr.x *= float(DIM_X_N(BITMAP_MULTIPIER)); // tablica dwuwymiarowa, mno¿ymy zerow¹ kolumnê tablicy dwuwymiarowej ## shape to d³ugoœæ w danym wymiarze
    uvarr.y *= float(DIM_Y_N(BITMAP_MULTIPIER)); // tablica dwuwymiarowa, mno¿ymy pierwsz¹ kolumnê tablicy dwuwymiarowej ## shape to d³ugoœæ w danym wymiarze
    int offset = (int)uvarr.x + (int)uvarr.y * DIM_X_N(BITMAP_MULTIPIER);
    return offset;
}

__global__ void RayTrace(float* __restrict__ image, cudaTextureObject_t bitmap, Camera_t* camera, BlackHole_t* blackhole)
{
    // walk along the local sky, shooting a ray for each pixel
    int gpu_x = threadIdx.x + blockIdx.x * blockDim.x;
    int gpu_y = threadIdx.y + blockIdx.y * blockDim.y;
    int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

    vector3 view;
    view.x = (gpu_x * 1.0f) / DIM_X - 0.5f;
    //view.y = ((gpu_y * -1.0f) / DIM_Y + 0.5f) * DIM_Y / DIM_X;
    view.y = ((gpu_y * 1.0f) / DIM_Y - 0.5f) * DIM_Y / DIM_X;
    view.z = 1.0f;

    view.x *= camera->FoVtangent;
    view.y *= camera->FoVtangent;

    view = MatrixMultiplyVector(camera->viewMatrix, view);

    vector3 point = camera->cameraPosition;

    vector3 velocity = VectorNormalize(view);

    vector3 colorRGB = Vector3Zero();
    float colorAlpha = 0.f;

    float h2 = VectorLenghtSquared(VectorCross(point, velocity));

    float pointSquared = 1.f;

    for (int iteration = 0; iteration < ITER; iteration++)
    {
        vector3 oldpoint = point;

        //runge-kutta algorithm RK4
        vector3 increment[2];
        rk4(point, velocity, h2, increment);
        velocity = velocity + increment[1];
        point = point + increment[0];

        pointSquared = VectorLenghtSquared(point);

        bool isMaskCrossing = (oldpoint.y > 0.f) ^ (point.y > 0.f);
        bool isMaskDistance = (pointSquared < blackhole->diskOuterSquared) & (pointSquared > blackhole->diskInnerSquared);

        bool diskMask = isMaskCrossing & isMaskDistance;

        if (diskMask)
        {
            float lambdaa = -1.f * (point.y / velocity.y);
            vector3 colpoint = point + velocity * lambdaa;
            float colpointsqr = VectorLenghtSquared(colpoint);

            float temperature = expf(DiskTemp(colpointsqr, 9.2103f)); // 9.2103 = ln(10000)
            //float temperature = DiskTemp(colpointsqr, 10000.f); // 9.2103 = ln(10000)

#ifdef REDSHIFT
            float R = sqrtf(colpointsqr);

            //vector3 disc_velocity = (M_SQRT1_2 * powf(ClipMin<float>((sqrtf(colpointsqr) - 1.f), 0.1f), -0.5f)) * VectorCross(vector3{ 0.f, 1.0f, 0.f }, VectorNormalize(colpoint)); 
            vector3 disc_velocity = (M_SQRT1_2 * rsqrtf(ClipMin<float>((sqrtf(colpointsqr) - 1.f), 0.1f)) * VectorCross(vector3{ 0.f, 1.0f, 0.f }, VectorNormalize(colpoint)));

            float gamma = rsqrtf(1.f - ClipMax<float>(VectorLenghtSquared(disc_velocity), 0.99f));

            float opz_doppler = gamma * (1.f + VectorMultiplyHV(disc_velocity, VectorNormalize(velocity)));
            float opz_gravitational = rsqrtf(1.f - 1.f / ClipMin<float>(R, 1.f));

            temperature /= ClipMin<float>(opz_doppler * opz_gravitational, 0.1f);
#endif

            float intensity = Intensity(temperature);
            vector3 diskColor = colorMap[Colour(temperature)] * 3.92156862f * intensity; // powinno byæ ju¿ znormalizowane do 0..1 zamiast 0..255
            // 3.92156.. = 1000/255

            //float iscotaper = Clip<float>((colpointsqr - blackhole->diskInnerSquared) * 0.3f, 0.f, 1.f);
            //float outertaper = Clip<float>(temperature * 0.001f, 0.f, 1.f);
            //float diskAlpha = diskMask * iscotaper * outertaper;

            float diskAlpha = Clip<float>(VectorLenghtSquared(diskColor), 0.f, 1.f);
            colorRGB = BlendColors(diskColor, diskAlpha, colorRGB, colorAlpha);
            colorAlpha = BlendAlpha(diskAlpha, colorAlpha);
        }
        float oldpointsqr = VectorLenghtSquared(oldpoint);
        bool mask_horizon = (pointSquared < 1.f) & (oldpointsqr > 1.f);
        if (mask_horizon)
        {
            float lambdaa = 1.f - ((1.f - oldpointsqr) / (pointSquared - oldpointsqr));
            vector3 colpoint = lambdaa * point + (1.f - lambdaa) * oldpoint;
            vector3 horizonColour = Vector3Zero();
            float horizonAlpha = mask_horizon;

            colorRGB = BlendColors(horizonColour, horizonAlpha, colorRGB, colorAlpha);
            colorAlpha = BlendAlpha(horizonAlpha, colorAlpha);
        }
    }
    
    float vphi = atan2f(velocity.x, velocity.z);
    float vtheta = atan2f(velocity.y, VectorLenght(vector2{ velocity.x, velocity.z }));

    vector2 vuv = Vector2Zero();

    vuv.x = fmodf(vphi + 4.5f, 2 * M_PI) * M_1_2PI;
    vuv.y = (vtheta + M_PI_2) * M_1_PI;

    int lookup = Lookup(vuv);
    uchar4 pixel = tex1Dfetch<uchar4>(bitmap, lookup);

    vector3 col_sky = vector3{
        (float)pixel.x,
        (float)pixel.y,
        (float)pixel.z };

    vector3 col_bg = col_sky * 0.0039215686f; //col_sky / 255.f
    col_bg = sRGBtoRGB(col_bg);

    vector3 col_bg_and_obj = BlendColors(SKYDISKRATIO * col_bg, 1, colorRGB, colorAlpha);

    col_bg_and_obj = col_bg_and_obj * 0.8f;

    image[3 * gpu_offset + 0] = col_bg_and_obj.x;
    image[3 * gpu_offset + 1] = col_bg_and_obj.y;
    image[3 * gpu_offset + 2] = col_bg_and_obj.z;
}

__global__ void FinishRayTracing(const float* __restrict__ imageFloat, unsigned char* __restrict__ image)
{
    int gpu_x = threadIdx.x + blockIdx.x * blockDim.x;
    int gpu_y = threadIdx.y + blockIdx.y * blockDim.y;
    int gpu_offset = gpu_x + gpu_y * blockDim.x * gridDim.x;

    vector3 finish;

    finish.x = RGBtosRGB(imageFloat[3 * gpu_offset + 0]);
    finish.y = RGBtosRGB(imageFloat[3 * gpu_offset + 1]);
    finish.z = RGBtosRGB(imageFloat[3 * gpu_offset + 2]);

    finish = finish * 255.f;

    // invert RGB to BGR format (used in .bmp & .jpeg files)
    image[3 * gpu_offset + 0] = (unsigned char)Clip<float>(finish.z, 0.f, 255.f);
    image[3 * gpu_offset + 1] = (unsigned char)Clip<float>(finish.y, 0.f, 255.f);
    image[3 * gpu_offset + 2] = (unsigned char)Clip<float>(finish.x, 0.f, 255.f);
}

/* create camera */
__host__ Camera_t* CreateCamera(void)
{
    Camera_t* camera = NULL;
    camera = (Camera_t*)malloc(sizeof(Camera_t));
    return(camera);
}

/* create black hole */
__host__ BlackHole_t* CreateBlackHole(void)
{
    BlackHole_t* blackhole = NULL;
    blackhole = (BlackHole_t*)malloc(sizeof(BlackHole_t));
    return(blackhole);
}

__host__ void SetCameraParameters(Camera_t* camera, vector3 cameraPosition, float FoVtangent, vector3 lookAt, vector3 upVector)
{
    camera->cameraPosition = cameraPosition;
    camera->FoVtangent = FoVtangent;
    camera->lookAt = lookAt;
    camera->upVector = upVector;

    vector3 frontvec = (lookAt - cameraPosition);
    frontvec = VectorNormalizeHost(frontvec);

    //vector3 leftvec = VectorCross(upVector, frontvec);
    vector3 rightvec = VectorCross(frontvec, upVector); //rightvec
    rightvec = VectorNormalizeHost(rightvec);

    //vector3 nupvec = VectorCross(frontvec, leftvec);

    // camera view matrix is standard matrix generates 3-axis world using camera, and center of coordinate system
    camera->viewMatrix.a11 = rightvec.x;
    //camera->viewMatrix.a12 = nupvec.x;
    camera->viewMatrix.a12 = upVector.x;
    camera->viewMatrix.a13 = frontvec.x;

    camera->viewMatrix.a21 = rightvec.y;
    //camera->viewMatrix.a22 = nupvec.y;
    camera->viewMatrix.a22 = upVector.y;
    camera->viewMatrix.a23 = frontvec.y;

    camera->viewMatrix.a31 = rightvec.z;
    //camera->viewMatrix.a32 = nupvec.z;
    camera->viewMatrix.a32 = upVector.z;
    camera->viewMatrix.a33 = frontvec.z;

}

__host__ void SetBlackHole(BlackHole_t* bh, float diskIn, float diskOut)
{
    bh->diskInner = diskIn;
    bh->diskOuter = diskOut;
    bh->diskInnerSquared = diskIn * diskIn;
    bh->diskOuterSquared = diskOut * diskOut;
}

void CreateTextureBitmap(unsigned char* dev_bitmap, cudaTextureObject_t* textureBitmap)
{
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = dev_bitmap;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 8;
    resDesc.res.linear.desc.y = 8;
    resDesc.res.linear.desc.z = 8;
    resDesc.res.linear.desc.w = 8;
    resDesc.res.linear.sizeInBytes = DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) * 4 * sizeof(unsigned char);
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaAssert(cudaCreateTextureObject(textureBitmap, &resDesc, &texDesc, null));
}

void ProcessRayTracing(unsigned char* dev_image)
{
    float* dev_imageFloat;

    Camera_t* camera;
    BlackHole_t* blackHole;
    Camera_t* dev_camera;
    BlackHole_t* dev_blackHole;

    unsigned char* dev_bitmap;

    camera = CreateCamera();
    blackHole = CreateBlackHole();

    SetBlackHole(blackHole, DISK_IN, DISK_OUT);
    SetCameraParameters(camera, vector3{ 0.0f, -1.f, -20.f }, FOV_TANGENT, Vector3Zero(), vector3{ 0.2f, -1.f, 0.f });
    
#pragma region Device memory allocation
    cudaAssert(cudaMalloc((void**)&dev_bitmap, DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) * 4 * sizeof(unsigned char)));
    cudaAssert(cudaMalloc((void**)&dev_imageFloat, DIM_X * DIM_Y * 3 * sizeof(float)));
    cudaAssert(cudaMalloc((void**)&dev_blackHole, sizeof(BlackHole_t)));
    cudaAssert(cudaMalloc((void**)&dev_camera, sizeof(Camera_t)));
#pragma endregion

    dim3 grid(DIM_X / 16, DIM_Y / 16);
    dim3 threads(16, 16);
    dim3 gridBmp(DIM_X_N(BITMAP_MULTIPIER) / 16, DIM_Y_N(BITMAP_MULTIPIER) / 16);
    dim3 grid1Axis(DIM_X / 16);
    dim3 threads1Axis(16);
    dim3 gridStar(DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) / STAR_DENSITY / 32);

    cudaAssert(cudaMemcpy(dev_blackHole, blackHole, sizeof(BlackHole_t), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(dev_camera, camera, sizeof(Camera_t), cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpyToSymbol(colorMap, colors, 500 * sizeof(vector3)));

    CreateGaussKernelRC();

    // prepare very dark sky
    cudaAssert(cudaMemset(dev_bitmap, 0x00, DIM_X_N(BITMAP_MULTIPIER) * DIM_Y_N(BITMAP_MULTIPIER) * sizeof(float)));
    PrepareSkyBackground(dev_bitmap);

    // copy bitmap to texture memory
    cudaTextureObject_t textureBitmap = 0;
    CreateTextureBitmap(dev_bitmap, &textureBitmap);

    // render black hole with accretion disc 
    RayTrace << < grid, threads >> > (dev_imageFloat, textureBitmap, dev_camera, dev_blackHole);
    UseGaussianKernelRC(dev_imageFloat);
    FinishRayTracing << <grid, threads >> > (dev_imageFloat, dev_image);

    cudaAssert(cudaDestroyTextureObject(textureBitmap));

#pragma region Cleanup CUDA memory
    cudaAssert(cudaFree(dev_bitmap));
    cudaAssert(cudaFree(dev_blackHole));
    cudaAssert(cudaFree(dev_camera));
    cudaAssert(cudaFree(dev_imageFloat));
#pragma endregion
}