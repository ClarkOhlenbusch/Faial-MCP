#include <util/check_cuda_errors.h>
#include <util/framebuffer.h>
#include <util/stb_image_write.h>

struct Framebuffer alloc_framebuffer(int width, int height)
{
    struct Framebuffer fb;
    fb.byte_size = width * height * 4; // RGBA
    fb.width = width;
    fb.height = height;

    checkCudaErrors(cudaMallocManaged(&fb.data, fb.byte_size));
    checkCudaErrors(cudaGetLastError());

    return fb;
}

void write_framebuffer_to_file(struct Framebuffer fb, char const* filename)
{
    stbi_write_png(
        filename,
        fb.width,
        fb.height,
        4, // 4 bytes per pixel, RGBA
        fb.data,
        fb.width * 4);
}
