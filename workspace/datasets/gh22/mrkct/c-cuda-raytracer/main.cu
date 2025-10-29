#include <math.h>
#include <raytracer.h>
#include <scenes/test_scene.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <util/arg_parsing.h>
#include <util/framebuffer.h>
#include <util/time_measurement.h>
#include <util/vec3.h>

#define MAX_PATH_LENGTH 512

void get_frame_output_path(char const* directory, int frame_index, char* buffer)
{
    sprintf(buffer, "%s_%09d.png", directory, frame_index);
}

int main(int argc, char const* argv[])
{
    struct Args args = parse_args(argc, argv);

    printf("Starting render with\n"
           "\tSize: %d x %d\n"
           "\tFrames: %d\n"
           "\tOutput folder: %s\n",
        args.image_width, args.image_height, args.frames, args.output_path);

    struct Framebuffer fb = alloc_framebuffer(args.image_width, args.image_height);
    struct Scene scene = create_test_scene();

#define DEG2RAD(d) (d * M_PI / 180.f)

    char* output_path = (char*)malloc(sizeof(char) * MAX_PATH_LENGTH);
    long long combined_render_only_time = 0;

    long long total_time_start = current_time_in_milliseconds();
    for (int i = 0; i < args.frames; i++) {
        long long render_time_start = current_time_in_milliseconds();
        raytrace_scene(fb, scene,
            make_vec3(cosf(DEG2RAD(i)) * 3.0f, 1.5, sinf(DEG2RAD(i)) * 3.0f),
            make_vec3(0, 0, -1));
        long long render_time
            = current_time_in_milliseconds() - render_time_start;

        printf("Frame %d took\t%lld ms\n", i, render_time);
        combined_render_only_time += render_time;

        get_frame_output_path(args.output_path, i, output_path);
        printf("Writing frame %d to '%s'\n", i, output_path);
        write_framebuffer_to_file(fb, output_path);
    }

    auto total_time = current_time_in_milliseconds() - total_time_start;

    printf("Render finished\n"
           "\tTotal Time: %lld ms\n"
           "\tRender-only Time: %lld ms\n",
        total_time, combined_render_only_time);

    return 0;
}