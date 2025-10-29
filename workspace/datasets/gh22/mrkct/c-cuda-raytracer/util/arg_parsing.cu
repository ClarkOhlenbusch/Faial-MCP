#include <stdlib.h>
#include <string.h>
#include <util/arg_parsing.h>

struct Args parse_args(int argc, char const* argv[])
{
    struct Args args = Args {
        .output_path = "",
        .image_width = 640,
        .image_height = 480,
        .frames = 1
    };

#define OPT_WITH_ARG(short, long) \
    ((strcmp(argv[i], short) == 0 || strcmp(argv[i], long) == 0) && i + 1 < argc)

    for (int i = 1; i < argc; i++) {
        if (OPT_WITH_ARG("-o", "--output")) {
            args.output_path = argv[++i];
        } else if (OPT_WITH_ARG("-w", "--width")) {
            args.image_width = parse_positive_integer_or_exit(argv[++i], "--width");
        } else if (OPT_WITH_ARG("-h", "--height")) {
            args.image_height = parse_positive_integer_or_exit(argv[++i], "--height");
        } else if (OPT_WITH_ARG("-f", "--frames")) {
            args.frames = parse_positive_integer_or_exit(argv[++i], "--frames");
        }
    }

    return args;
}

int parse_positive_integer_or_exit(char const* arg, char const*)
{
    // FIXME: Better error handling
    return atoi(arg);
}
