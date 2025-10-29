#include <sys/time.h>
#include <util/time_measurement.h>

long long current_time_in_milliseconds(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);

    return t.tv_usec * 1000;
}