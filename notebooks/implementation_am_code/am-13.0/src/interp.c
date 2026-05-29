/***********************************************************
* Smithsonian Astrophysical Observatory
* Submillimeter Receiver Laboratory
* am
*
* interp.c                       S. Paine rev. 2021 March 30
*
* Interpolation functions.
************************************************************/

#include <float.h>
#include <math.h>

#include "interp.h"

/***********************************************************
* double lin_interp(
*         const double x1,
*         const double y1,
*         const double x2,
*         const double y2,
*         const double x)
*
* Purpose:
*   Returns y such that the point (x, y) is linearly
*   interpolated between (x1, y1) and (x2, y2).
***********************************************************/

double lin_interp(
        const double x1,
        const double y1,
        const double x2,
        const double y2,
        const double x)
{
    double r = x2 - x1;

    if (fabs(r) < DBL_EPSILON)
        r = 1.0;
    else
        r = (x - x1) / r;
    return r * y2 + (1.0 - r) * y1;
}   /* lin_interp() */


/***********************************************************
* double log_x_interp(
*         const double x1,
*         const double y1,
*         const double x2,
*         const double y2,
*         const double x)
*
* Purpose:
*   Returns y such that the point (log(x), y) is linearly
*   interpolated between (log(x1), y1) and (log(x2), y2).
*
*   Returns y linearly interpolated between (x1, y2) and
*   (x2, y2) if any of x, x1, x2 are outside the domain of
*   log().
***********************************************************/

double log_x_interp(
        const double x1,
        const double y1,
        const double x2,
        const double y2,
        const double x)
{
    if (    x  < DBL_MIN || x  > DBL_MAX ||
            x1 < DBL_MIN || x1 > DBL_MAX ||
            x2 < DBL_MIN || x2 > DBL_MAX    )
        return lin_interp(x1, y1, x2, y2, x);
    else
        return lin_interp(log(x1), y1, log(x2), y2, log(x));
}   /* log_x_interp() */


/***********************************************************
* double log_y_interp(
*         const double x1,
*         const double y1,
*         const double x2,
*         const double y2,
*         const double x)
*
* Purpose:
*   Returns y such that the point (x, log(y)) is linearly
*   interpolated between (x1, log(y1)) and (x2, log(y2)).
*
*   Returns y linearly interpolated between (x1, y2) and
*   (x2, y2) if y1 or y2 are outside the domain of log().
***********************************************************/

double log_y_interp(
        const double x1,
        const double y1,
        const double x2,
        const double y2,
        const double x)
{
    if (    y1 < DBL_MIN || y1 > DBL_MAX ||
            y2 < DBL_MIN || y2 > DBL_MAX    )
        return lin_interp(x1, y1, x2, y2, x);
    else
        return exp(lin_interp(x1, log(y1), x2, log(y2), x));
}   /* log_y_interp() */
