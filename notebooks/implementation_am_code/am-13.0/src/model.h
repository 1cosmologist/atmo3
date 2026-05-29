/***********************************************************
* Smithsonian Astrophysical Observatory
* Submillimeter Receiver Laboratory
* am
*
* model.h                           S. Paine rev. 2022 May 2
*
* Constants and declarations for model.c
************************************************************/

#ifndef AM_MODEL_H
#define AM_MODEL_H

#include "am_types.h"

/*
 * Bits in model_t.ifmode, which controls IF spectrum computation.
 */
enum {
    IFMODE_DSB = 0x1,
    IFMODE_USB = 0x2,
    IFMODE_LSB = 0x4
};

/*
 * Bits in model_t.PTmode, which controls how layer pressures and
 * temperatures are specified.  See the comment under "Layers" below.
 */
enum {
    PTMODE_P_NOT_SET = 0x1,
    PTMODE_P         = 0x2,
    PTMODE_DP        = 0x4,
    PTMODE_PBASE     = 0x8,
    PTMODE_T_NOT_SET = 0x10,
    PTMODE_T         = 0x20,
    PTMODE_TBASE     = 0x40
};

enum {
    PTMODE_NOT_SET     = (PTMODE_P_NOT_SET | PTMODE_T_NOT_SET),
    PTMODE_MIDPOINT    = (PTMODE_P | PTMODE_T),
    PTMODE_HYDROSTATIC = (PTMODE_DP | PTMODE_PBASE)
};

/*
 * Bits in model_t.geometry.
 */
enum {
    GEOMETRY_PLANE_PARALLEL         = 0x1,
    GEOMETRY_SPHERICAL              = 0x2,
    GEOMETRY_LIMB                   = 0x4,
    GEOMETRY_REFRACT_NONE           = 0x8,
    GEOMETRY_REFRACT_RADIO          = 0x10,
    GEOMETRY_REFRACT_OPTICAL        = 0x20,
    GEOMETRY_REFRACT_USER_DEFINED   = 0x40,
    GEOMETRY_POBS_USER_DEFINED      = 0x80,
    GEOMETRY_ZOBS_USER_DEFINED      = 0x100,
    GEOMETRY_PSOURCE_USER_DEFINED   = 0x200,
    GEOMETRY_ZSOURCE_USER_DEFINED   = 0x400,
    GEOMETRY_PTAN_USER_DEFINED      = 0x800,
    GEOMETRY_ZTAN_USER_DEFINED      = 0x1000,
    GEOMETRY_R0_USER_DEFINED        = 0x2000,
    GEOMETRY_Z0_USER_DEFINED        = 0x4000,
    GEOMETRY_ZA_USER_DEFINED        = 0x8000,
    GEOMETRY_SEC_ZA_USER_DEFINED    = 0x10000,
    GEOMETRY_REVERSE                = 0x20000,
    GEOMETRY_SOURCE_NEAR            = 0x40000
};

enum {
    GEOMETRY_DEFAULTS  =
        (GEOMETRY_PLANE_PARALLEL |
         GEOMETRY_REFRACT_NONE),
    GEOMETRY_DISPLAY_AIRMASS = 
        (GEOMETRY_SPHERICAL |
         GEOMETRY_LIMB |
         GEOMETRY_REFRACT_OPTICAL |
         GEOMETRY_REFRACT_RADIO),
    GEOMETRY_DISPLAY_REFRACTION = 
        (GEOMETRY_REFRACT_OPTICAL |
         GEOMETRY_REFRACT_RADIO),
    GEOMETRY_DISPLAY_ZA_BASE = 
        (GEOMETRY_SPHERICAL |
         GEOMETRY_LIMB |
         GEOMETRY_REFRACT_OPTICAL |
         GEOMETRY_REFRACT_RADIO),
    GEOMETRY_MODE_BITS =
        (GEOMETRY_PLANE_PARALLEL |
         GEOMETRY_SPHERICAL |
         GEOMETRY_LIMB),
    GEOMETRY_REFRACT_BITS =
        (GEOMETRY_REFRACT_NONE |
         GEOMETRY_REFRACT_RADIO |
         GEOMETRY_REFRACT_OPTICAL |
         GEOMETRY_REFRACT_USER_DEFINED),
    GEOMETRY_OBS_LEVEL_USER_DEFINED =
        (GEOMETRY_POBS_USER_DEFINED |
         GEOMETRY_ZOBS_USER_DEFINED),
    GEOMETRY_SOURCE_LEVEL_USER_DEFINED =
        (GEOMETRY_PSOURCE_USER_DEFINED |
         GEOMETRY_ZSOURCE_USER_DEFINED),
    GEOMETRY_TAN_LEVEL_USER_DEFINED =
        (GEOMETRY_PTAN_USER_DEFINED |
         GEOMETRY_ZTAN_USER_DEFINED),
    GEOMETRY_AT_LEAST_ONE_USER_DEFINED_LEVEL =
        (GEOMETRY_OBS_LEVEL_USER_DEFINED |
         GEOMETRY_SOURCE_LEVEL_USER_DEFINED |
         GEOMETRY_TAN_LEVEL_USER_DEFINED),
    GEOMETRY_AT_LEAST_ONE_USER_DEFINED_Z =
        (GEOMETRY_Z0_USER_DEFINED |
         GEOMETRY_ZOBS_USER_DEFINED |
         GEOMETRY_ZSOURCE_USER_DEFINED |
         GEOMETRY_ZTAN_USER_DEFINED)

};

int    compute_model(model_t*, model_t*);
int    setup_atmospheric_model(model_t*, model_t*);
double source_to_obs_geometric_distance(model_t*);
double source_to_obs_path_distance(model_t*);
double source_to_obs_projected_distance(model_t*);
double total_airmass(model_t*);
double total_refraction(model_t*);

#endif /* AM_MODEL_H */
