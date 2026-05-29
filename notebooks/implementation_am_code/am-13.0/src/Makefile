# Makefile for am-13.0
# S. Paine rev. 2023 August 29


# The default target is a help message:

help:
	@echo ""
	@echo "Building (with gcc) and installing am:"
	@echo "  make am         # build parallel am (OpenMP, recommended)"
	@echo "  make serial     # build single-threaded am"
	@echo "  make install    # install to $(INSTALL_DIR) (as root, or via sudo)"
	@echo "  make clean      # clean up build files (required before changing"
	@echo "                  # the build target or compiler)"
	@echo ""
	@echo "Building with other compilers:"
	@echo "  make pgi-omp    # Nvidia (PGI) pgcc (aka Nvidia nvc)"
	@echo "  make pgi-serial"
	@echo "  make icc-omp    # Intel icc"
	@echo "  make icc-serial"
	@echo ""
	@echo "Use EXTRA_CFLAGS, EXTRA_LDFLAGS and EXTRA_LDLIBS to add compiler"
	@echo "and linker flags.  Some examples:"
	@echo ""
	@echo "  Generate optimization and vectorization reports:"
	@echo "    make gcc-omp 'EXTRA_CFLAGS = -fopt-info'"
	@echo "    make pgi-omp 'EXTRA_CFLAGS = -Minfo -Mneginfo'"
	@echo "    make icc-omp 'EXTRA_CFLAGS = -qopt-report'"
	@echo ""
	@echo "  Build with multi-arch support, useful on heterogeneous clusters:"
	@echo "    make pgi-omp 'EXTRA_CFLAGS = -tp x64'"
	@echo ""
	@echo "  Tune for 16K 4-way L1 cache, overriding default (32K 8-way):"
	@echo "    make gcc-omp 'EXTRA_CFLAGS = -DL1_CACHE_BYTES=0x4000 -DL1_CACHE_WAYS=4'"
	@echo ""

clean:
	rm -f $(OBJS) version.o am


# INSTALL_DIR is the installation directory used by the 'install'
# target.  /usr/local/bin is the appropriate directory on modern
# Unix-like systems that follow the Filesystem Hierarchy Standard.
#
# On macOS, the directory /usr/local/bin is listed in the file
# /etc/paths which is read by the path_helper utility that constructs
# the shell PATH environment variable.  However, /usr/local/bin may
# not actually exist on a fresh macOS installation. The 'install'
# target will create it if necessary.

INSTALL_DIR = /usr/local/bin/

install:
	install -v -d $(INSTALL_DIR) && install -v am $(INSTALL_DIR)

list-targets:
	@grep "^[a-zA-Z].*:$$" Makefile


# The crlf-source target creates a set of source files for use on Windows.

crlf-source:
	$(MAKE) clean;\
	DEST=$$(pwd)-crlf;\
	rm -rf $${DEST};\
	mkdir $${DEST};\
	for f in *; do sed "s/$$//" $$f > $${DEST}/$$f; done


# If invoked directly, the 'am' target builds am with the default
# options here.  Recursive invocations build this target with modified
# macros that override these defaults.  The 'omp' and 'serial' targets
# point to the respective default build targets for the OpenMP and
# serial builds.

CC      = $(COMPILER_GCC)
CFLAGS  = $(CFLAGS_GCC_OPENMP)  $(OPT_FLAGS_GCC) $(EXTRA_CFLAGS)
LDFLAGS = $(LDFLAGS_GCC_OPENMP) $(EXTRA_LDFLAGS)
LDLIBS  = $(LDLIBS_GCC_OPENMP)  $(EXTRA_LDLIBS)


# User-definable extra flags from the command line.  Some examples:
#
#   Detailed optimization and vectorization reporting:
#     make gcc-omp "EXTRA_CFLAGS = -fopt-info"
#     make pgi-omp "EXTRA_CFLAGS = -Minfo -Mneginfo"
#     make icc-omp "EXTRA_CFLAGS = -qopt-report"
#
#   Multi-architecture support, useful on heterogeneous clusters:
#     make pgi-omp "EXTRA_CFLAGS = -tp x64"

EXTRA_CFLAGS  =
EXTRA_LDFLAGS = 
EXTRA_LDLIBS  = 


am: $(OBJS) version.o
	$(CC) -o am $(OBJS) version.o $(LDLIBS) $(LDFLAGS)

omp: gcc-omp

serial: gcc-serial


# Build targets for the gcc compiler.  Tested with gcc 12.3
# (2023 July 18).

COMPILER_GCC       = gcc

CFLAGS_GCC         = -ansi -Wall -W -Wshadow -Wwrite-strings $(EXTRA_CFLAGS)
CFLAGS_GCC_OPENMP  = $(CFLAGS_GCC) -fopenmp

LDFLAGS_GCC        = $(EXTRA_LDFLAGS)
LDFLAGS_GCC_OPENMP = $(LDFLAGS_GCC) 

LDLIBS_GCC         = -lm $(EXTRA_LDLIBS)
LDLIBS_GCC_OPENMP  = $(LDLIBS_GCC) -lgomp

OPT_FLAGS_GCC      = -O3

DEBUG_FLAGS_GCC    = -g -O0

gcc-omp:
	$(MAKE) am \
		"CC      = $(COMPILER_GCC)" \
		"CFLAGS  = $(CFLAGS_GCC_OPENMP) $(OPT_FLAGS_GCC)" \
		"LDFLAGS = $(LDFLAGS_GCC_OPENMP)" \
		"LDLIBS  = $(LDLIBS_GCC_OPENMP)"

gcc-serial:
	$(MAKE) am \
		"CC      = $(COMPILER_GCC)" \
		"CFLAGS  = $(CFLAGS_GCC) $(OPT_FLAGS_GCC)" \
		"LDFLAGS = $(LDFLAGS_GCC)" \
		"LDLIBS  = $(LDLIBS_GCC)"

gcc-no-opt:
	$(MAKE) am \
		"CC      = $(COMPILER_GCC)" \
		"CFLAGS  = $(CFLAGS_GCC)" \
		"LDFLAGS = $(LDFLAGS_GCC)" \
		"LDLIBS  = $(LDLIBS_GCC)"

gcc-no-opt-debug:
	$(MAKE) am \
		"CC      = $(COMPILER_GCC)" \
		"CFLAGS  = $(CFLAGS_GCC) $(DEBUG_FLAGS_GCC)" \
		"LDFLAGS = $(LDFLAGS_GCC)" \
		"LDLIBS  = $(LDLIBS_GCC)"


# Build targets for the Nvidia (Portland Group) pgcc compiler.
# Tested with pgcc (aka nvc) 23.3-0 (2023 Jul 20).  Note that
# OpenMP nested parallelism support was removed in pgcc/nvc
# following the acquisiton of PGI by Nvidia, and has still not
# been restored as of this version.

COMPILER_PGI       = pgcc

CFLAGS_PGI         = $(EXTRA_CFLAGS)
CFLAGS_PGI_OPENMP  = $(CFLAGS_PGI) -mp

LDFLAGS_PGI        = $(EXTRA_LDFLAGS)
LDFLAGS_PGI_OPENMP = $(LDFLAGS_PGI) -mp=allcores

LDLIBS_PGI         = $(EXTRA_LDLIBS)
LDLIBS_PGI_OPENMP  = $(LDLIBS_PGI)

OPT_FLAGS_PGI      = -fast -O3 -Msafeptr

pgi-omp:
	$(MAKE) am \
		"CC      = $(COMPILER_PGI)" \
		"CFLAGS  = $(CFLAGS_PGI_OPENMP) $(OPT_FLAGS_PGI)" \
		"LDFLAGS = $(LDFLAGS_PGI_OPENMP)" \
		"LDLIBS  = $(LDLIBS_PGI_OPENMP)"

pgi-serial:
	$(MAKE) am \
		"CC      = $(COMPILER_PGI)" \
		"CFLAGS  = $(CFLAGS_PGI) $(OPT_FLAGS_PGI)" \
		"LDFLAGS = $(LDFLAGS_PGI)" \
		"LDLIBS  = $(LDLIBS_PGI)"


# Build targets for the Intel icc compiler. Tested with icc 2021.5.0,
# distributed with the OneAPI 2022.2 toolkit (2023 Jul 20)

COMPILER_ICC       = icc

CFLAGS_ICC         = $(EXTRA_CFLAGS)
CFLAGS_ICC_OPENMP  = $(CFLAGS_ICC) -qopenmp

LDFLAGS_ICC        = $(EXTRA_LDFLAGS)
LDFLAGS_ICC_OPENMP = $(LDFLAGS_ICC) -qopenmp

LDLIBS_ICC         = $(EXTRA_LDLIBS)
LDLIBS_ICC_OPENMP  = $(LDLIBS_ICC)

OPT_FLAGS_ICC      = -O3 -fno-alias -xHost

icc-omp:
	$(MAKE) am \
		"CC      = $(COMPILER_ICC)" \
		"CFLAGS  = $(CFLAGS_ICC_OPENMP) $(OPT_FLAGS_ICC)" \
		"LDFLAGS = $(LDFLAGS_ICC_OPENMP)" \
		"LDLIBS  = $(LDLIBS_ICC_OPENMP)"

icc-serial:
	$(MAKE) am \
		"CC      = $(COMPILER_ICC)" \
		"CFLAGS  = $(CFLAGS_ICC) $(OPT_FLAGS_ICC)" \
		"LDFLAGS = $(LDFLAGS_ICC)" \
		"LDLIBS  = $(LDLIBS_ICC)"



# hno3.o and o3.o are listed first among all the object files because
# they take longest to compile.  This saves time if make is invoked
# with the -j option to run parallel jobs.  Note that version.o is not
# included in this list, to avoid creating a circular dependency.

OBJS = hno3.o o3.o abscoeff.o am_alloc.o am_sysdep.o am_types.o\
 ch4.o ch3oh.o clo.o cia.o co.o co2.o column.o config.o dcache.o\
 doc.o errlog.o fileops.o fit.o hbr.o hcl.o hcn.o h2co.o hf.o\
 h2o.o h2o_continuum.o h2o_ice.o h2o_liquid.o h2o_psat.o h2o2.o\
 h2s.o ho2.o hocl.o ils.o interp.o jacobian.o kcache.o layer.o\
 linesum.o main.o mapping.o model.o mt_ckd.o nh3.o nscale.o n2o.o\
 no.o no2.o o.o o2.o ocs.o oh.o oneline.o output.o planck.o\
 rayleigh.o rt.o simplex.o so2.o specfunc.o spectra.o tags.o\
 transform.o units.o

am: $(OBJS) version.o

$(OBJS): am_types.h version.h

version.o: $(OBJS) version.h

abscoeff.o: abscoeff.h am_alloc.h cia.h dcache.h errlog.h\
 h2o_continuum.h h2o_ice.h h2o_liquid.h kcache.h linesum.h output.h\
 phys_const.h units.h ch4.h ch3oh.h co.h co2.h clo.h hbr.h hcn.h\
 hcl.h hf.h hno3.h h2co.h h2o.h h2o2.h ho2.h hocl.h h2s.h nh3.h\
 n2o.h no.h no2.h o.h o2.h o3.h ocs.h oh.h so2.h oneline.h

am_alloc.o: abscoeff.h am_alloc.h column.h errlog.h kcache.h layer.h\
 model.h molecules.h nscale.h output.h simplex.h units.h

am_sysdep.o: am_sysdep.h

am_types.o: abscoeff.h column.h fit.h ils.h layer.h model.h\
 molecules.h output.h phys_const.h units.h

cia.o: am_sysdep.h errlog.h math_const.h phys_const.h specfunc.h

column.o: abscoeff.h cia.h column.h errlog.h layer.h molecules.h\
 phys_const.h spectra.h units.h

config.o: abscoeff.h am_alloc.h column.h config.h errlog.h fileops.h\
 fit.h ils.h layer.h linesum.h mapping.h math_const.h model.h\
 nscale.h output.h phys_const.h simplex.h tags.h units.h

dcache.o: abscoeff.h dcache.h errlog.h fileops.h phys_const.h

doc.o: doc.h

errlog.o: errlog.h

fileops.o: am_sysdep.h fileops.h

fit.o: abscoeff.h am_alloc.h am_sysdep.h config.h dcache.h errlog.h\
 fileops.h fit.h jacobian.h kcache.h model.h nscale.h output.h\
 simplex.h spectra.h units.h

h2o_continuum.o: errlog.h h2o_continuum.h mt_ckd.h phys_const.h

h2o_ice.o: errlog.h h2o_ice.h math_const.h molecules.h phys_const.h\
 rayleigh.h

h2o_liquid.o: errlog.h h2o_liquid.h math_const.h molecules.h\
 phys_const.h rayleigh.h

h2o_psat.o: column.h errlog.h h2o_psat.h phys_const.h

ils.o: errlog.h ils.h math_const.h output.h phys_const.h specfunc.h\
 transform.h

jacobian.o: errlog.h jacobian.h mapping.h model.h output.h simplex.h

kcache.o: errlog.h kcache.h

layer.o: abscoeff.h am_alloc.h column.h errlog.h h2o_psat.h interp.h\
 layer.h mapping.h model.h nscale.h phys_const.h units.h

linesum.o: am_sysdep.h errlog.h linesum.h math_const.h phys_const.h\
 specfunc.h

main.o: am_alloc.h am_sysdep.h config.h dcache.h doc.h errlog.h fit.h\
 jacobian.h kcache.h model.h nscale.h output.h planck.h simplex.h\
 tags.h transform.h units.h

mapping.o: errlog.h mapping.h

model.o: am_sysdep.h column.h errlog.h ils.h interp.h layer.h\
 math_const.h model.h output.h phys_const.h rt.h spectra.h\
 units.h

mt_ckd.o: mt_ckd.h

nscale.o: nscale.h

output.o: abscoeff.h column.h dcache.h errlog.h fit.h ils.h kcache.h\
 layer.h linesum.h mapping.h model.h nscale.h output.h phys_const.h\
 simplex.h tags.h units.h

planck.o: am_sysdep.h phys_const.h planck.h specfunc.h

rt.o: am_sysdep.h column.h errlog.h layer.h model.h phys_const.h\
 planck.h specfunc.h spectra.h rt.h

rayleigh.o: errlog.h math_const.h phys_const.h rayleigh.h

simplex.o: errlog.h simplex.h

spectra.o: abscoeff.h column.h errlog.h math_const.h model.h output.h\
	phys_const.h planck.h specfunc.h transform.h units.h

transform.o: am_sysdep.h math_const.h transform.h

units.o: errlog.h units.h

ch4.o: ch4.h molecules.h

ch3oh: ch3oh.h molecules.h

clo.o: clo.h molecules.h

co.o: co.h molecules.h

co2.o: co2.h molecules.h

hbr.o: hbr.h molecules.h

hcl.o: hcl.h molecules.h

hcn.o: hcn.h molecules.h

hf.o: hf.h molecules.h

h2co.o: h2co.h molecules.h

hno3.o: hno3.h molecules.h

h2o.o: h2o.h molecules.h

h2o2.o: h2o2.h molecules.h

ho2.o: ho2.h molecules.h

hocl.o: hocl.h molecules.h

h2s.o: h2s.h molecules.h

nh3.o: nh3.h molecules.h

n2o.o: n2o.h molecules.h

no.o: no.h molecules.h

no2.o: no2.h molecules.h

o.o: o.h molecules.h

o2.o: o2.h molecules.h

o3.o: o3.h molecules.h

ocs.o: ocs.h molecules.h

oh.o: oh.h molecules.h

so2.o: so2.h molecules.h

oneline.o: oneline.h molecules.h
