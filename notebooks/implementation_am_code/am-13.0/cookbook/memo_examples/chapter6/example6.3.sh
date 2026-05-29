# Example 6.3 - Shell script to compute components of the wet excess
#               delay on Mauna Kea, in the low frequency limit
#
for COLTYPE in \
    h2o_lines \
    h2o_optical_refractivity \
    h2o_air_continuum \
    h2o_self_continuum \
    h2o \
; do
L=$(
am - <<- end 2>/dev/null
    f  0 THz  15 THz  25 MHz
    fout  0 THz  0 THz
    selfbroad_vmr_tol 0
    output L um
    layer
    P 625 mbar
    T 4 C
    h 1 m
    column $COLTYPE 1 um_pwv
end
)
echo $L $COLTYPE
done
