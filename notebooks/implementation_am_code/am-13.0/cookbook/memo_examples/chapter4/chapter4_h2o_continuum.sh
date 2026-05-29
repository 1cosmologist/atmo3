# file chapter4_h2o_continuum.sh - generates h2o_self_continuum and
# h2o_air_continuum spectra for Fig. 4.4

for T in 260 296; do
am - <<- end > am_h2o_self_${T}K.out
    f 0 GHz 15000 GHz 50 GHz
    output f GHz  k cm^5
    layer
    P 1 atm
    T ${T} K
    col h2o_self_continuum 1e22 cm^-2
end
am - <<- end > am_h2o_air_${T}K.out
    f 0 GHz 15000 GHz 50 GHz
    output f GHz  k cm^5
    layer
    P 1 atm
    T ${T} K
    col h2o_air_continuum 1e22 cm^-2
end
done

