# file chapter4_n2n2.sh - generates n2n2 absorption coefficient spectra for
# Figs. 4.1 and 4.2

for T in 93 126 149 179 212 228.3 253.5 277.5 297.5 322.6 343.0; do
am - <<- end > am_n2n2_${T}K.out
    f 0 GHz 15000 GHz 50 GHz
    output f GHz  k cm^5
    layer
    P 1 atm
    T ${T} K
    col n2n2 1e25 cm^-2
end
done
