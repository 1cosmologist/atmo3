# Example 4.1 - Shell script to compute mass absorption coefficient
#               spectra in m^2 kg-1 for liquid water and ice in the
#               Rayleigh limit.

for T in -60 -30 0; do
am - <<- end > am_iwp_${T}C.out
    f 0 GHz 1000 GHz 1 GHz
    output f GHz tau
    layer
    P 1 atm
    T ${T} C
    column iwp_abs_Rayleigh 1 kg*m^-2
end
done

for T in -30 0 30; do
am - <<- end > am_lwp_${T}C.out
    f 0 GHz 1000 GHz 1 GHz
    output f GHz tau
    layer
    P 1 atm
    T ${T} C
    column lwp_abs_Rayleigh 1 kg*m^-2
end
done
