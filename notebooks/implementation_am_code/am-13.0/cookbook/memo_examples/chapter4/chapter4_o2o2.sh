# file chapter4_o2o2.sh - generates o2o2 absorption coefficient spectrum for
# Fig 4.3

am - <<- end > am_o2o2_300K.out
    f 0 GHz 15000 GHz 50 GHz
    output f GHz  k cm^5
    layer
    P 1 atm
    T 300 K
    col o2o2 1e25 cm^-2
end
