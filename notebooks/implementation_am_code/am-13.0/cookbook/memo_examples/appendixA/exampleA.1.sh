# Example A.1 - A line-by-line speed benchmark

am -b - <<- end
    f 0 GHz 1000 GHz 10 MHz
    tol ${TOL}
    layer
    P 30 mbar
    T 220 K
    lineshape ${LINESHAPE} o3
    col o3 1e19 cm^-2
end
