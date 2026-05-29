# Radio and optical horizon distance for dry and moist
# US Standard atmosphere

rm -f radio_horiz_dry.out
rm -f radio_horiz_wet.out

# log sequence
for HEIGHT in $(awk '
        BEGIN {
            f = exp(log(10.) / 100.)
            for (h = 0.001; h <= 10; h *= f)
                printf("%e\n", h)
            }
    '); do

printf "%s " $HEIGHT >> radio_horiz_dry.out
am -a point2point.amc $HEIGHT km 0 km 90 deg   0% near radio \
    >> radio_horiz_dry.out

printf "%s " $HEIGHT >> radio_horiz_wet.out
am -a point2point.amc $HEIGHT km 0 km 90 deg 100% near radio \
    >> radio_horiz_wet.out 

done
