dset ^cm1out_00%y4_w.dat                                                                                                      
options template
title CM1 output, using version cm1r21.1; units of x,y,z are km; time is generic, see cm1out_metadata for actual times
undef  -999999.9
xdef     24 linear  -1.150000E+01   1.000000E+00
ydef     24 linear  -1.150000E+01   1.000000E+00
zdef     41 levels 
    0.000000E+00
    1.250000E-01
    2.500000E-01
    3.750000E-01
    5.000000E-01
    6.785715E-01
    9.107143E-01
    1.196429E+00
    1.535714E+00
    1.928572E+00
    2.375000E+00
    2.875000E+00
    3.375000E+00
    3.875000E+00
    4.375000E+00
    4.875000E+00
    5.375000E+00
    5.875000E+00
    6.375000E+00
    6.875000E+00
    7.375000E+00
    7.875000E+00
    8.375000E+00
    8.875000E+00
    9.375000E+00
    9.875000E+00
    1.037500E+01
    1.087500E+01
    1.137500E+01
    1.187500E+01
    1.237500E+01
    1.287500E+01
    1.337500E+01
    1.387500E+01
    1.437500E+01
    1.487500E+01
    1.537500E+01
    1.587500E+01
    1.637500E+01
    1.687500E+01
    1.737500E+01
tdef          2 linear 00:00Z01JAN0001 1YR
vars      6
w                41 99 vertical velocity                                            (m/s)       
tke              41 99 subgrid turbulence kinetic energy                            (m^2/s^2)   
kmh              41 99 horizontal eddy viscosity for momentum                       (m^2/s)     
kmv              41 99 vertical eddy viscosity for momentum                         (m^2/s)     
khh              41 99 horizontal eddy diffusivity for scalars                      (m^2/s)     
khv              41 99 vertical eddy diffusivity for scalars                        (m^2/s)     
endvars
