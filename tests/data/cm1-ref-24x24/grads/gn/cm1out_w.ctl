dset ^cm1out_00%y4_w.dat                                                                                                      
options template
title CM1 output, using version cm1r21.1; units of x,y,z are km; time is generic, see cm1out_metadata for actual times
undef  -999999.9
xdef     24 linear  -1.150000E+01   1.000000E+00
ydef     24 linear  -1.150000E+01   1.000000E+00
zdef     41 linear   0.000000E+00   5.000000E-01
tdef          2 linear 00:00Z01JAN0001 1YR
vars      6
w                41 99 vertical velocity                                            (m/s)       
tke              41 99 subgrid turbulence kinetic energy                            (m^2/s^2)   
kmh              41 99 horizontal eddy viscosity for momentum                       (m^2/s)     
kmv              41 99 vertical eddy viscosity for momentum                         (m^2/s)     
khh              41 99 horizontal eddy diffusivity for scalars                      (m^2/s)     
khv              41 99 vertical eddy diffusivity for scalars                        (m^2/s)     
endvars
