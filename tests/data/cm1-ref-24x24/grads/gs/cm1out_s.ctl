dset ^cm1out_00%y4_s.dat                                                                                                      
options template
title CM1 output, using version cm1r21.1; units of x,y,z are km; time is generic, see cm1out_metadata for actual times
undef  -999999.9
xdef     24 linear  -1.150000E+01   1.000000E+00
ydef     24 linear  -1.150000E+01   1.000000E+00
zdef     40 levels 
    6.250000E-02
    1.875000E-01
    3.125000E-01
    4.375000E-01
    5.892857E-01
    7.946429E-01
    1.053571E+00
    1.366071E+00
    1.732143E+00
    2.151786E+00
    2.625000E+00
    3.125000E+00
    3.625000E+00
    4.125000E+00
    4.625000E+00
    5.125000E+00
    5.625000E+00
    6.125000E+00
    6.625000E+00
    7.125000E+00
    7.625000E+00
    8.125000E+00
    8.625000E+00
    9.125000E+00
    9.625000E+00
    1.012500E+01
    1.062500E+01
    1.112500E+01
    1.162500E+01
    1.212500E+01
    1.262500E+01
    1.312500E+01
    1.362500E+01
    1.412500E+01
    1.462500E+01
    1.512500E+01
    1.562500E+01
    1.612500E+01
    1.662500E+01
    1.712500E+01
tdef          2 linear 00:00Z01JAN0001 1YR
vars     38
rain              0 99 accumulated surface rainfall                                 (cm)        
prate             0 99 surface precipitation rate                                   (kg/m2/s)   
sws               0 99 max horiz wind speed at lowest model level                   (m/s)       
svs               0 99 max vert vorticity at lowest model level                     (1/s)       
sps               0 99 min pressure at lowest model level                           (Pa)        
srs               0 99 max qr at lowest model level                                 (kg/kg)     
sgs               0 99 max qg at lowest model level                                 (kg/kg)     
sus               0 99 max w at 5 km AGL                                            (m/s)       
shs               0 99 max integrated updraft helicity                              (m2/s2)     
rain2             0 99 translated surface rainfall                                  (cm)        
sws2              0 99 translated max horiz wspd at lowest model level              (m/s)       
svs2              0 99 translated max vert vort at lowest model level               (1/s)       
sps2              0 99 translated min pressure at lowest model level                (Pa)        
srs2              0 99 translated max qr at lowest model level                      (kg/kg)     
sgs2              0 99 translated max qg at lowest model level                      (kg/kg)     
sus2              0 99 translated max w at 5 km AGL                                 (m/s)       
shs2              0 99 translated max integrated updraft helicity                   (m2/s2)     
uh                0 99 integrated (2-5 km) AGL) updraft helicity                    (m2/s2)     
cref              0 99 composite reflectivity                                       (dBZ)       
th               40 99 potential temperature                                        (K)         
prs              40 99 pressure                                                     (Pa)        
qv               40 99 water vapor mixing ratio                                     (kg/kg)     
qc               40 99 qc                                                           (kg/kg)     
qr               40 99 qr                                                           (kg/kg)     
qi               40 99 qi                                                           (kg/kg)     
qs               40 99 qs                                                           (kg/kg)     
qg               40 99 qg                                                           (kg/kg)     
nci              40 99 nci                                                          (#/kg)      
ncs              40 99 ncs                                                          (#/kg)      
ncr              40 99 ncr                                                          (#/kg)      
ncg              40 99 ncg                                                          (#/kg)      
dbz              40 99 reflectivity                                                 (dBZ)       
uinterp          40 99 u interpolated to scalar points (grid-relative)              (m/s)       
vinterp          40 99 v interpolated to scalar points (grid-relative)              (m/s)       
winterp          40 99 w interpolated to scalar points                              (m/s)       
xvort            40 99 horizontal vorticity (x)                                     (1/s)       
yvort            40 99 horizontal vorticity (y)                                     (1/s)       
zvort            40 99 vertical vorticity                                           (1/s)       
endvars
