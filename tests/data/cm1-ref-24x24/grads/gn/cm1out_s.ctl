dset ^cm1out_00%y4_s.dat                                                                                                      
options template
title CM1 output, using version cm1r21.1; units of x,y,z are km; time is generic, see cm1out_metadata for actual times
undef  -999999.9
xdef     24 linear  -1.150000E+01   1.000000E+00
ydef     24 linear  -1.150000E+01   1.000000E+00
zdef     40 linear   2.500000E-01   5.000000E-01
tdef          2 linear 00:00Z01JAN0001 1YR
vars     30
rain              0 99 accumulated surface rainfall                                 (cm)        
prate             0 99 surface precipitation rate                                   (kg/m2/s)   
sws               0 99 max horiz wind speed at lowest model level                   (m/s)       
svs               0 99 max vert vorticity at lowest model level                     (1/s)       
sps               0 99 min pressure at lowest model level                           (Pa)        
srs               0 99 max qr at lowest model level                                 (kg/kg)     
sgs               0 99 max qg at lowest model level                                 (kg/kg)     
sus               0 99 max w at 5 km AGL                                            (m/s)       
shs               0 99 max integrated updraft helicity                              (m2/s2)     
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
