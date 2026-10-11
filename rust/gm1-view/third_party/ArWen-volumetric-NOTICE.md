# ArWen volumetric viewer source (Apache-2.0)

`src/arwen_wrfout.rs` adapts the classic-netCDF (CDF-1/CDF-2) header decoder
of the ArWen volumetric viewer source, published by FahrenheitResearch under
the Apache License, Version 2.0 (full text: the repository's `LICENSE`).
Archive identity: `arwen-volumetric-source.zip`, sha256
`fc7c53b73fcf9ec3a45ea90ee19b380045e7487779bbebaf1980592d3fb8bf84`.

Changes made for GM1 View (Apache-2.0 section 4(b)): the header is bounded
incrementally instead of read with the whole file, the file is hashed by
streaming, one contiguous variable record is scanned at a time, and
netCDF-4/HDF5 is refused explicitly. Nothing else of the archive is used.

NOTICE carries the attribution (Apache-2.0 section 4(c)): ArWen, Copyright
FahrenheitResearch, Apache License, Version 2.0.
