HDF Explorer readme


The contents of the CD distribution of HDF Explorer are

To install start Setup/Setup.exe. 
The CD directories are: 
      Setup          Setup program for HDF Explorer
      NCSA           HDF and HDF5 libraries, as supplied by NCSA.
      Data           Extra HDF, HDF5 netCDF data sample files, not included in the setup 
                      installation.
      DataProject    C code programs that generate HDF5 and netCDF files. A 
                      Microsoft Visual C++ project is provided 
        hdf5data.   generates several kinds of HDF5 data 
        hdf5images. reads realistic image ASCII  data and generates HDF5 8bit 
                     and 24bit images 
        modb2hdf5.  reads an ASCII file in the MODB (Mediterranean Oceanic Data 
                     Base) format and saves the data in HDF5 (modb.h5) 
        netcdfdata. reads an ASCII file with bathymetry of the North Atlantic, 
                     latitude and longitude data, and generates a netCDF file with coordinate 
                      variables (omex.nc). The spatial resolution of the grid is variable. 
                     HDF Explorer supports map generation with variable grid (dimensions) sizes. 


Copyright (C) 2000/2004 Space Research Software, Inc.
www.space-research.org

