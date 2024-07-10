/*
 *  This file is part of Healpix_cxx.
 *
 *  Healpix_cxx is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  Healpix_cxx is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Healpix_cxx; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 *  For more information about HEALPix, see http://healpix.sourceforge.net
 */

/*
 *  Healpix_cxx is being developed at the Max-Planck-Institut fuer Astrophysik
 *  and financially supported by the Deutsches Zentrum fuer Luft- und Raumfahrt
 *  (DLR).
 */

/*
 *  Copyright (C) 2003-2013 Max-Planck-Society
 *  Author: Martin Reinecke
 */

#include "healpix_map.h"

using namespace std;

template<typename T> void Healpix_Map<T>::Import_degrade
  (const Healpix_Map<T> &orig, bool pessimistic)
  {
  planck_assert(nside_<orig.nside_,"Import_degrade: this is no degrade");
  int fact = orig.nside_/nside_;
  planck_assert (orig.nside_==nside_*fact,
    "the larger Nside must be a multiple of the smaller one");

  int minhits = pessimistic ? fact*fact : 1;
#pragma omp parallel
{
  int m;
#pragma omp for schedule (static)
  for (m=0; m<npix_; ++m)
    {
    int x,y,f;
    pix2xyf(m,x,y,f);
    int hits = 0;
    kahan_adder<double> adder;
    for (int j=fact*y; j<fact*(y+1); ++j)
      for (int i=fact*x; i<fact*(x+1); ++i)
        {
        int opix = orig.xyf2pix(i,j,f);
        if (!approx<double>(orig.map[opix],Healpix_undef))
          {
          ++hits;
          adder.add(orig.map[opix]);
          }
        }
    map[m] = T((hits<minhits) ? Healpix_undef : adder.result()/hits);
    }
}
  }

template void Healpix_Map<float>::Import_degrade
  (const Healpix_Map<float> &orig, bool pessimistic);
template void Healpix_Map<double>::Import_degrade
  (const Healpix_Map<double> &orig, bool pessimistic);

template<typename T> void Healpix_Map<T>::minmax (T &Min, T &Max) const
  {
  Min = T(1e30); Max = T(-1e30);
  for (int m=0; m<npix_; ++m)
    {
    T val = map[m];
    if (!approx<double>(val,Healpix_undef))
      {
      if (val>Max) Max=val;
      if (val<Min) Min=val;
      }
    }
  }

template void Healpix_Map<float>::minmax (float &Min, float &Max) const;
template void Healpix_Map<double>::minmax (double &Min, double &Max) const;
