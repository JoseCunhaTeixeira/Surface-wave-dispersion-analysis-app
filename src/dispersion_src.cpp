#include <fftw3.h>
#include <vector>
#include <stdio.h>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <complex>
#include <float.h>
#include <limits>
#include <iostream>




struct output {
    std::vector<double> xs, ys;
    std::vector<std::vector<double> > arr;
};


/**
 * Computes dispersion data using a standard frequency-wavenumber 
 * transformation (i.e., convert the data from x-t to f-k domain using a 
 * two-dimensional Fast Fourier Transformation).
 * (source : https://github.com/dpteague/PyMASWdisp/blob/master/dcprocessing.py)
 * 
 * @param XT shot-gather
 * @param dt sampling interval in seconds
 * @param offsets offsets in meter
 * @param f_min minimal frequency in Hz
 * @param f_max maximal freqeuncy in Hz
 * @param k_min mnimal phase velocity in Hz
 * @param k_max maximal phase velocity in m/s
*/
output FK_src(const std::vector<std::vector<double> >& XT,
                        double dt,
                        const std::vector<double>& offset,
                        double f_min,
                        double f_max,
                        double k_min,
                        double k_max) {

    // Check that spacing is uniform
    double dx = offset[1] - offset[0];
    for(size_t i = 1; i < offset.size() - 1; i++){
        if(offset[i + 1] - offset[i] != dx){
            throw std::invalid_argument("Receiver spacing must be uniform for FK transform");
        }
    }

    int n_channels = XT.size();
    int n_samples = XT[0].size();

    // Time
    double f_nyq = 1. / (2. * dt);
    double df = 1. / (n_samples * dt);
    std::vector<double> fs;
    for (float f = 0.0; f < f_nyq + df; f += df) {
        fs.push_back(f);
    }

    // Space
    double knyq = M_PI / dx;
    double dk = (2. * M_PI) / (n_channels * dx);
    std::vector<double> ks;
    for(double k = k_min; k < k_max + dk; k += dk){
        ks.push_back(k);
    }

    // Perform two-dimensional FFT
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_samples * n_channels);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n_samples * n_channels);

    for (int i = 0; i < n_channels; ++i) {
        for (int j = 0; j < n_samples; ++j) {
            in[i * n_samples + j][0] = XT[i][j];
            in[i * n_samples + j][1] = 0.0;
        }
    }
    
    fftw_plan plan;
    plan = fftw_plan_dft_2d(n_channels, n_samples, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    fftw_free(in);
     
    std::vector<std::vector<double> > KF(n_channels, std::vector<double>(n_samples));
    for (int i = 0; i < n_channels; ++i) {
        for (int j = 0; j < n_samples; ++j) {
            KF[i][j] = std::abs(std::complex<double>(out[i * n_samples + j][0], out[i * n_samples + j][1]));
        }
    }
    fftw_free(out);


    // Transpose the KF array
    std::vector<std::vector<double>> FK(n_samples, std::vector<double>(n_channels));
    for (int i = 0; i < n_channels; ++i) {
        for (int j = 0; j < n_samples; ++j) {
            FK[j][i] = KF[i][j];
        }
    }
    // Flip the FK array upside down
    std::reverse(FK.begin(), FK.end());

    // Trim the FK array to the desired frequency range
    f_max = std::min(f_max, f_nyq);
    int n_fs = fs.size();
    int i_min = 0;
    for (int i = 0; i < n_fs; ++i) {
        if (fs[i] >= f_min) {
            i_min = i;
            break;
        }
    }
    int i_max = 0;
    for (int i = 0; i < n_fs; ++i) {
        if (fs[i] >= f_max) {
            i_max = i;
            break;
        }
    }
    fs = std::vector<double>(fs.begin() + i_min, fs.begin() + i_max + 1);
    FK = std::vector<std::vector<double> >(FK.begin() + i_min, FK.begin() + i_max + 1);
    
    k_max = std::min(k_max, knyq);
    int n_ks = ks.size();
    i_min = 0;
    for (int i = 0; i < n_ks; ++i) {
        if (ks[i] >= k_min) {
            i_min = i;
            break;
        }
    }
    i_max = 0;
    for (int i = 0; i < n_ks; ++i) {
        if (ks[i] >= k_max) {
            i_max = i;
            break;
        }
    }
    ks = std::vector<double>(ks.begin() + i_min, ks.begin() + i_max + 1);
    for (size_t i = 0; i < fs.size(); ++i) {
        FK[i] = std::vector<double>(FK[i].begin() + i_min + 1, FK[i].begin() + i_max + 1 + 1);
    }

    output res;
    res.xs = fs;
    res.ys = ks;
    res.arr = FK;

    return res;
}


/**
 * @author J. Cunha Teixeira
 * @author B. Decker
 * 
 * Constructs a FV dispersion diagram using phase shift method
 * 
 * @param XT shot-gather
 * @param dt sampling interval in seconds
 * @param offsets offsets in meter
 * @param f_min minimal frequency in Hz
 * @param f_max maximal freqeuncy in Hz
 * @param v_min mnimal phase velocity in Hz
 * @param v_max maximal phase velocity in m/s
 * @param dv phase velocity step in m/s
*/
output phase_shift_src(const std::vector<std::vector<double> >& XT,
                                                        double dt,
                                                        const std::vector<double>& offsets,
                                                        double f_min,
                                                        double f_max,
                                                        double v_min,
                                                        double v_max,
                                                        double dv) {

    int Nt = XT[0].size();
    int Nx = XT.size();
    
    // Transformation de Fourier avec FFTW
    double* in = (double*)fftw_malloc(sizeof(double) * Nt);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nt);
    fftw_plan p = fftw_plan_dft_r2c_1d(Nt, in, out, FFTW_ESTIMATE);
    
    std::vector<std::vector<std::complex<double> > > XF(Nx, std::vector<std::complex<double> >(Nt));
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Nt; ++j) {
            in[j] = XT[i][j];
        }
        fftw_execute(p);
        for (int j = 0; j < Nt; ++j) {
            XF[i][j] = std::complex<double>(out[j][0], out[j][1]);
        }
    }
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    // Calcul de l'axe des fréquences
    std::vector<double> fs(Nt);
    for (int i = 0; i < Nt; ++i) {
        fs[i] = i / (dt * Nt);
    }
    
    int imax = 0;
    for (int i = 0; i < Nt; ++i) {
        if (fs[i] >= f_max) {
            imax = i;
            break;
        }
    }
    int imin = 0;
    for (int i = 0; i < Nt; ++i) {
        if (fs[i] >= f_min) {
            imin = i;
            break;
        }
    }
    fs = std::vector<double>(fs.begin() + imin, fs.begin() + imax + 1);
    for (int i = 0; i < Nx; ++i) {
        XF[i] = std::vector<std::complex<double> >(XF[i].begin() + imin, XF[i].begin() + imax + 1);
    }
    
    // Axe des vitesses
    std::vector<double> vs;
    for (double v = v_min; v <= v_max; v += dv) {
        vs.push_back(v);
    }

    std::vector<std::vector<double> > FV(fs.size(), std::vector<double>(vs.size(), 0.0));

    #pragma omp parallel for
        for (size_t i = 0; i < vs.size(); ++i) {
            double v = vs[i];
            for (size_t j = 0; j < fs.size(); ++j) {
                double f = fs[j];
                std::complex<double> sum_exp(0.0, 0.0);
                for (size_t k = 0; k < offsets.size(); ++k) {
                    double dphi = 2 * M_PI * offsets[k] * f / v;
                    std::complex<double> exp_term = std::polar(1., dphi) * (XF[k][j] / abs(XF[k][j]));
                    sum_exp += exp_term;
                }
                FV[j][i] = std::abs(sum_exp);
            }
        }

    output res;
    res.xs = fs;
    res.ys = vs;
    res.arr = FV;

    return res;
}
