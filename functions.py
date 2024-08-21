import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from matplotlib.path import Path
from scipy.fft import fft, fftfreq, fft2
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from evodcinv import EarthModel, Layer, Curve
from plotly.subplots import make_subplots
from disba import PhaseDispersion
from matplotlib.pyplot import cm




### -----------------------------------------------------------------------------------------------
def stream_to_array(stream, Nx, Nt):
    """
    Transform stream from obspy (Stream object) to numpy array in order to plot them
    """
    array = np.zeros((Nx,Nt))
    for i, trace in enumerate(stream):
        array[i,:] = trace.data
    return array
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def phase_shift(XT, dt, offsets, fmin, fmax, vmin, vmax, dv):
    """   Constructs a FV dispersion diagram
    args :
        XT (numpy array) : data
        dt (float) : sampling interval in seconds
        offsets (numpy array) : offsets from source in meters
        fmin (float) : minimum frequency computed
        fmax (float) : maximum frequency computed
        vmin, vmax (float) : velocities to scan in m/s
        dv (float) : velocity step in m/s
    returns :
        fs : frequency axis
        vs : velocity axis
        FV (numpy array) : data
    """
    if not isinstance(offsets, np.ndarray):
        offsets = np.array(offsets)
    
    Nt = XT.shape[1]
    XF = fft(XT, axis=(1), n=Nt)
    XF = XF[:Nt//2+1]
    
    fs = fftfreq(Nt, dt)
    fs = np.abs(fs[:Nt//2+1])
    imax = np.where(fs > fmax)[0][0]
    imin = np.where(fs >= fmin)[0][0]
    fs = fs[imin:imax]
    XF = XF[: , imin:imax]
    
    vs = np.arange(vmin, vmax+dv, dv)
    
    FV = np.zeros((len(fs), len(vs)))
    
    # Vecrorized version
    for v_i, v in enumerate(vs):
        dphi = 2 * np.pi * offsets[..., None] * fs / v
        FV[:, v_i] = np.abs(np.sum(XF/np.abs(XF)*np.exp(1j*dphi), axis=0))
        
    # Loop version
    # FV = np.zeros((len(fs), len(vs)))
    # for j, v in enumerate(vs):
    #     for i, f in enumerate(fs):   
    #         sum_exp = 0
    #         for k in range(len(offsets)):
    #             dphi = 2 * np.pi * offsets[k] * f / v
    #             exp_term = np.exp(1j * dphi) * (XF[k, i] / abs(XF[k, i]))
    #             sum_exp += exp_term
    #             FV[i, j] = abs(sum_exp)

    return fs, vs, FV
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def FK(XT, dt, offsets, fmin, fmax, kmin, kmax):
    """   Constructs a FK dispersion diagram
    args :
        XT (numpy array) : data
        dt (float) : sampling interval in seconds
        offsets (numpy array) : offsets in meter
        fmin (float) : minimum frequency computed
        fmax (float) : maximum frequency computed
        kmin, kmax (float) : velocities to scan in m/s
        dv (float) : velocity step in m/s
    returns :
        fs (numpy array) : frequency axis
        vs (numpy array) : velocity axis
        FK (numpy array) : FK diagram
    """
    dx = offsets[1] - offsets[0]
    for i in range(1, len(offsets)-1):
        if offsets[i+1] - offsets[i] != dx:
            raise ValueError("Offsets must be equally spaced for FK analysis")
    if len(offsets) != XT.shape[0]:
        raise ValueError("Offsets must have the same length as the number of geophones")
    
    Nt = XT.shape[1]
    Nx = len(offsets)
    
    KF = fft2(XT)
    KF = np.flipud(KF)
    FK = np.abs(KF.T)
        
    fs = fftfreq(Nt, dt)
    fs = np.abs(fs[:Nt//2+1])
    fnyq = 1 / (2. * dt)
    if fmax > fnyq:
        fmax = fnyq
    imax = np.where(fs >= fmax)[0][0]
    imin = np.where(fs >= fmin)[0][0]
    fs = fs[imin:imax+1]
    FK = FK[imin+1:imax+1+1, :]
    
    ks = fftfreq(Nx, dx)
    ks = np.abs(ks[:Nx//2+1])
    ks = 2 * np.pi * ks
    knyq = np.pi / dx
    if kmax > knyq:
        kmax = knyq
    imax = np.where(ks >= kmax)[0][0]
    imin = np.where(ks >= kmin)[0][0]
    ks = ks[imin:imax+1]
    FK = FK[:, imin:imax+1]
    
    return fs, ks, FK
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def plot_wiggle(XT, positions, dt, norm=None):    
    Nt = XT.shape[1]
    ts = np.arange(0, Nt*dt, dt)
    Nx = XT.shape[0]
    
    if norm == 'trace':
        for i in range(Nx):
            XT[i, :] = XT[i, :] / np.nanmax(XT[i, :])
    elif norm == 'global':
        XT = XT / np.nanmax(XT)
    
    fig = px.imshow(XT.T,
                    labels=dict(x="Position [m]", y="Time [s]", color="Amplitude"),
                    x=positions,
                    y=ts,
                    aspect='auto',
                    title="Shot-gather",
                    color_continuous_scale='RdBu_r',
                    zmin=-max(np.nanmin(XT), np.nanmax(XT)),
                    zmax=max(np.nanmin(XT), np.nanmax(XT)),
                    )
    
    fig.update_layout(xaxis=dict(side='top'))
    
    return fig
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def plot_spectrum(XT, positions, dt, norm='trace'):
    Nt = XT.shape[1]
    ts = np.arange(0, Nt*dt, dt)
    Nx = XT.shape[0]
    
    XF = fft(XT, axis=(1), n=Nt)
    XF = np.abs(XF[:, :Nt//2+1])
    fs = fftfreq(Nt, dt)
    fs = np.abs(fs[:Nt//2+1])
    
    if norm == 'trace':
        for i in range(Nx):
            XF[i, :] = XF[i, :] / np.nanmax(XF[i, :])
    elif norm == 'global':
        XF = XF / np.nanmax(XF)
        
    fig = px.imshow(XF.T,
                    labels=dict(x="Position [m]", y="Frequency [Hz]", color="Amplitude"),
                    x=positions,
                    y=fs,
                    aspect='auto',
                    title="Spectrum",
                    color_continuous_scale='gray_r',
                    )
    
    fig.update_layout(xaxis=dict(side='top'))
    
    return fig
### -----------------------------------------------------------------------------------------------




def plot_disp(arr, xs, ys, type, norm=None):
    if type == "FV":
        fig = plot_FV(arr, xs, ys, norm)
    elif type == "FK":
        fig = plot_FK(arr, xs, ys, norm)
    return fig



### -----------------------------------------------------------------------------------------------
def plot_FV(FV, fs, vs, norm=None):
    FV = np.copy(FV)
    if norm == "Axis 0":
        for i, f in enumerate(fs):
            FV[i, :] = FV[i, :] / np.nanmax(FV[i, :])
    elif norm == "Axis 1":
        for i, v in enumerate(vs):
            FV[:, i] = FV[:, i] / np.nanmax(FV[:, i])
    elif norm == 'Global':
        FV /= np.nanmax(FV) 
    
    fig = px.imshow(FV.T,
                    labels=dict(x="Frequency [Hz]", y="Phase velocity [m/s]", color="Amplitude"),
                    x=fs,
                    y=vs,
                    aspect='auto',
                    title="FV dispersion diagram",
                    color_continuous_scale='turbo',
                    origin='lower',
                    )
    return fig
### -----------------------------------------------------------------------------------------------
    
    
    

### -----------------------------------------------------------------------------------------------
def plot_FK(FK, fs, ks, norm=None):
    FK = np.copy(FK)
    if norm == "Axis 0":
        for i, f in enumerate(fs):
            FK[i, :] = FK[i, :] / np.nanmax(FK[i, :])
    elif norm == "Axis 1":
        for i, v in enumerate(ks):
            FK[:, i] = FK[:, i] / np.nanmax(FK[:, i])
    elif norm == 'Global':
        FK /= np.nanmax(FK)
            
    fig = px.imshow(FK.T,
                    labels=dict(x="Frequency [Hz]", y="Wavenumber [m^-1]", color="Amplitude"),
                    x=fs,
                    y=ks,
                    aspect='auto',
                    title="FK dispersion diagram",
                    color_continuous_scale='turbo',
                    origin='lower',
                    )
    
    return fig
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def plot_geophones(selected_geophone_positions, geophone_positions, source_position):    
    non_selected_geophones = [x for x in geophone_positions if x not in selected_geophone_positions]

    # Create a scatter plot
    fig = go.Figure()
    
    # Add scatter points for non-selected geophones in grey
    fig.add_trace(go.Scatter(
        x=non_selected_geophones,
        y=[0] * len(non_selected_geophones),
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='grey'),  # Grey color
        showlegend=False,
    ))
    
    # Add scatter points for selected geophones in blue
    fig.add_trace(go.Scatter(
        x=selected_geophone_positions,
        y=[0] * len(selected_geophone_positions),
        mode='markers',
        marker=dict(symbol='triangle-down', size=10, color='blue'),  # Blue color
        showlegend=False,
    ))
    
    # Add scatter points for source in red
    fig.add_trace(go.Scatter(
        x=[source_position],
        y=[0],
        mode='markers',
        marker=dict(symbol='star', size=10, color='red'),  # Blue color
        showlegend=False,
    ))

    # Update layout to hide the y-axis
    fig.update_layout(
        title="Geophones",
        xaxis_title="Position [m]",
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis_range=[-1, 1],  # Optional: Control the vertical space
        xaxis_range=[min(source_position, min(geophone_positions))-1, max(source_position, max(geophone_positions))+1],  # Optional: Add some padding around x points
        height=200,
    )

    return fig
### -----------------------------------------------------------------------------------------------  



### -----------------------------------------------------------------------------------------------
def extract_curve(FV, fs, vs, poly_coords, smooth):
    """
    Extracts f-v dispersion curve from f-v dispersion diagram by aiming maximums

    args :
        FV (2D numpy array) : dispersion diagram
        fs (1D numpy array) : frequency axis
        vs (1D numpy array) : velocity axis
        start (tuple of floats) : starting coordinates (f,v) values
        end (tuple of floats) : ending coordinates (f,v) values

    returns :
        curve (1D numpy array[velocity]) : f-v dispersion curve
    """

    df = fs[1] - fs[0]
    dv = vs[1] - vs[0]
    idx = np.zeros((len(poly_coords), 2), dtype=int)
    for i, (f,v) in enumerate(poly_coords):
        idx[i][0] = int(f/df)
        idx[i][1] = int(v/dv)

    poly_path = Path(idx)
    x,y = np.mgrid[:FV.shape[0], :FV.shape[1]]
    coors = np.hstack((x.reshape(-1, 1), y.reshape(-1,1)))

    mask = poly_path.contains_points(coors)
    mask = mask.reshape(FV.shape)

    FV_masked = FV * mask
    
    f_picked = []
    v_picked =[]

    f_start_i = np.min(idx[:, 0])
    f_end_i = np.max(idx[:, 0])
    v_start_i = np.min(idx[:, 1])
    v_end_i = np.max(idx[:, 1])

    FV_tmp = FV_masked[f_start_i:f_end_i, v_start_i+1:v_end_i]

    for i, FV_f in enumerate(FV_tmp): # FV_f is a vector of velocities for a frequency f
        v_max_i = np.where(FV_f == FV_f.max())[0][0]
        v_max = vs[v_max_i+v_start_i]
        if v_max_i+v_start_i == v_end_i-1 and i != 0:
            v_picked.append(v_picked[-1])
        else:
            v_picked.append(v_max)
        f_picked.append(fs[i+f_start_i+1])

    f_picked = np.array(f_picked)
    v_picked = np.array(v_picked)

    if smooth == True:
        if (len(v_picked)/2) % 2 == 0:
            wl = len(v_picked)/2 + 1
        else:
            wl = len(v_picked)/2
        v_picked_curve = savgol_filter(v_picked, window_length=wl, polyorder=4, mode="nearest")
    else :
        v_picked_curve = v_picked

    return f_picked, v_picked_curve
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def resamp(f, v, err, type="wavelength"):
    w = v / f
    func_v = interp1d(w, v)
    func_err = interp1d(w, err)

    if type == "wavelength":
        w_resamp = np.linspace(min(w), max(w), len(f))
    elif type == "wavelength-log":
        w_resamp = np.geomspace(min(w), max(w), len(f))
    
    v_resamp = func_v(w_resamp)
    err_resamp = func_err(w_resamp)
    f_resamp = v_resamp/w_resamp
    
    return f_resamp[::-1], v_resamp[::-1], err_resamp[::-1]
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def lorentzian_error(v_picked, f_picked, dx, Nx, a):
    # Resolution
    Dc_left = 1 / (1/v_picked - 1/(2*f_picked*Nx*dx))
    Dc_right = 1 / (1/v_picked + 1/(2*f_picked*Nx*dx))
    Dc = np.abs(Dc_left - Dc_right)
    
    # Absolute uncertainty
    dc = (10**-a) * Dc

    # for i, (err, v) in enumerate(zip(dc, v_picked)):
    #     if err > 0.6*v :
    #         dc[i] = 0.6*v

    return dc
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def invert_evodcinv(fs : np.array, vs: np.array, dcs : np.array, layers : dict, runs : int, iters : int, mode : int):
    # Initialize model
    model = EarthModel()
      
    for i in range(1, len(layers)+1):
        thickness_min = layers[f'Layer {i}']["thickness_min"] / 1000
        thickness_max = layers[f'Layer {i}']["thickness_max"] / 1000
        vs_min = layers[f'Layer {i}']["vs_min"] / 1000
        vs_max = layers[f'Layer {i}']["vs_max"] / 1000
        model.add(Layer([thickness_min, thickness_max], [vs_min, vs_max]))
        
    # Configure model
    model.configure(
        optimizer="cpso",  # Evolutionary algorithm
        misfit="rmse",  # Misfit function type
        optimizer_args={
            "popsize": 10,  # Population size
            "maxiter": iters,  # Number of iterations
            "workers": -1,  # Number of cores
            "seed": 0,
        },
    )

    ts = 1 / fs[::-1]
    vs_tmp =  vs[::-1]/1000
    curves = [Curve(ts, vs_tmp, mode, "rayleigh", "phase", uncertainties=dcs/1000)]

    # Run inversion
    res = model.invert(curves, runs)
    
    models = np.array(res.models)
    misfits = np.array(res.misfits)
    
    sorted_indices = np.argsort(misfits)
    models = models[sorted_indices]
    misfits = misfits[sorted_indices]
    
    return models, misfits
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def mean_model(models, misfits, fs, vs, dc, mode):
    sorted_indices = np.argsort(misfits)
    models = models[sorted_indices]
    misfits = misfits[sorted_indices]    
    
    cpt = 0
    curves = []
    frequencies = []
    models_tmp = []
    misfits_tmp = []
    for i in range(len(misfits)):
        try :
            f , curve = direct(models[i], fs, mode)
            cpt += 1
        except:
            continue
        curves.append(curve)
        frequencies.append(f)
        models_tmp.append(models[i])
        misfits_tmp.append(misfits[i])
                            
    models_in_range = []
    frequencies_in_range = []
    misfits_in_range = []
    curves_in_range = []
    for i, (model, f, misfit, curve) in enumerate(zip(models, frequencies, misfits, curves)):
        fs = np.round(fs,3)
        f = np.round(f,3)
        idx = [index for index, element in enumerate(fs) if element in f]

        is_within_range = np.all(np.abs(curve - vs[idx]) <= dc[idx])
        if is_within_range:
            models_in_range.append(model)
            frequencies_in_range.append(f)
            misfits_in_range.append(misfit)
            curves_in_range.append(curve)
            
    models_in_range = np.array(models_in_range)
    misfits_in_range = np.array(misfits_in_range)
    
    if models_in_range.size == 0:
        raise SystemError("No models inside the error-bars")
    
    
    
    
          
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Dispersion curves", "Vs models"))

    colormap = cm.get_cmap('viridis', len(models_in_range))
    
    max_depth = np.max(np.cumsum(models_in_range[:, :, 0], axis=1)) * 1000
    
    # Plot models as stairs    
    for i, model in enumerate(models_in_range[::-1]):
        model = model * 1000
        model[-1, 0] = max_depth - np.sum(model[:, 0])
        top_layer = np.copy(model[0, :])
        top_layer[0] = 0
        model = np.vstack((top_layer, model))
        
        color = colormap(i)
        color = f'rgb({colormap(i)[0]*255},{colormap(i)[1]*255},{colormap(i)[2]*255})'
        fig.add_trace(go.Scatter(x=model[:, 2],
                                 y=np.cumsum(model[:, 0]),
                                 name=f'msifit={misfits_in_range[::-1][i]:.2f}',
                                 mode='lines',
                                 line=dict(color=color, shape='hv')),
                                row=1, col=2)
    # Plot dispersion curves
    colormap = cm.get_cmap('viridis', len(curves_in_range))
    for i, curve in enumerate(curves_in_range[::-1]):
        color = colormap(i)
        color = f'rgb({colormap(i)[0]*255},{colormap(i)[1]*255},{colormap(i)[2]*255})'
        fig.add_trace(go.Scatter(x=frequencies_in_range[::-1][i],
                                 y=curve,
                                 mode='lines',
                                 name=f'msifit={misfits_in_range[::-1][i]:.2f}',
                                 line=dict(color=color)),
                                row=1, col=1)

    # Add measured dispersion curve with error bars
    fig.add_trace(go.Scatter(x=fs,
                             y=vs,
                             mode='markers',
                             error_y=dict(type='data', array=dc, visible=True, color='grey', width=2, thickness=0.75),
                             name='Measured',
                             marker=dict(color='grey')),
                            row=1, col=1)

    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Frequency [Hz]", row=1, col=1)
    fig.update_yaxes(title_text="Velocity [m/s]", row=1, col=1)
    fig.update_yaxes(title_text="Depth [m]", row=1, col=2, autorange="reversed")
    fig.update_xaxes(title_text="Shear-velocity [m/s]", row=1, col=2)
    
       
       
       
    
        
    # dx = 0.01/1000
    # models_in_range[:,-1, 0] = dx
    
    # max_len = int(np.max(np.sum(models_in_range[:, :, 0], axis=1))/dx)
    
    # # misfits_d = np.full((len(models_in_range), max_len), np.nan)

    # for model_i, model in enumerate(models_in_range):
        
    #     model_d = np.full((max_len, 4), np.nan)
    #     start = 0
        
    #     # misfits_d[model_i, :] = misfits_in_range[model_i]
        
    #     for layer_i, layer in enumerate(model):
    #         thick = layer[0]
    #         vp = layer[1]
    #         vs = layer[2]
    #         rho = layer[3]
            
    #         N = int(thick / dx)
    #         end = start + N
            
    #         model_d[start:end, :] = [dx, vp, vs, rho]
            
    #         start = end
                
    #     if model_i == 0:
    #         models_d = [model_d]
    #     else:
    #         models_d.append(model_d)

    # models_d = np.array(models_d)
    

    # # weights = 1/misfits_in_range
    # # weights = weights / np.sum(weights, axis=0)
    
    # mean_model = np.nanmedian(models_d, axis=0)
    # mean_misfit = np.nanmedian(misfits_in_range)
    # nb_models_in_range = len(models_in_range)
    
    # isnotnan = ~np.isnan(mean_model).any(axis=1)
    # mean_model = mean_model[isnotnan]





    # weights = 1/misfits_in_range
    # weights = weights / np.sum(weights, axis=0)
    
    mean_model = np.nanmedian(models_in_range, axis=0)
    mean_misfit = np.nanmean(misfits_in_range)
    nb_models_in_range = len(models_in_range)
    
    
    
        
    return mean_model, mean_misfit, nb_models_in_range, fig
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def plot_inversion(model):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Vp", "Vs", "Rho"))

    model = model * 1000
    model[-1, 0] = 1
    
    top_layer = np.copy(model[0, :])
    top_layer[0] = 0
        
    model = np.vstack((top_layer, model))
    
    fig.add_trace(go.Scatter(
        x=model[:, 1],  # vp values
        y=np.cumsum(model[:, 0]),
        mode='lines',
        name='Vp',
        line=dict(shape='hv'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=model[:, 2],  # vs values
        y=np.cumsum(model[:, 0]),
        mode='lines',
        name='Vs',
        line=dict(shape='hv'),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=model[:, 3],  # rho values
        y=np.cumsum(model[:, 0]),
        mode='lines',
        name='Rho',
        line=dict(shape='hv'),
    ), row=1, col=3)
    
    fig.update_yaxes(title_text="Depth [m]", row=1, col=1, autorange="reversed")
    fig.update_yaxes(row=1, col=2, autorange="reversed")
    fig.update_yaxes(row=1, col=3, autorange="reversed")
    fig.update_xaxes(title_text="Velocity [m/s]", row=1, col=1)
    fig.update_xaxes(title_text="Velocity [m/s]", row=1, col=2)
    fig.update_xaxes(title_text="Density [kg/m^3]", row=1, col=3)

    fig.update_layout(
        showlegend=False,
    )
    return fig
### -----------------------------------------------------------------------------------------------




### -----------------------------------------------------------------------------------------------
def direct(model, fs, mode):
    if fs[0] == 0:
        fs = fs[1:]
    t = 1 / fs[::-1]
    
    pd = PhaseDispersion(*model.T)
    cpr = [pd(t, mode=mode, wave="rayleigh")]
    
    fs = 1/cpr[0][0]
    vs = cpr[0][1]*1000
    
    fs = fs[::-1]
    vs = vs[::-1]
    
    return fs, vs
### -----------------------------------------------------------------------------------------------




### ----------------------------------------------------------------------------------------------- 
def plot_dispersion_curves(fs, vs, dc, fs_inv, vs_inv):
       
    fig = go.Figure()
    
    fig.add_trace(go.Scatter
    (
        x=fs,
        y=vs,
        mode='markers',
        name='Measured',
        marker=dict(symbol='circle', size=5, color='cornflowerblue'),
        showlegend=True,
        error_y=dict(type='data', array=dc, visible=True, color='cornflowerblue', width=2, thickness=0.75),
        )
    )

    fig.add_trace(go.Scatter
    (
        x=fs_inv,
        y=vs_inv,
        mode='markers',
        name='Inverted',
        marker=dict(symbol='circle', size=5, color='red'),
        showlegend=True,
        )
    )

    fig.update_layout(
        xaxis_title="Frequency [Hz]",
        yaxis_title="Velocity [m/s]",
        )
    
    fs = np.round(fs,3)
    fs_inv = np.round(fs_inv,3)
    idx = [index for index, element in enumerate(fs) if element in fs_inv]
    vs = vs[idx]
    rmse = np.sqrt(np.mean((vs - vs_inv)**2))
    nrmse = rmse / (vs.max() - vs.min()) * 100

    return fig, rmse, nrmse
### -----------------------------------------------------------------------------------------------