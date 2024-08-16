"""
Compile C++ functions:
python3.10 setup.py build_ext --inplace clean

Launch the app with the following command:
streamlit run app.py
or
streamlit run app.py --server.enableXsrfProtection false
"""

import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
from obspy import read


sys.path.append("./lib")
from dispersion import phase_shift as phase_shift_cpp
from dispersion import FK as FK_cpp
sys.path.append("./src")
from functions import phase_shift as phase_shift_py
from functions import FK as FK_py
from functions import stream_to_array, plot_wiggle, plot_spectrum, plot_FV, plot_FK, plot_geophones, extract_curve, lorentzian_error, invert_evodcinv, plot_inversion, direct, plot_dispersion_curves

import warnings
warnings.filterwarnings("ignore")



import plotly.graph_objects as go
import plotly.express as px



if 'picked' not in st.session_state:
    st.session_state.picked = False
    
if "clicked_pick" not in st.session_state:
    st.session_state.clicked_pick = False
    
if 'layers' not in st.session_state:
    st.session_state.layers = {}
    st.session_state.layers_nb = 0
    
def handle_picked():
    if st.session_state.event:
        selected_data = st.session_state.event.selection['lasso']
        if selected_data:
            x = selected_data[0]['x']
            y = selected_data[0]['y']
            poly_coords = np.array([x, y]).T
            f_picked, v_picked = extract_curve(FV, fs, vs, poly_coords, smooth=True)
            dc = lorentzian_error(v_picked, f_picked, st.session_state.dx, st.session_state.Nx, a=0.3)
            st.session_state.fs_picked = f_picked
            st.session_state.vs_picked = v_picked
            st.session_state.dc_picked = dc
    st.session_state.picked = True


def clicked_add_layer(thickness_min, thickness_max, vs_min, vs_max):
    st.session_state.layers_nb += 1
    st.session_state.layers[f'Layer {st.session_state.layers_nb}'] = {"thickness_min":thickness_min, "thickness_max":thickness_max, "vs_min":vs_min, "vs_max":vs_max}

    
def clicked_remove_layer():
    if st.session_state.layers_nb > 0:
        del st.session_state.layers[f'Layer {st.session_state.layers_nb}']
        st.session_state.layers_nb -= 1
    





st.header("Data file")
uploaded_file = st.file_uploader("# Import data file", type=["dat"])


if uploaded_file is not None:
    stream = read(uploaded_file, "seg2")
    
    dt = stream[0].stats.delta    
    n_channels = int(len(stream))
    n_samples = int(stream[0].stats.npts)
    source_position = float(stream[0].stats.seg2['SOURCE_LOCATION'])
    delay = float(stream[0].stats.seg2['DELAY'])
    geophone_positions = np.zeros(n_channels)
    for i in range(0, n_channels):
        geophone_positions[i] = float(stream[i].stats.seg2['RECEIVER_LOCATION']) 
    XT = stream_to_array(stream, n_channels, n_samples)
    
    st.caption("Header information")
    data = {
        'Time step [s]': [dt],
        'Number of time samples': [n_samples],
        'Delay [s]': [delay],
        'Number of traces': [n_channels],
        'Source position [m]': [source_position],
        'Trace positions [m]': [list(geophone_positions)]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True, use_container_width=True)
    
    
    
    
    st.divider()
    st.header("Shot-gather")
    nb_geophones = st.selectbox("Trace selection",
                                ["All",
                                "1/2",
                                "1/3",
                                "1/4",
                                "1/5",
                                "First half",
                                "Second half",
                                "First tier",
                                "Second tier",
                                "Third tier"],
                                )
        
    
    match nb_geophones:
        case "1/2":
            selected_geophones = [i for i in range(0, n_channels, 2)]
        case "1/3":
            selected_geophones = [i for i in range(0, n_channels, 3)]
        case "1/4":
            selected_geophones = [i for i in range(0, n_channels, 4)]
        case "1/5":
            selected_geophones = [i for i in range(0, n_channels, 5)]
        case "First half":
            selected_geophones = [i for i in range(n_channels // 2)]
        case "Second half":
            selected_geophones = [i for i in range(n_channels // 2, n_channels)]
        case "First tier":
            selected_geophones = [i for i in range(n_channels // 3)]
        case "Second tier":
            selected_geophones = [i for i in range(n_channels // 3, 2 * n_channels // 3)]
        case "Third tier":
            selected_geophones = [i for i in range(2 * n_channels // 3, n_channels)]
        case _:
            selected_geophones = [i for i in range(n_channels)]
            
    selected_geophone_positions = geophone_positions[selected_geophones]
    offsets = np.abs(selected_geophone_positions - source_position)
    
    st.session_state.dx = selected_geophone_positions[1] - selected_geophone_positions[0]
    st.session_state.Nx = len(selected_geophone_positions)
                                                        
    XT = XT[selected_geophones, :]
    
    fig_geophones = plot_geophones(selected_geophone_positions, geophone_positions, source_position)
    st.plotly_chart(fig_geophones)
    
    fig_stream = plot_wiggle(XT, selected_geophone_positions, dt, norm='trace')
    st.plotly_chart(fig_stream)
    
    fig_spectrum = plot_spectrum(XT, selected_geophone_positions, dt, norm='trace')
    st.plotly_chart(fig_spectrum)
    
        

    st.divider()
    st.header("Dispersion extraction")
    function = st.selectbox("Dispersion computing function",
                            ["Phase-Shift (C++)", "Phase-Shift (Python)", "FK (C++)", "FK (Python)"],
                            )
        
    if function in ["Phase-Shift (C++)", "Phase-Shift (Python)"]:
        f_min = st.number_input("Min frequency [Hz]", value=0.0)
        f_max = st.number_input("Max frequency [Hz]", value=200.0)
        v_min = st.number_input("Min velocity [m/s]", value=0.0)
        v_max = st.number_input("Max velocity [m/s]", value=1000.0)
        dv = st.number_input("Velocity step [m/s]", value=1.0)
        norm = st.selectbox("Normalization method", ["None", "Global", "Frequency"])
        function = function
        f_min = f_min
        f_max = f_max
        v_min = v_min
        v_max = v_max
        dv = dv
        norm = norm
        
    elif function in ["FK (C++)", "FK (Python)"]:
        f_min = st.number_input("Min frequency [Hz]", 0.0)
        f_max = st.number_input("Max frequency [Hz]", value=200.0)
        k_min = st.number_input("Min wavenumber [m^-1]", value=0.0)
        k_max = st.number_input("Max wavenumber [m^-1]", value=5.0)
        norm = st.selectbox("Normalization method", [None, "Global", "Wavenumber"])
        


    if function in ["Phase-Shift (C++)", "Phase-Shift (Python)"]:
        if function == "Phase-Shift (C++)":
            tic = time.time()
            (fs, vs, FV) = phase_shift_cpp(XT, dt, offsets, f_min, f_max, v_min, v_max, dv)
            tac = time.time()
            FV = np.array(FV)
            fs = np.array(fs)
            vs = np.array(vs)
        elif function == "Phase-Shift (Python)":
            tic = time.time()
            (fs, vs, FV) = phase_shift_py(XT, dt, offsets, f_min, f_max, v_min, v_max, dv)
            tac = time.time()
        
        
        if not st.session_state.clicked_pick:
            st.text(f"Elapsed time:  {tac - tic:4f} s")
            fig = plot_FV(FV, fs, vs, norm=norm)
            
            if st.session_state.picked:
                fig.add_trace(go.Line(x=st.session_state.fs_picked,
                                    y=st.session_state.vs_picked,
                                    mode='lines',
                                    name='Picked curve',
                                    line=dict(color='white', width=2),
                                    error_y=dict(type='data', array=st.session_state.dc_picked, visible=True, color='white', width=2, thickness=0.75),
                                    ))
            st.plotly_chart(fig)
            
            if st.session_state.picked:
                st.success("Success: Dispersion curve picked.")
                st.info("Info: Pick another curve to remove the previous one.")
            else:
                st.warning("Warning: Please click on 'Pick' pick a curve on the dispersion diagram.")
        
        
        elif st.session_state.clicked_pick:
            
            fs_tmp = np.copy(fs)
            vs_tmp = np.copy(vs)
            FV_tmp = np.copy(FV)
            if len(fs) > 1000:
                fs_tmp = fs[::10]
                FV_tmp = FV[::10]
            if len(vs) > 1000:
                vs_tmp = vs[::10]
                FV_tmp = FV[:,::10]
            f_grid, v_grid = np.meshgrid(fs_tmp, vs_tmp)
            f_grid = f_grid.flatten()
            v_grid = v_grid.flatten()
            
            fig = px.scatter(x=f_grid, y=v_grid, color=FV_tmp.T.flatten(),
                    labels=dict(x="Frequency [Hz]", y="Phase velocity [m/s]", color="Amplitude"),
                    title="Dispersion diagram",
                    color_continuous_scale='turbo',
                    hover_data=None,
                    )
                               
            event = st.plotly_chart(fig, selection_mode=["lasso"], on_select=handle_picked, key='event')
            
            st.info("Info: Please draw a zone with the lasso on the dispersion diagram to pick a curve.")
                        
            
        button = st.button("Pick", key="clicked_pick")
        st.divider()
        
        if st.session_state.picked:
            st.header("Inversion")
            
            inv_method = st.selectbox("Inversion method",
                            ["evodcinv"],
                            )
            
            
            if inv_method == "evodcinv":
                thickness_min = st.number_input("Min thickness [m]", value=0.5)
                thickness_max = st.number_input("Max thickness [m]", value=5.0)
                vs_min = st.number_input("Min S-wave velocity [m/s]", value=100.0)
                vs_max = st.number_input("Max S-wave velocity [m/s]", value=2000.0)
                
                button_add = st.button("Add layer")
                if button_add:
                    clicked_add_layer(thickness_min, thickness_max, vs_min, vs_max)
                
                button_remove = st.button("Remove layer")
                if button_remove:
                    clicked_remove_layer()
                    
                if st.session_state.layers:
                    st.write(st.session_state.layers)
                    
            
            if st.session_state.layers_nb > 0:
            
                button_invert = st.button("Invert")
                if button_invert:
                    model, misfit = invert_evodcinv(st.session_state.fs_picked, st.session_state.vs_picked, st.session_state.dc_picked, st.session_state.layers)
                    fig = plot_inversion(model)
                    st.plotly_chart(fig)
                    df = pd.DataFrame(model*1000, columns=["Thickness [m]", "P-wave velocity [m/s]", "S-wave velocity [m/s]", "Density [kg/m^3]"])
                    st.dataframe(df)
                    st.success("Success: Inversion completed.")
                    st.info(f"Info: Misfit = {misfit:.2f}")
                    
                    fs_inverted, vs_inverted = direct(model, st.session_state.fs_picked)
                    
                    fig, rmse, nrmse = plot_dispersion_curves(st.session_state.fs_picked, st.session_state.vs_picked, st.session_state.dc_picked, fs_inverted, vs_inverted)
                    st.plotly_chart(fig)
                    
                    st.info(f"Info: RMSE = {rmse:.2f} m/s | NRMSE = {nrmse:.2f} %")
                    
            else:
                st.warning("Warning: Please add at least one layer to perform the inversion.")
                    
            st.divider()


                    
                    
                    
                    
                    
                    
                    
                    
                    
                
    elif function in ["FK (C++)", "FK (Python)"]:
        if function == "FK (C++)":
            tic = time.time()
            (fs, ks, FK) = FK_cpp(XT, dt, offsets, f_min, f_max, k_min, k_max)
            FK = np.array(FK)
            fs = np.array(fs)
            ks = np.array(ks)
            tac = time.time()
        elif function == "FK (Python)":
            tic = time.time()
            (fs, ks, FK) = FK_py(XT, dt, offsets, f_min, f_max, k_min, k_max)
            tac = time.time()
        st.text(f"Elapsed time:  {tac - tic:4f} s")
        fig = plot_FK(FK, fs, ks, norm=norm)
        st.plotly_chart(fig)
        
        st.divider()