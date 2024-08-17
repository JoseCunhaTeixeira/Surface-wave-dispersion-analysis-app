"""
Launch the app with the following command:
streamlit run app.py
or
streamlit run app.py --server.enableXsrfProtection false
"""

import numpy as np
import pandas as pd
import streamlit as st
from obspy import read

from functions import phase_shift
from functions import FK
from functions import stream_to_array, plot_wiggle, plot_spectrum, plot_disp, plot_geophones, extract_curve, lorentzian_error, invert_evodcinv, plot_inversion, direct, plot_dispersion_curves

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
    
if 'clicked_compute' not in st.session_state:
    st.session_state.clicked_compute = False
    
if 'type' not in st.session_state:
    st.session_state.type = None

if 'clicked_load' not in st.session_state:
    st.session_state.clicked_load = False
    
if 'selected_geophones' not in st.session_state:
    st.session_state.selected_geophones = None
    
if 'clicked_invert' not in st.session_state:
    st.session_state.clicked_invert = False
    
def handle_picked():
    if st.session_state.event:
        selected_data = st.session_state.event.selection['lasso']
        if selected_data:
            x = selected_data[0]['x']
            y = selected_data[0]['y']
            poly_coords = np.array([x, y]).T
            f_picked, v_picked = extract_curve(st.session_state.FV, st.session_state.fs, st.session_state.vs, poly_coords, smooth=True)
            dc = lorentzian_error(v_picked, f_picked, st.session_state.dx, st.session_state.Nx, a=0.3)
            st.session_state.fs_picked = f_picked
            st.session_state.vs_picked = v_picked
            st.session_state.dc_picked = dc
    st.session_state.picked = True
    if st.session_state.layers:
        st.session_state.layers = {}
        st.session_state.layers_nb = 0


def clicked_add_layer(thickness_min, thickness_max, vs_min, vs_max):
    st.session_state.layers_nb += 1
    st.session_state.layers[f'Layer {st.session_state.layers_nb}'] = {"thickness_min":thickness_min, "thickness_max":thickness_max, "vs_min":vs_min, "vs_max":vs_max}
    if 'clicked_invert' in st.session_state:
        st.session_state.clicked_invert = False

    
def clicked_remove_layer():
    if st.session_state.layers_nb > 0:
        del st.session_state.layers[f'Layer {st.session_state.layers_nb}']
        st.session_state.layers_nb -= 1
    if 'clicked_invert' in st.session_state:
        st.session_state.clicked_invert = False
        
        
def handle_compute():
    st.session_state.clicked_compute = True
    if 'FV' in st.session_state:
        del st.session_state['FV']
    if 'fs' in st.session_state:
        del st.session_state['fs']
    if 'vs' in st.session_state:
        del st.session_state['vs']
    if st.session_state.picked:
        st.session_state.picked = False
    if st.session_state.layers:
        st.session_state.layers = {}
        st.session_state.layers_nb = 0
    if st.session_state.type :
        del st.session_state.type
        
def handle_load():
    st.session_state.clicked_load = True
    if "clicked_compute" in st.session_state:
        del st.session_state["clicked_compute"]
    if 'FV' in st.session_state:
        del st.session_state['FV']
    if 'fs' in st.session_state:
        del st.session_state['fs']
    if 'vs' in st.session_state:
        del st.session_state['vs']
    if st.session_state.picked:
        st.session_state.picked = False
    if st.session_state.layers:
        st.session_state.layers = {}
        st.session_state.layers_nb = 0
    if st.session_state.type :
        del st.session_state.type
        
def handle_invert():
    st.session_state.clicked_invert = True


### -----------------------------------------------------------------------------------------------

st.title("Surface-wave dispersion analysis and 1D invertion")

st.write("👋 📢 This app allows you to perform a surface-wave dispersion analysis and 1D inversion on a seismic record data file.")
st.write("📚 The dispersion analysis is based on the Phase-Shift and FK methods implemented by José Cunha Teixeira and Benjamin Decker, and the inversion is performed using the evodcinv method implemented by Keurfon Luu.")
st.write("For more information, you can visit the following URLs:")
st.write("https://https://github.com/JoseCunhaTeixeira")
st.write("https://github.com/LilDiabetX")
st.write("https://github.com/keurfonluu")
st.divider()

st.header("Data file")
uploaded_file = st.file_uploader("# Import data file", type=["DAT", "SU"], accept_multiple_files=False)


if uploaded_file is not None:
    stream = read(uploaded_file)
    
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
    
    
    button = st.button("Load", on_click=handle_load, key="click_load", type="primary")
    if not st.session_state.clicked_load:
        st.info("👆 Click on the 'Load' button to load the shot-gather.")
        st.warning("⚠️ No shot-gather loaded.")
    elif st.session_state.clicked_load:
        st.info("⬆️ You can select other traces and reload the shot-gather.")
        st.success("👌 Shot-gather loaded.")
    
    if st.session_state.click_load:    
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
        st.session_state.selected_geophones = selected_geophones
    
    else:
        selected_geophones = st.session_state.selected_geophones
    
    st.divider()
    
    
    
### -----------------------------------------------------------------------------------------------


        
    if st.session_state.clicked_load:
        
        st.header("Shot-gather")
                    
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
        
        
        
    ### -----------------------------------------------------------------------------------------------
        
        
        
        st.header("Dispersion computing")
        function = st.selectbox("Dispersion computing function",
                                ["Phase-Shift", "FK"],
                                )
            
        if function in ["Phase-Shift"]:
            f_min = st.number_input("Min frequency [Hz]", value=0, min_value=0, step=10)
            f_max = st.number_input("Max frequency [Hz]", value=200, min_value=f_min, step=10)
            v_min = st.number_input("Min velocity [m/s]", value=0, min_value=0, step=10)
            v_max = st.number_input("Max velocity [m/s]", value=1000, min_value=v_min, step=10)
            dv = st.number_input("Velocity step [m/s]", value=1.0, min_value=0.1, step=0.1)
            norm = st.selectbox("Normalization method", ["None", "Global", "Frequency"])
            function = function
            f_min = f_min
            f_max = f_max
            v_min = v_min
            v_max = v_max
            dv = dv
            norm = norm
            
        elif function in ["FK"]:
            f_min = st.number_input("Min frequency [Hz]", 0)
            f_max = st.number_input("Max frequency [Hz]", value=200)
            k_min = st.number_input("Min wavenumber [m^-1]", value=0)
            k_max = st.number_input("Max wavenumber [m^-1]", value=5)
            norm = st.selectbox("Normalization method", [None, "Global", "Wavenumber"])
        
        button = st.button("Compute", on_click=handle_compute, type="primary")
        if st.session_state.clicked_compute:
            st.info("⬆️ You can change the parameters and recompute the dispersion diagram.")
            st.success(f"👌 Dispersion diagram computed.")
        elif not st.session_state.clicked_compute:
            st.info("👆 Click on the 'Compute' button to compute the dispersion diagram.")
            st.warning("⚠️ No dispersion diagram computed.")
        
        st.divider()
        


    ### -----------------------------------------------------------------------------------------------



        if st.session_state.clicked_compute:
            st.header("Dispersion diagram")
                
            if function == "Phase-Shift":
                if 'FV' not in st.session_state:
                    (fs, vs, FV) = phase_shift(XT, dt, offsets, f_min, f_max, v_min, v_max, dv)
                    st.session_state.fs = fs
                    st.session_state.vs = vs
                    st.session_state.FV = FV
                    st.session_state.type = 'FV'
                elif 'FV' in st.session_state:
                    FV = st.session_state.FV
                    fs = st.session_state.fs
                    vs = st.session_state.vs
                
                
            elif function == "FK":
                if 'FV' not in st.session_state:
                    (fs, vs, FV) = FK(XT, dt, offsets, f_min, f_max, k_min, k_max)
                    st.session_state.fs = fs
                    st.session_state.vs = vs
                    st.session_state.FV = FV
                    st.session_state.type = 'FK'
                elif 'FV' not in st.session_state:
                    FV = st.session_state.FV
                    fs = st.session_state.fs
                    vs = st.session_state.vs
                

            if not st.session_state.clicked_pick:
                fig = plot_disp(st.session_state.FV, st.session_state.fs, st.session_state.vs, type=st.session_state.type, norm=norm)
                
                if st.session_state.picked:
                    fig.add_trace(go.Line(x=st.session_state.fs_picked,
                                        y=st.session_state.vs_picked,
                                        mode='lines',
                                        name='Picked curve',
                                        line=dict(color='white', width=2),
                                        error_y=dict(type='data', array=st.session_state.dc_picked, visible=True, color='white', width=2, thickness=0.75),
                                        ))
                    
                fig.update_layout(yaxis_range=[min(st.session_state.vs), max(st.session_state.vs)])
                st.plotly_chart(fig)
            
            
            elif st.session_state.clicked_pick and function == "Phase-Shift":
                
                fs_tmp = np.copy(st.session_state.fs)
                vs_tmp = np.copy(st.session_state.vs)
                FV_tmp = np.copy(st.session_state.FV)
                if len(st.session_state.fs) > 1000:
                    fs_tmp = st.session_state.fs[::10]
                    FV_tmp = st.session_state.FV[::10]
                if len(st.session_state.vs) > 1000:
                    vs_tmp = st.session_state.vs[::10]
                    FV_tmp = st.session_state.FV[:,::10]
                f_grid, v_grid = np.meshgrid(fs_tmp, vs_tmp)
                f_grid = f_grid.flatten()
                v_grid = v_grid.flatten()
                
                fig = px.scatter(x=f_grid, y=v_grid, color=FV_tmp.T.flatten(),
                        labels=dict(x="Frequency [Hz]", y="Phase velocity [m/s]", color="Amplitude"),
                        title=f"{st.session_state.type} dispersion diagram",
                        color_continuous_scale='turbo',
                        hover_data=None,
                        )
                                
                event = st.plotly_chart(fig, selection_mode=["lasso"], on_select=handle_picked, key='event')
                
                st.info("📿 Use the lasso to draw a zone on the dispersion diagram where to pick the curve.")
                st.info("👇 Or click on the button 'Cancel picking' to cancel the picking.")
                st.button("Cancel picking")
                
            if st.session_state.type == 'FV' and not st.session_state.clicked_pick:
                button = st.button("Start picking", key="clicked_pick", type="primary")
                if st.session_state.picked:
                    st.info("⬆️ You can pick another curve to replace the current one.")
                    st.success("👌 Dispersion curve picked.")
                elif not st.session_state.picked:
                    st.info("👆 Click on the button 'Start picking' to pick a dispersion curve.")
                    st.warning("⚠️ No picked dispersion curve.")
                
                
            st.divider()
            
            
            
    ### -----------------------------------------------------------------------------------------------



            if st.session_state.picked and not st.session_state.clicked_pick:
                st.header("Inversion parameters")
                
                inv_method = st.selectbox("Inversion method",
                                ["evodcinv"],
                                )
                
                if inv_method == "evodcinv":
                    thickness_min = st.number_input("Min thickness [m]", value=0.5, min_value=0.1, step=0.5)
                    thickness_max = st.number_input("Max thickness [m]", value=5.0, min_value=thickness_min, step=0.5)
                    vs_min = st.number_input("Min S-wave velocity [m/s]", value=10, min_value=1, step=10)
                    vs_max = st.number_input("Max S-wave velocity [m/s]", value=2000, min_value=vs_min, step=10)
                    runs = st.number_input("Number of runs", value=1, min_value=1)
                    
                    col1, col2 = st.columns([1,1])
                    
                    with col1:
                        button_add = st.button("Add layer")
                        if button_add:
                            clicked_add_layer(thickness_min, thickness_max, vs_min, vs_max)
                    with col2:
                        button_remove = st.button("Remove layer")
                        if button_remove:
                            clicked_remove_layer()
                        
                    if st.session_state.layers:
                        # st.write(st.session_state.layers)
                        df = pd.DataFrame(st.session_state.layers).T
                        st.dataframe(df)
                        
                        
                if st.session_state.layers_nb > 1:
                    
                    button_invert = st.button("Invert", key="click_invert", on_click=handle_invert, type="primary")
                    
                    if not st.session_state.clicked_invert:
                        st.info("👆 Click on the 'Invert' button to start the inversion.")
                    elif st.session_state.clicked_invert:
                        st.info("⬆️ You can change the parameters and recompute the inversion.")
                        st.success("👌 Inversion completed.")
                    
                    if st.session_state.click_invert:
                        model, misfit = invert_evodcinv(st.session_state.fs_picked, st.session_state.vs_picked, st.session_state.dc_picked, st.session_state.layers, runs)
                        st.session_state.model = model
                        st.session_state.misfit = misfit
                        
                    if st.session_state.clicked_invert:
                        st.divider()
                        st.header('Inversion results')
                        fig = plot_inversion(st.session_state.model)
                        st.plotly_chart(fig)
                        model = st.session_state.model
                        model[-1,0] = 'Infinity'
                        df = pd.DataFrame(st.session_state.model*1000, columns=["Thickness [m]", "P-wave velocity [m/s]", "S-wave velocity [m/s]", "Density [kg/m^3]"])
                        st.dataframe(df)
                        
                        fs_inverted, vs_inverted = direct(st.session_state.model, st.session_state.fs_picked)
                        
                        fig, rmse, nrmse = plot_dispersion_curves(st.session_state.fs_picked, st.session_state.vs_picked, st.session_state.dc_picked, fs_inverted, vs_inverted)
                        st.plotly_chart(fig)
                        
                        st.info(f"📋 RMSE = {rmse:.2f} m/s | NRMSE = {nrmse:.2f} %")
                        
                else:
                    st.warning("⚠️ Define at least two layers to perform the inversion.")
                        
                st.divider()
                
elif uploaded_file is None:
    st.info("👆 Please upload a seismic record data file.")
    if 'clicked_load' in st.session_state:
        del st.session_state['clicked_load']
    if 'selected_geophones' in st.session_state:
        del st.session_state['selected_geophones']
    if 'clicked_compute' in st.session_state:
        del st.session_state['clicked_compute']
    if 'type' in st.session_state:
        del st.session_state['type']
    if 'clicked_invert' in st.session_state:
        del st.session_state['clicked_invert']
    if 'clicked_pick' in st.session_state:
        del st.session_state['clicked_pick']
    if 'picked' in st.session_state:
        del st.session_state['picked']
    if 'layers' in st.session_state:
        del st.session_state['layers']
        del st.session_state['layers_nb']
    if 'fs' in st.session_state:
        del st.session_state['fs']
    if 'vs' in st.session_state:
        del st.session_state['vs']
    if 'FV' in st.session_state:
        del st.session_state['FV']
    if 'fs_picked' in st.session_state:
        del st.session_state['fs_picked']
    if 'vs_picked' in st.session_state:
        del st.session_state['vs_picked']
    if 'dc_picked' in st.session_state:
        del st.session_state['dc_picked']
    if 'model' in st.session_state:
        del st.session_state['model']
    if 'misfit' in st.session_state:
        del st.session_state['misfit']
    if 'clicked_invert' in st.session_state:
        del st.session_state['clicked_invert']
    if 'click_invert' in st.session_state:
        del st.session_state['click_invert']
    st.cache_data.clear()