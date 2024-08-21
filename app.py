"""
Compile C++ functions:
python3.10 setup.py build_ext --inplace clean

Launch the app with the following command:
streamlit run app.py
or
streamlit run app.py --server.enableXsrfProtection false
"""

import numpy as np
import pandas as pd
import streamlit as st
from obspy import read

from functions import FK, phase_shift
from functions import stream_to_array
from functions import plot_geophones, plot_wiggle, plot_spectrum, plot_disp, plot_dispersion_curves, plot_inversion
from functions import extract_curve, lorentzian_error
from functions import invert_evodcinv, mean_model, direct

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
    
if 'func' not in st.session_state:
    st.session_state.func = None
    
def handle_picked():
    if st.session_state.event:
        selected_data = st.session_state.event.selection['lasso']
        if selected_data:
            x = selected_data[0]['x']
            y = selected_data[0]['y']
            poly_coords = np.array([x, y]).T
            f_picked, v_picked = extract_curve(st.session_state.FV, st.session_state.fs, st.session_state.vs, poly_coords, smooth=True)
            dc = lorentzian_error(v_picked, f_picked, st.session_state.dx, st.session_state.Nx, a=0.5)
            st.session_state.fs_picked = f_picked
            st.session_state.vs_picked = v_picked
            st.session_state.dc_picked = dc
    st.session_state.picked = True
    if st.session_state.layers:
        st.session_state.layers = {}
        st.session_state.layers_nb = 0


def clicked_add_layer(thickness_min, thickness_max, vs_min, vs_max, vp_min=None, vp_max=None, rho_min=None, rho_max=None, func=None):    
    if func == 'Evodcinv':
        if st.session_state.layers_nb != 0 and "vp_min" in st.session_state.layers['Layer 1'].keys():
            st.session_state.layers = {}
            st.session_state.layers_nb = 0
            
    elif func == 'Dinver':
        if st.session_state.layers_nb != 0 and "vp_min" not in st.session_state.layers['Layer 1'].keys():
            st.session_state.layers = {}
            st.session_state.layers_nb = 0
    
    if st.session_state.func != func:
        st.session_state.func = func
    
            
    st.session_state.layers_nb += 1
    if func == 'Evodcinv':
        st.session_state.layers[f'Layer {st.session_state.layers_nb}'] = {"thickness_min":thickness_min, "thickness_max":thickness_max, "vs_min":vs_min, "vs_max":vs_max}
    elif func == 'Dinver':
        st.session_state.layers[f'Layer {st.session_state.layers_nb}'] = {"thickness_min":thickness_min, "thickness_max":thickness_max, "vs_min":vs_min, "vs_max":vs_max, "vp_min":vp_min, "vp_max":vp_max, "rho_min":rho_min, "rho_max":rho_max}
        
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
                                "First 1/2",
                                "Second 1/2",
                                "First 1/3",
                                "Second 1/3",
                                "Third 1/3",
                                "First 1/4",
                                "Last 1/4",
                                "First 1/5",
                                "Last 1/5",
                                ],
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
            case "First 1/2":
                selected_geophones = [i for i in range(n_channels // 2)]
            case "Second 1/2":
                selected_geophones = [i for i in range(n_channels // 2, n_channels)]
            case "First 1/3":
                selected_geophones = [i for i in range(n_channels // 3)]
            case "Second 1/3":
                selected_geophones = [i for i in range(n_channels // 3, 2 * n_channels // 3)]
            case "Third 1/3":
                selected_geophones = [i for i in range(2 * n_channels // 3, n_channels)]
            case "First 1/4":
                selected_geophones = [i for i in range(n_channels // 4)]
            case "Last 1/4":
                selected_geophones = [i for i in range(3 * n_channels // 4, n_channels)]
            case "First 1/5":
                selected_geophones = [i for i in range(n_channels // 5)]
            case "Last 1/5":
                selected_geophones = [i for i in range(4 * n_channels // 5, n_channels)]
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
            
            
        elif function in ["FK"]:
            f_min = st.number_input("Min frequency [Hz]", 0)
            f_max = st.number_input("Max frequency [Hz]", value=200)
            k_min = st.number_input("Min wavenumber [m^-1]", value=0)
            k_max = st.number_input("Max wavenumber [m^-1]", value=5)
        
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
                
            if function in ["Phase-Shift"]:
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
                
                
            elif function in ["FK"]:
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
                st.session_state.norm = st.selectbox("Normalization method", ["None", "Global", "Axis 0", "Axis 1"])
                fig = plot_disp(st.session_state.FV, st.session_state.fs, st.session_state.vs, type=st.session_state.type, norm=st.session_state.norm)
                
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
            
            
            elif st.session_state.clicked_pick and function in ["Phase-Shift"]:
                
                fs_tmp = np.copy(st.session_state.fs)
                vs_tmp = np.copy(st.session_state.vs)
                FV_tmp = np.copy(st.session_state.FV)
                
                if st.session_state.norm == "Axis 0":
                    for i, f in enumerate(fs_tmp):
                        FV_tmp[i, :] = FV_tmp[i, :] / np.nanmax(FV_tmp[i, :])
                elif st.session_state.norm == "Axis 1":
                    for i, v in enumerate(vs_tmp):
                        FV_tmp[:, i] = FV_tmp[:, i] / np.nanmax(FV_tmp[:, i])
                        
                if len(fs_tmp) > 1000:
                    fs_tmp = fs_tmp[::10]
                    FV_tmp = FV_tmp[::10]
                if len(vs_tmp) > 1000:
                    vs_tmp = vs_tmp[::10]
                    FV_tmp = FV_tmp[:,::10]
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
                                ["Evodcinv"],#, "Dinver"],
                                )
                
                if inv_method == "Evodcinv":
                    thickness_min = st.number_input("Min thickness [m]", value=0.5, min_value=0.1, step=0.5)
                    thickness_max = st.number_input("Max thickness [m]", value=5.0, min_value=thickness_min, step=0.5)
                    vs_min = st.number_input("Min S-wave velocity [m/s]", value=100, min_value=1, step=10)
                    vs_max = st.number_input("Max S-wave velocity [m/s]", value=1000, min_value=vs_min, step=10)
                    
                    col1, col2 = st.columns([1,1])
                    
                    with col1:
                        button_add = st.button("Add layer")
                        if button_add:
                            clicked_add_layer(thickness_min, thickness_max, vs_min, vs_max, func="Evodcinv")
                    with col2:
                        button_remove = st.button("Remove layer")
                        if button_remove:
                            clicked_remove_layer()
                        
                                               
                elif inv_method == "Dinver":
                    thickness_min = st.number_input("Min thickness [m]", value=0.5, min_value=0.1, step=0.5)
                    thickness_max = st.number_input("Max thickness [m]", value=5.0, min_value=thickness_min, step=0.5)
                    vs_min = st.number_input("Min S-wave velocity [m/s]", value=100, min_value=1, step=10)
                    vs_max = st.number_input("Max S-wave velocity [m/s]", value=1000, min_value=vs_min, step=10)
                    vp_min = st.number_input("Min P-wave velocity [m/s]", value=200, min_value=1, step=10)
                    vp_max = st.number_input("Max P-wave velocity [m/s]", value=2000, min_value=vp_min, step=10)
                    rho_min = st.number_input("Min density [kg/m^3]", value=1000, min_value=1, step=100)
                    rho_max = st.number_input("Max density [kg/m^3]", value=2000, min_value=rho_min, step=100)
                   
                    col1, col2 = st.columns([1,1])
                    
                    with col1:
                        button_add = st.button("Add layer")
                        if button_add:
                            clicked_add_layer(thickness_min, thickness_max, vs_min, vs_max, vp_min, vp_max, rho_min, rho_max, func='Dinver')
                    with col2:
                        button_remove = st.button("Remove layer")
                        if button_remove:
                            clicked_remove_layer()
                   
                   
                                           
                st.text('')
                st.text('')
                
                if st.session_state.layers_nb < 2:
                    if st.session_state.layers:
                        st.markdown(f"Model parameters for inversion with **{st.session_state.func}**")
                        df = pd.DataFrame(st.session_state.layers).T
                        st.dataframe(df)
                    st.text('')
                    st.text('')
                    st.warning("⚠️ Define at least two layers to be able to perform an inversion.")
                    st.divider()
                
                 
                if st.session_state.layers_nb >= 2:
                    if st.session_state.layers:
                        st.markdown(f"Model parameters for inversion with **{st.session_state.func}**")
                        df = pd.DataFrame(st.session_state.layers).T
                        st.dataframe(df)
                    st.text('')
                    st.text('')
                    st.success("👇 You can perform an inversion.")
                    st.divider()
                        
                    st.header("Inversion computation")
                    
                    mode = st.number_input("Mode to invert", value=0, min_value=0, step=1)
                    runs = st.number_input("Number of runs", value=1, min_value=1)
                    iters = st.number_input("Number of iterations", value=100, min_value=100, step=100)
                    button_invert = st.button("Invert", key="click_invert", on_click=handle_invert, type="primary")
                    
                    if not st.session_state.clicked_invert:
                        st.info("👆 Click on the 'Invert' button to launch the inversion.")
                    
                    if st.session_state.click_invert:
                        st.session_state.mode = mode
                        if st.session_state.func == 'Evodcinv':
                            models, misfits = invert_evodcinv(st.session_state.fs_picked, st.session_state.vs_picked, st.session_state.dc_picked, st.session_state.layers, runs, iters, st.session_state.mode)
                        elif st.session_state.func == 'Dinver':
                            pass
                            
                        avg_model, misfit, nb_models_in_range, fig = mean_model(models, misfits, st.session_state.fs_picked, st.session_state.vs_picked, st.session_state.dc_picked, st.session_state.mode)
                        st.session_state.misfit = misfit
                        st.session_state.best_model = models[0]
                        st.session_state.avg_model = avg_model
                        st.session_state.modes_fig = fig
                        st.session_state.nb_models_in_range = nb_models_in_range
                                               
                    if st.session_state.clicked_invert:
                        st.info("⬆️ You can change the inversion parameters and recompute an inversion.")
                        st.success("👌 Inversion completed.")

                        st.divider()
                        st.header('Inversion results')
                        
                        st.info(f"📋 {st.session_state.nb_models_in_range} generated models with dispersion curves inside the error-bars.")
                        st.plotly_chart(st.session_state.modes_fig)
                        
                        
                        st.text('')
                        st.text('')
                        st.text('')
                        st.text('')
                        
                        
                        st.subheader("Best model")
                        model = np.copy(st.session_state.best_model)
                        model[-1,0] = None
                        
                        df = pd.DataFrame(model*1000, columns=["Thickness [m]", "P-wave velocity [m/s]", "S-wave velocity [m/s]", "Density [kg/m^3]"])
                        st.dataframe(df)
                        
                        fig = plot_inversion(model)
                        st.plotly_chart(fig)
                        
                        fs_inverted, vs_inverted = direct(model, st.session_state.fs_picked, st.session_state.mode)
                        
                        fig, rmse, nrmse = plot_dispersion_curves(st.session_state.fs_picked, st.session_state.vs_picked, st.session_state.dc_picked, fs_inverted, vs_inverted)
                        st.plotly_chart(fig)
                        
                        st.info(f"📋 RMSE = {rmse:.2f} m/s | NRMSE = {nrmse:.2f} %")
                        
                        
                        st.text('')
                        st.text('')
                        st.text('')
                        st.text('')
                        
                        
                        st.subheader("**Median model**")
                        model = np.copy(st.session_state.avg_model)
                        model[-1,0] = None
                        
                        df = pd.DataFrame(model*1000, columns=["Thickness [m]", "P-wave velocity [m/s]", "S-wave velocity [m/s]", "Density [kg/m^3]"])
                        st.dataframe(df)
                        
                        fig = plot_inversion(model)
                        st.plotly_chart(fig)
                        
                        fs_inverted, vs_inverted = direct(model, st.session_state.fs_picked, st.session_state.mode)
                        
                        fig, rmse, nrmse = plot_dispersion_curves(st.session_state.fs_picked, st.session_state.vs_picked, st.session_state.dc_picked, fs_inverted, vs_inverted)
                        st.plotly_chart(fig)
                        
                        st.info(f"📋 RMSE = {rmse:.2f} m/s | NRMSE = {nrmse:.2f} %")
                        
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
    if 'nb_models_in_range' in st.session_state:
        del st.session_state['nb_models_in_range']
    if 'clicked_invert' in st.session_state:
        del st.session_state['clicked_invert']
    if 'click_invert' in st.session_state:
        del st.session_state['click_invert']
    st.cache_data.clear()