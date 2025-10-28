import os
# import sys # If job split
from os.path import join

from glob import glob
import numpy as np

import neuron
import LFPy
import brainsignals.neural_simulations as ns # From Hagen and Ness, ElectricBrainSignals doi: "https://doi.org/10.5281/zenodo.8255422"

import scipy.fftpack as ff

ns.load_mechs_from_folder(ns.cell_models_folder)
np.random.seed(1534)


def return_BBP_neuron(cell_name, tstop, dt):

    # load some required neuron-interface files
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    CWD = os.getcwd()
    cell_folder = join(join(bbp_folder, cell_name))
    if not os.path.isdir(cell_folder):
        ns.download_BBP_model(cell_name)

    neuron.load_mechanisms(bbp_mod_folder)
    os.chdir(cell_folder)
    add_synapses = False
    # get the template name
    f = open("template.hoc", 'r')
    templatename = ns.get_templatename(f)
    f.close()

    # get biophys template name
    f = open("biophysics.hoc", 'r')
    biophysics = ns.get_templatename(f)
    f.close()

    # get morphology template name
    f = open("morphology.hoc", 'r')
    morphology = ns.get_templatename(f)
    f.close()

    # get synapses template name
    f = open(ns.posixpth(os.path.join("synapses", "synapses.hoc")), 'r')
    synapses = ns.get_templatename(f)
    f.close()

    neuron.h.load_file('constants.hoc')

    if not hasattr(neuron.h, morphology):
        """Create the cell model"""
        # Load morphology
        neuron.h.load_file(1, "morphology.hoc")
    if not hasattr(neuron.h, biophysics):
        # Load biophysics
        neuron.h.load_file(1, "biophysics.hoc")
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, ns.posixpth(os.path.join('synapses', 'synapses.hoc')
                                       ))
    if not hasattr(neuron.h, templatename):
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    templatefile = ns.posixpth(os.path.join(cell_folder, 'template.hoc'))

    morphologyfile = glob(os.path.join('morphology', '*'))[0]


    # Instantiate the cell(s) using LFPy
    cell = LFPy.TemplateCell(morphology=morphologyfile,
                             templatefile=templatefile,
                             templatename=templatename,
                             templateargs=1 if add_synapses else 0,
                             tstop=tstop,
                             dt=dt,
                             lambda_f = 500,
                             nsegs_method='lambda_f',
                             v_init = -65)
    os.chdir(CWD)
    # set view as in most other examples
    cell.set_rotation(x=np.pi / 2)
    return cell


def get_dipole_transformation_matrix(cell): # From LFPy
        return np.stack([cell.x.mean(axis=-1),
                         cell.y.mean(axis=-1),
                         cell.z.mean(axis=-1)])


def make_white_noise_stimuli(cell, input_idx, freqs, tvec, input_scaling=0.005): # From Hagen and Ness, ElectricBrainSignals doi: "https://doi.org/10.5281/zenodo.8255422"

    I = np.zeros(len(tvec))

    for freq in freqs:
        I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())    
    input_array = input_scaling * I

    noise_vec = neuron.h.Vector(input_array)

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print("Input inserted in ", sec.name())
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.dt)
    return cell, syn, noise_vec 

def check_existing_data(cdm_data, cell_name):
    if cell_name in cdm_data:
        if cell_name in cdm_data.keys():
            return True
    return False  

def find_closest_indices(target_freqs, available_freqs):
    return [np.argmin(np.abs(available_freqs - tf)) for tf in target_freqs]

def run_white_noise_stim(freqs, 
                         neurons,
                         tvec,
                         t0_idx,
                         # job_nr, # if splitted jobs during sim
                         cdm_data_filename='cdm_and_imem_data_neo',
                         directory='/Users/susannedahle/CellTypeDependenceElStim/simulation_data',
                         plot_imem_filename = 'plot_imem',
                         ):
    
    # cdm_data_filename = f'{cdm_data_filename}_{job_nr}.npy' # If splitted jobs
    cdm_data_filename = f'{cdm_data_filename}.npy'
    cdm_data_file_path = os.path.join(directory, cdm_data_filename)
    failed_cells = []
    
    # Initialize or load existing data
    if os.path.exists(cdm_data_file_path):
        cdm_data = np.load(cdm_data_file_path, allow_pickle=True).item()
    else:
        cdm_data = {}
    
    # plot_imem_filename = f'{plot_imem_filename}_{job_nr}.npy' # If splitted jobs
    plot_imem_filename = f'{plot_imem_filename}.npy'
    plot_imem_file_path = os.path.join(directory, plot_imem_filename)

    # Initialize or load existing plot data
    if os.path.exists(plot_imem_file_path):
        plot_imem_data = np.load(plot_imem_file_path, allow_pickle=True).item()
    else:
        plot_imem_data = {}

    
    for neuron_idx, cell_name in enumerate(neurons):
        if check_existing_data(cdm_data, cell_name):
            print(f"Skipping {cell_name} (already exists in data)")
            continue

        try:
            cell = return_BBP_neuron(cell_name, tstop=tstop, dt=dt)

            # Insert noise
            cell, syn, noise_vec = make_white_noise_stimuli(cell, input_idx=0, freqs=freqs[freqs < 2200], tvec=tvec)
            ns.remove_active_mechanisms(remove_list, cell)

            # Run simulation
            cell.simulate(rec_imem=True, rec_vmem=True)

            # Cut initial segment
            cell.vmem = cell.vmem[:, t0_idx:]
            cell.imem = cell.imem[:, t0_idx:]
            cell.tvec = cell.tvec[t0_idx:] - cell.tvec[t0_idx]

            # Compute dipole moment (z-component)
            cdm = get_dipole_transformation_matrix(cell) @ cell.imem
            cdm = cdm[2, :] # 2: z-cordinate, : all timestep

            # Get frequency and amplitude of cdm
            freqs_s, amp_cdm_s = ns.return_freq_and_amplitude(cell.tvec, cdm)
            
            #Find amplitude of input currents
            input_current = np.array(noise_vec)
            input_current = input_current[t0_idx:len(cdm)+t0_idx]

            freqs_input, amp_input_current = ns.return_freq_and_amplitude(cell.tvec, input_current)
            cdm_per_input_current = amp_cdm_s / amp_input_current

            target_freqs = sorted(np.concatenate((np.arange(1, 10, 1), np.arange(10, 100, 10), np.arange(100, 2200, 100))))
            closest_indices = find_closest_indices(target_freqs, freqs_s)

            matched_freqs = freqs_s[closest_indices]
            matched_amp_cdm = amp_cdm_s[0, closest_indices]
            matched_cdm_per_input_current = cdm_per_input_current[0, closest_indices]

            # Distance from soma to closest endpoint                  
            upper_z_endpoint = cell.z.mean(axis=-1)[cell.get_closest_idx(z=10000)]
            bottom_z_endpoint = cell.z.mean(axis=-1)[cell.get_closest_idx(z=-10000)]
            closest_z_endpoint = min(upper_z_endpoint, abs(bottom_z_endpoint))
            distant_z_endpoint = max(upper_z_endpoint, abs(bottom_z_endpoint))
            total_len_z_direction = closest_z_endpoint + distant_z_endpoint
            symmetry_factor_z_direction = closest_z_endpoint/distant_z_endpoint
            
            # Soma diam
            soma_diam = cell.d[0]

            # Avg dend diam
            tot_z_diam_abs = 0
            numb_z_comp_abs = 0
            for idx in range(cell.totnsegs):
                dz = cell.z[idx,0] - cell.z[idx,1]
                dx = cell.x[idx,0] - cell.x[idx,1]
                dy = cell.y[idx,0] - cell.y[idx,1]

                if abs(dz) > abs(dx) and abs(dz) > abs(dy):
                    numb_z_comp_abs += 1
                    tot_z_diam_abs += cell.d[idx]
                    
            avg_z_diam = tot_z_diam_abs/numb_z_comp_abs

            # Imem amplitude and return position

            # Calculate imem amplitudes at each target frequency for each segment
            imem_amps_at_target_freqs = np.zeros((cell.totnsegs, len(matched_freqs)))

            # The frequencies from FFT will be the same for all segments, so get them once
            freqs_imem, _ = ns.return_freq_and_amplitude(cell.tvec, cell.imem[0, :])
            imem_freq_indices = find_closest_indices(matched_freqs, freqs_imem)

            for idx in range(cell.totnsegs):
                imem_seg = cell.imem[idx, :]
                _, imem_amps = ns.return_freq_and_amplitude(cell.tvec, imem_seg)
                imem_amps_at_target_freqs[idx, :] = imem_amps[0, imem_freq_indices]

            # Calculate frequency-dependent average return current positions
            avg_return_pos_above_soma_freq = []
            avg_return_pos_below_soma_freq = []

            z_coords = cell.z.mean(axis=-1)
            soma_z_pos = z_coords[0]  # Soma is at index 0

            # Get indices for segments above and below the soma once
            above_indices = np.where(z_coords > soma_z_pos)[0]
            below_indices = np.where(z_coords < soma_z_pos)[0]

            for f_idx in range(len(matched_freqs)):
                current_amps_at_freq = imem_amps_at_target_freqs[:, f_idx]

                # For currents above the soma
                if len(above_indices) > 0:
                    amps_above = current_amps_at_freq[above_indices]
                    pos_above = z_coords[above_indices]
                    total_amp_above = np.sum(amps_above)
                    if total_amp_above > 1e-12:  # Avoid division by zero
                        avg_pos = np.sum(pos_above * amps_above) / total_amp_above
                        avg_return_pos_above_soma_freq.append(avg_pos)
                    else:
                        avg_return_pos_above_soma_freq.append(0)
                else:
                    avg_return_pos_above_soma_freq.append(0)

                # For currents below the soma
                if len(below_indices) > 0:
                    amps_below = current_amps_at_freq[below_indices]
                    pos_below = z_coords[below_indices]
                    total_amp_below = np.sum(amps_below)
                    if total_amp_below > 1e-12: # Avoid division by zero
                        avg_pos = np.sum(pos_below * amps_below) / total_amp_below
                        avg_return_pos_below_soma_freq.append(avg_pos)
                    else:
                        avg_return_pos_below_soma_freq.append(0)
                else:
                    avg_return_pos_below_soma_freq.append(0)

            # Store data in dictionary
            cdm_data[cell_name] = {
                'frequency': matched_freqs,
                'cdm': matched_amp_cdm,
                'cdm_per_input_current': matched_cdm_per_input_current,
                'closest_z_endpoint': closest_z_endpoint,
                'distant_z_endpoint': distant_z_endpoint,
                'upper_z_endpoint': upper_z_endpoint,
                'bottom_z_endpoint': bottom_z_endpoint,
                'total_len': total_len_z_direction,
                'symmetry_factor': symmetry_factor_z_direction,
                'soma_diam': soma_diam,
                'avg_z_diam': avg_z_diam, 
                'avg_return_pos_above_soma': avg_return_pos_above_soma_freq,
                'avg_return_pos_below_soma': avg_return_pos_below_soma_freq
            }


            # Save amp data to .npy file
            np.save(cdm_data_file_path, cdm_data)
            print(f"Amplitude data has been saved to {os.path.abspath(cdm_data_file_path)}")

            try:
                # Store plot data of imem amplitudes at 10, 100, 1000 Hz
                plot_imem_amplitudes_at_freqs = []
                plot_freqs = [10, 100, 1000]

                for idx in range(cell.totnsegs):
                    imem_seg = cell.imem[idx, :]
                    freqs_imem, imem_amps = ns.return_freq_and_amplitude(cell.tvec, imem_seg)

                    # Extract amplitudes for the plot frequencies
                    segment_amplitudes = []
                    for f in plot_freqs:
                        freq_idx = np.argmin(np.abs(freqs_imem - f))
                        amplitude = imem_amps[0, freq_idx]
                        segment_amplitudes.append(amplitude)

                    plot_imem_amplitudes_at_freqs.append(segment_amplitudes)

                # Store in dictionary
                plot_imem_data[cell_name] = {
                    'freqs': plot_freqs,
                    'x': cell.x.tolist(),
                    'z': cell.z.tolist(),
                    'totnsegs': cell.totnsegs,
                    'tvec': cell.tvec.tolist(),
                    'imem_amps': plot_imem_amplitudes_at_freqs,  # Shape: (totnsegs, len(plot_freqs))
                }

                # Save plot data to .npy file
                np.save(plot_imem_file_path, plot_imem_data)
                print(f"Amplitude data has been saved to {os.path.abspath(plot_imem_file_path)}")
            except Exception as e_plot: 
                print(f'Cell failed to store plot data due to error {e_plot}')


            del cell, cdm, freqs_s, amp_cdm_s 
        
            print(f'Simulation complete for neuron {cell_name}: nr.{neuron_idx+1} of {len(neurons)}')

        except Exception as e:
            print(f"Skipping neuron {cell_name} due to error: {e}")
            failed_cells.append(cell_name)
            continue

    # Save failed cells
    if failed_cells:
        failed_path = os.path.join(directory, f"failed_cells_cdm.npy")
        np.save(failed_path, np.array(failed_cells))
        print(f"Saved list of failed cells to: {failed_path}")


if __name__=='__main__':
    ns.load_mechs_from_folder(ns.cell_models_folder)

    h = neuron.h

    all_cells_folder = '/Users/susannedahle/CellTypeDependenceElStim/simulations/all_cells_folder' # From the Blue Brain Project (Markram et al. 2015)
    bbp_folder = os.path.abspath(all_cells_folder)                              # Make this the bbp_folder

    cell_models_folder = '/Users/susannedahle/CellTypeDependenceElStim/simulations/brainsignals/cell_models'
    bbp_mod_folder = join(cell_models_folder, "bbp_mod")                        # Mappen med ulike parametere og mekanismer 

    # List to store the neuron names
    neurons = []

    # Check if the directory exists
    if os.path.exists(all_cells_folder):
        # Iterate over the directories in the all_cells_folder
        for folder_name in os.listdir(all_cells_folder):
            folder_path = os.path.join(all_cells_folder, folder_name)
            if os.path.isdir(folder_path):
                neurons.append(folder_name)
    else:
        print(f"The directory {all_cells_folder} does not exist.")

    remove_list = ["Ca_HVA", "Ca_LVAst", "Ca", "CaDynamics_E2", 
                   "Ih", "Im", "K_Pst", "K_Tst", "KdShu2007", "Nap_Et2",
                   "NaTa_t", "NaTs2_t", "SK_E2", "SKv3_1", "StochKv"]
    cut_off = 200
    tstop = 2**12 + cut_off
    dt = 2**-6

    rate = 5000 # * Hz

    # Common setup
    num_tsteps = int(tstop / dt + 1)
    tvec = np.arange(num_tsteps) * dt
    t0_idx = np.argmin(np.abs(tvec - cut_off))

    sample_freq = ff.fftfreq(num_tsteps - t0_idx, d=dt / 1000)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    cdm_amp_dict = {}  # To store amplitude spectra for each cell
    imem_amp_dict = {}

    # Simulation for the first neuron, full list of neurons computationally expencive, reccomend to split like shown below
    run_white_noise_stim(freqs, neurons[:1], tvec, t0_idx)

    ## To save time, reccomended to split jobs 
    ## Here splitted into 8 different jobs 
    # neurons.sort()
    # idx = int(sys.argv[1])
    # job_nr = idx
    
    # if idx == 0:
    #     neur_slice = neurons[:130]
    # elif idx == 1:
    #     neur_slice = neurons[130:260]
    # elif idx == 2:
    #     neur_slice = neurons[260:390]
    # elif idx == 3:
    #     neur_slice = neurons[390:520]
    # elif idx == 4:
    #     neur_slice = neurons[520:650]
    # elif idx == 5:
    #     neur_slice = neurons[650:780]
    # elif idx == 6:
    #     neur_slice = neurons[780:910]
    # else:
    #     neur_slice = neurons[910:]

    # run_white_noise_stim(freqs, neur_slice, tvec, t0_idx, job_nr)