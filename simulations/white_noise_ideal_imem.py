import neuron
import LFPy
import numpy as np

import os
from os.path import join
from glob import glob
import brainsignals.neural_simulations as ns
import scipy.fftpack as ff

h = neuron.h

def return_ideal_cell(tstop, dt, apic_soma_diam = 20, apic_dend_diam=2, apic_upper_len = 1000, apic_bottom_len = -200):

    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1], dend[2]

    proc topol() { local i
      basic_shape()
      connect dend[0](0), soma(1)
      connect dend[1](0), soma(0)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -10., %s)
      pt3dadd(0, 0, 10., %s)}
      dend[0] {pt3dclear()
      pt3dadd(0, 0, 10., %s)
      pt3dadd(0, 0, %s, %s)}
      dend[1] {pt3dclear()
      pt3dadd(0, 0, -10., %s)
      pt3dadd(0, 0, %s, %s)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        dend[0] all.append()
        dend[1] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 1}
    dend[0] {nseg = 500}
    dend[1] {nseg = 500}
    }
    proc biophys() {
    }
    celldef()

    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        Ra = 150.
        cm = 1.
        }
    """ % (apic_soma_diam, apic_soma_diam, apic_dend_diam, apic_upper_len, apic_dend_diam, apic_dend_diam, apic_bottom_len, apic_dend_diam))
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -100.,
                'tstop': tstop,
                'pt3d': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
    return cell


def get_dipole_transformation_matrix(cell):
    '''
    Get linear response matrix

    Returns
    -------
    response_matrix: ndarray
        shape (3, n_seg) ndarray

    Raises
    ------
    AttributeError
        if ``cell is None``
    '''
    return np.stack([cell.x.mean(axis=-1),
                        cell.y.mean(axis=-1),
                        cell.z.mean(axis=-1)])


def make_white_noise_stimuli(cell, input_idx, freqs, tvec, input_scaling=0.005):

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

def check_existing_data(multipole_data, cell_name):
    if cell_name in multipole_data:
        if cell_name in multipole_data.keys():
            return True
    return False  

def run_white_noise_imem(tstop,
                         dt,
                         freqs,
                         freqs_limit, 
                         soma_diam, dend_diam, upper_len, bottom_len,
                         tvec,
                         t0_idx,
                         imem_data_filename='plot_imem_data',
                         directory='/Users/susannedahle/CellTypeDependenceElStim/sim_imem_data_ideal'
                        ):
    
    imem_data_filename = f'{imem_data_filename}.npy'
    imem_data_file_path = os.path.join(directory, imem_data_filename)
    
    # Initialize or load existing data
    if os.path.exists(imem_data_file_path):
        imem_data = np.load(imem_data_file_path, allow_pickle=True).item()
    else:
        imem_data = {}

    i = 0

    for bot_l in bottom_len:
        for up_l in upper_len:
            for s_d in soma_diam:
                for d_d in dend_diam:

                    if d_d > s_d:
                        continue

                    i += 1
                    cell_name = f'BL_{bot_l}_UL_{up_l}_SD_{s_d}_DD_{d_d}'

                    if check_existing_data(imem_data, cell_name):
                        print(f"Skipping {cell_name} (already exists in data)")
                        continue

                    cell = return_ideal_cell(tstop, dt,
                                             apic_soma_diam=s_d,
                                             apic_dend_diam=d_d,
                                             apic_upper_len=up_l,
                                             apic_bottom_len=bot_l)

                    print(f"Running wn simulation with {cell_name}", flush=True)

                    # White noise stimulus
                    cell, syn, noise_vec = make_white_noise_stimuli(
                        cell,
                        input_idx=0,
                        freqs=freqs[freqs < freqs_limit],
                        tvec=tvec
                    )

                    # Run simulation
                    cell.simulate(rec_imem=True, rec_vmem=True)

                    # Trim pre-t0
                    cell.vmem = cell.vmem[:, t0_idx:]
                    cell.imem = cell.imem[:, t0_idx:]
                    cell.tvec = cell.tvec[t0_idx:] - cell.tvec[t0_idx]

                    # Compute geometry-based metrics
                    closest_z_endpoint = min(abs(bot_l), abs(up_l))
                    distant_z_endpoint = max(abs(bot_l), abs(up_l))
                    total_len = abs(bot_l) + abs(up_l)
                    symmetry_factor = closest_z_endpoint / distant_z_endpoint

                    # Store imem amplitudes at 10, 100, 1000 Hz
                    imem_amplitudes_at_freqs = []
                    target_freqs = [5,10,50,100,500,1000]

                    for idx in range(cell.totnsegs):
                        imem_seg = cell.imem[idx, :]
                        freqs_imem, imem_amps = ns.return_freq_and_amplitude(cell.tvec, imem_seg)

                        # Extract amplitudes for the target frequencies
                        segment_amplitudes = []
                        for f in target_freqs:
                            freq_idx = np.argmin(np.abs(freqs_imem - f))
                            amplitude = imem_amps[0, freq_idx]
                            segment_amplitudes.append(amplitude)

                        imem_amplitudes_at_freqs.append(segment_amplitudes)
                    
                    # Calculate average return current positions for each target frequency
                    positive_avg_imem_pos = []
                    negative_avg_imem_pos = []
                    z_coords = cell.z.mean(axis=-1)
                    imem_amps_array = np.array(imem_amplitudes_at_freqs)

                    for f_idx in range(len(target_freqs)):
                        current_amps = imem_amps_array[:, f_idx]

                        # Positive z-direction
                        pos_indices = np.where(z_coords > 0)[0]
                        if len(pos_indices) > 0:
                            pos_z = z_coords[pos_indices]
                            pos_amps = current_amps[pos_indices]
                            sum_pos_amps = np.sum(pos_amps)
                            if sum_pos_amps > 0:
                                avg_pos_z = np.sum(pos_z * pos_amps) / sum_pos_amps
                                positive_avg_imem_pos.append(avg_pos_z)
                            else:
                                positive_avg_imem_pos.append(0)
                        else:
                            positive_avg_imem_pos.append(0)

                        # Negative z-direction
                        neg_indices = np.where(z_coords < 0)[0]
                        if len(neg_indices) > 0:
                            neg_z = z_coords[neg_indices]
                            neg_amps = current_amps[neg_indices]
                            sum_neg_amps = np.sum(neg_amps)
                            if sum_neg_amps > 0:
                                avg_neg_z = np.sum(neg_z * neg_amps) / sum_neg_amps
                                negative_avg_imem_pos.append(avg_neg_z)
                            else:
                                negative_avg_imem_pos.append(0)
                        else:
                            negative_avg_imem_pos.append(0)

                    # Store all data
                    imem_data[cell_name] = {
                        'freqs': target_freqs,
                        'x': cell.x.tolist(),
                        'z': cell.z.tolist(),
                        'totnsegs': cell.totnsegs,
                        'tvec': cell.tvec.tolist(),
                        'imem_amps': imem_amplitudes_at_freqs, 
                        'closest_z_endpoint': closest_z_endpoint,
                        'distant_z_endpoint': distant_z_endpoint,
                        'total_len': total_len,
                        'symmetry_factor': symmetry_factor,
                        'soma_diam': s_d,
                        'tot_dend_diam': 2 * d_d,
                        'positive_avg_imem_pos': positive_avg_imem_pos, # Added
                        'negative_avg_imem_pos': negative_avg_imem_pos
                    }

                    # Save to file
                    np.save(imem_data_file_path, imem_data)
                    print(f"Amplitude and PSD data saved to {os.path.abspath(imem_data_file_path)}")

                    del cell

                    print(f"Simulation complete for neuron {cell_name}: nr.{i}")

    print('All simulations complete')



if __name__=='__main__':
    upper_len_1 = np.array([1000])
    bottom_len_1 = np.array([-500])
    dend_diam_1 = np.array([2])
    soma_diam_1 = np.array([20])

    upper_len_2 = np.array([200])
    bottom_len_2 = np.array([-100])
    dend_diam_2 = np.array([2])
    soma_diam_2 = np.array([20])

    cut_off = 200
    tstop = 2**12 + cut_off
    dt = 2**-6

    rate = 5000 # * Hz
    freqs_limit = 10**4

    # Common setup
    num_tsteps = int(tstop / dt + 1)
    tvec = np.arange(num_tsteps) * dt
    t0_idx = np.argmin(np.abs(tvec - cut_off))

    sample_freq = ff.fftfreq(num_tsteps - t0_idx, d=dt / 1000)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    run_white_noise_imem(tstop, dt,freqs, freqs_limit, soma_diam_1, dend_diam_1, upper_len_1, bottom_len_1, tvec, t0_idx)
    run_white_noise_imem(tstop, dt,freqs, freqs_limit, soma_diam_2, dend_diam_2, upper_len_2, bottom_len_2, tvec, t0_idx)