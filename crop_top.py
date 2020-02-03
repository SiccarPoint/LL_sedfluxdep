"""
Loads an existing landscape created as::

    sde = SedDepEroder(mg, K_sp=1.e-4, sed_dependency_type='almost_parabolic',
                       Qc='power_law', K_t=1.e-4)

Then perturbs it via reducing the gradient of the upper reaches.

"""
from landlab import CLOSED_BOUNDARY, RasterModelGrid
from landlab.components import FlowRouter, FastscapeEroder, SedDepEroder, \
    StreamPowerEroder, SteepnessFinder, LinearDiffuser
from landlab.plot import imshow_grid_at_node
from landlab.plot.channel_profile import channel_nodes, get_distances_upstream
from matplotlib.pyplot import show, figure, plot, close, savefig, loglog
from matplotlib.pyplot import xlabel, ylabel, ylim
import numpy as np
import time
import os
import datetime
from copy import deepcopy
from shutil import copyfile
from scipy.stats import linregress

allow_init_from_existing_file = True
attempt_extensions_of_existing_runs = True
# if True, will scour the dir for existing runs that only differ in terms
# of out_interval (must be a factor) and max_loops, then makes new runs that
# harvest existing data from these and extend the runtimes as specified.

uplift_rates = (0.00001, 0.0001)

bevel_amounts = (0.3, 0.8)

elev_fraction_to_bevel = (0.2, )

max_loops = 160000  # Try 10000 for K = 1.e-5, & adj out_interval (25 for 1e-5)

raster_params = {'shape': (50, 50), 'dx': 200.,
                 'initcond': 'initcondst5000.txt'}

inputs_sde = {'K_sp': 5.e-6, 'sed_dependency_type': 'almost_parabolic',
              'Qc': 'power_law', 'K_t': 5.e-6, 'forbid_deposition': True}  # , 'm_sp': 0.5, 'n_sp': 1.,
#               'm_t': 1.5, 'n_t': 1.}
inputs_ld = {'linear_diffusivity': 1.e-2}

dt = 200.  # was 100.
out_interval = 500

multiplierforstab = 3

color = 'gnuplot2'  # 'winter'

out_fields = [
        'topographic__elevation',
        'surface_water__discharge',
        'channel_sediment__relative_flux',
        'channel_sediment__volumetric_discharge',
        'channel_sediment__depth']

# build the structures:
mg = RasterModelGrid(raster_params['shape'], raster_params['dx'])
for edge in (mg.nodes_at_left_edge, mg.nodes_at_top_edge,
             mg.nodes_at_right_edge):
    mg.status_at_node[edge] = CLOSED_BOUNDARY

z = mg.add_field('node', 'topographic__elevation',
                 np.loadtxt(raster_params['initcond']))
sed = mg.add_zeros('node', 'channel_sediment__depth', dtype=float)

fr = FlowRouter(mg)
eroder = SedDepEroder(mg, **inputs_sde)
ld = LinearDiffuser(mg, **inputs_ld)


def build_master_dict(expt_ID):
    total_dict = inputs_sde.copy()
    total_dict.update(inputs_ld)
    total_dict.update(raster_params)
    total_dict['expt_ID'] = expt_ID
    total_dict['dt'] = dt
    total_dict['max_loops'] = max_loops
    total_dict['multiplierforstab'] = multiplierforstab
    total_dict['out_interval'] = out_interval
    total_dict['uplift_rates'] = uplift_rates
    total_dict['bevel_amounts'] = bevel_amounts
    total_dict['elev_fraction_to_bevel'] = elev_fraction_to_bevel
    return total_dict


def search_for_starting_file(dict_of_all_params, uplift_rate, directory='.',
                             enforce_equilibrium_time=False):
    """
    Returns a tuple of (path_to_existing_equilibrium_topo_file, run_time) or
    (None, None) if none exists. the file must have the same run conditions as
    this run. Exceptions are the run ID, the output interval, the chosen
    uplift rates and accel factors used in that run, and the forbid_deposition
    flag.
    If more than one exists, the one with the longest total run time will be
    selected.
    If enforce_equilibrium_time, the function also checks for consistency with
    the time-relevant params in the model runs (i.e., max_loops, dt,
    multiplierforstab).
    """
    paths_of_valid_starting_files = []
    equivalent_timestostab = []
    for root, dirs, files in os.walk(directory):
        files = [f for f in files if f.startswith('expt_ID_paramdict.npy')]
        if len(files) > 0:
            assert len(files) == 1
            # load the dict; check it's valid
            dicts_are_the_same = True
            found_params = np.load(root + '/expt_ID_paramdict.npy').item()
            for key in dict_of_all_params.keys():
                if key in ['expt_ID', 'out_interval', 'uplift_rates',
                           'accel_factors', 'forbid_deposition',
                           'bevel_amounts', 'elev_fraction_to_bevel']:
                    continue
                if key in ['dt', 'multiplierforstab', 'max_loops']:
                    if not enforce_equilibrium_time:  # ignore time to =brium
                        continue
                try:
                    if dict_of_all_params[key] != found_params[key]:
                        dicts_are_the_same = False
                except KeyError:
                    dicts_are_the_same = False
            if dicts_are_the_same:  # it is valid
                # now check the uplift rate exists:
                uplift_str = 'uplift_rate_' + str(uplift_rate)
                if uplift_str in os.listdir(root):
                    found_time_to_stab = (
                        found_params['multiplierforstab'] *
                        found_params['max_loops'] *
                        found_params['dt'])
                    if enforce_equilibrium_time:
                        time_to_stab = (
                            dict_of_all_params['multiplierforstab'] *
                            dict_of_all_params['max_loops'] *
                            dict_of_all_params['dt'])
                        filename = ('topographic__elevation_after' +
                                    str(time_to_stab) + 'y.txt')
                        if filename in os.listdir(root + '/' + uplift_str):
                            paths_of_valid_starting_files.append(
                                root + '/' + uplift_str + '/' + filename)
                            equivalent_timestostab.append(found_time_to_stab)
                    else:
                        poss_files = [
                            f for f in os.listdir(root + '/' + uplift_str) if
                            f.startswith('topographic__elevation_after')]
                        assert not len(poss_files) > 1
                        if len(poss_files) == 1:
                            paths_of_valid_starting_files.append(
                                root + '/' + uplift_str + '/' + poss_files[0])
                            equivalent_timestostab.append(found_time_to_stab)
    if len(paths_of_valid_starting_files) == 0:
        return (None, None)
    elif len(paths_of_valid_starting_files) == 1:
        print('Found an initial topo to use.')
        return (paths_of_valid_starting_files[0], equivalent_timestostab[0])
    else:
        print('Found an initial topo to use.')
        which_one_is_best = np.argmax(equivalent_timestostab)
        return (paths_of_valid_starting_files[which_one_is_best],
                equivalent_timestostab[which_one_is_best])


def make_expt_folder_and_store_master_dict():
    """
    Assumes the various dicts and input variables all already exist in the
    workspace.
    Builds the useful variables expt_ID & total_dict.
    """
    # make a master folder for the run:
    expt_ID = datetime.datetime.now().strftime('%y%m%d_%H.%M.%S')
    os.mkdir(expt_ID)
    total_dict = build_master_dict(expt_ID)
    np.save(expt_ID + '/expt_ID_paramdict.npy', total_dict)
    # load it again with params = np.load('expt_ID_paramdict.npy').item()
    # save a txt version for inspection too:
    _ = read_and_print_file_for_inputs(directory=expt_ID, print_file=True)
    return expt_ID, total_dict


def run_fresh_perturbations():
    """
    This is the main experimental function. Call it to run to equilibrium for
    various uplift rates (or load existing if flag is set), and then to perturb
    these equilibria by changing uplift rate by given fractions. These
    perturbation runs will always be "fresh".
    """
    global z, sed, allow_init_from_existing_file
    expt_ID, total_dict = make_expt_folder_and_store_master_dict()
    for uplift_rate in uplift_rates:
        time_to_stab = multiplierforstab * max_loops * dt
        path_to_data = expt_ID + '/uplift_rate_' + str(uplift_rate)
        os.mkdir(path_to_data)
        # note this should never exist, as we're inside our ID'd folder
        # look for some existing data, if desired:
        if allow_init_from_existing_file:
            try:
                initfile, equib_time = search_for_starting_file(
                    total_dict, uplift_rate, directory='.')
            except KeyError:
                initfile = None
        else:
            initfile = None
        accel_factor = 1.
        if initfile is None:
            # use time as a unique identifier:
            run_ID = int(time.time())
            # this is a baseline run, so extend the time for run:
            for i in xrange(multiplierforstab*max_loops):
                z_pre = z.copy()
                fr.route_flow()
                ld.run_one_step(dt)
                # now work out where the loose sed is:
                loose_sed = (z - z_pre).clip(0.)
                BR_surface = np.isclose(loose_sed, 0.)
                # topo is BR topo, so:
                z -= loose_sed
                sed[BR_surface] = 0.
                sed[:] += loose_sed
                eroder.run_one_step(dt)
                z[mg.core_nodes] += accel_factor * uplift_rate * dt
                print(i)
                if i % (out_interval * multiplierforstab) == 0:
                    max_zeros = len(str(multiplierforstab * max_loops))
                    zeros_to_add = max_zeros - len(str(i)) + 1
                    # note an OoM buffer! Just to be safe
                    if zeros_to_add < 0:
                        # ...just in case, though should never happen
                        print('Problem allocating zeros on savefiles')
                    ilabel = '0' * zeros_to_add + str(i)
                    identifier = ilabel + '_' + str(run_ID)
                    for field in out_fields:
                        np.savetxt(path_to_data + '/' + field + '_' +
                                   identifier + '.txt', mg.at_node[field])
                    # remember, we can back-calc the actual sed discharge from
                    # K_t, since flux capacity = K_t*A**mt*S**nt, and also
                    # q = k*A**m. K_t -> k using an assumed actual trp
                    # relation.
                    # Or a simpler approach?
            # save the final topo for use next time out:
            initfile = (path_to_data + '/topographic__elevation_after' +
                        str(time_to_stab) + 'y.txt')
            np.savetxt(initfile, mg.at_node['topographic__elevation'])

        else:
            print('Found a file to use to initialize on this run with ' +
                  'uplift_rate = ' + str(uplift_rate) + ': ' + initfile)
            # a file to let us know what we loaded here:
            outfile = open(path_to_data + '/initfile_details_readme.txt', 'w')
            outfile.write('file: ' + initfile + '\n')
            outfile.write('equilibration time (y): ' + str(equib_time))
            outfile.close()
        for bevel_amt in bevel_amounts:
            for whr_to_bevel in elev_fraction_to_bevel:
                z[:] = np.loadtxt(initfile)
                # use time as a unique identifier:
                run_ID = int(time.time())
                path_to_data = (expt_ID + '/uplift_rate_' + str(uplift_rate) +
                                '/bevel_' + str(bevel_amt) + '_elevfrac_' +
                                str(whr_to_bevel))
                os.mkdir(path_to_data)
                # now modify that topo
                zrange = z.max() - z.min()
                z+=10.
                z_flattened = (z - z.min())*bevel_amt
                z_flatandshift = z_flattened + z.min() + zrange * whr_to_bevel
                z[:] = np.where(z < z_flattened, z, z_flattened)
                ###
                fr.route_flow()
                imshow_grid_at_node(mg, z)
                draw_profile(mg)
                mg.at_node['topographic__elevation'][:] = z_flattened
                draw_profile(mg)
                show()
                ###

                for i in xrange(max_loops):
                    fr.route_flow()
                    ld.run_one_step(dt)
                    eroder.run_one_step(dt)
                    z[mg.core_nodes] += uplift_rate * dt
                    print(i)
                    if i % out_interval == 0:
                        max_zeros = len(str(max_loops))  # i.e., 2000 is 4
                        zeros_to_add = max_zeros - len(str(i)) + 1
                        # note an OoM buffer! Just to be safe
                        if zeros_to_add < 0:
                            # ...just in case, though should never happen
                            print('Problem allocating zeros on savefiles')
                        ilabel = '0' * zeros_to_add + str(i)
                        identifier = ilabel + '_' + str(run_ID)
                        for field in out_fields:
                            np.savetxt(path_to_data + '/' + field + '_' +
                                       identifier + '.txt', mg.at_node[field])


def draw_profile(grid, figure_name='profile'):
    """Plot the current channel long profile.
    """
    figure(figure_name)
    # profile_IDs = channel_nodes(
    #     grid, grid.at_node['topographic__steepest_slope'],
    #     grid.at_node['drainage_area'], grid.at_node['flow__receiver_node'])
    # ch_dists = get_distances_upstream(
    #     grid, len(grid.at_node['topographic__steepest_slope']),
    #     profile_IDs, grid.at_node['flow__link_to_receiver_node'])
    profile_str = channel_nodes(
        grid, None,
        grid.at_node['drainage_area'], grid.at_node['flow__receiver_node'])
    profile_IDs = profile_str[0][0]
    dists_str = get_distances_upstream(
        grid, profile_str, grid.at_node['flow__link_to_receiver_node'])
    ch_dists = dists_str[0][0]
    plot(ch_dists, grid.at_node['topographic__elevation'][profile_IDs])
    xlabel('Distance downstream (m)')
    ylabel('Elevation (m)')


def draw_profile_evolution(start, stop, step, format, directory='.',
                           force_same_nodes=False, plot_field_downstream=None,
                           different_runs_different_plots=True):
    """Plot a set of long profiles, loaded from saved data.

    "format" is the number of numbers in the file string.
    e.g., "00150" is 5. "012345" is 6.
    "plot_field_downstream" is the string of a field at which values should
    also be plotted, on a second figure.
    """
    mg2 = deepcopy(mg)
    z2 = mg2.add_zeros('node', 'topographic__elevation', noclobber=False)
    fr2 = FlowRouter(mg2)
    first = True
    for i in xrange(start, stop, step):
        num_zeros = format - len(str(i))
        numberstr = '0'*num_zeros + str(i) + '_'
        # search for this string as a topo:
        prefixed = [filename for filename in os.listdir(directory) if
                    filename.startswith('topographic__elevation_' + numberstr)]
        # grab the unique ID, so we can differentiate plots (if desired)
        unique_ID = prefixed[0][-14:-4]
        print('Plotting the profile with unique ID: ' + unique_ID)
        try:
            z2[:] = np.loadtxt(directory + '/' + prefixed[0])
        except IndexError:
            raise IndexError(
                "it's likely you've mis-set directory, or end is out of range")
        if plot_field_downstream is not None:
            prefixedvals = [filename for filename in os.listdir(directory) if
                            filename.startswith(
                                plot_field_downstream + '_' + numberstr)]
            field_vals = np.loadtxt(directory + '/' + prefixedvals[0])
        fr2.run_one_step()
        if force_same_nodes:
            if first:
                if different_runs_different_plots:
                    draw_profile(mg2, figure_name=('profile_' + unique_ID))
                else:
                    draw_profile(mg2)
                first = False
                if plot_field_downstream is not None:
                    if different_runs_different_plots:
                        figure('field_values_downstream_' + unique_ID)
                    else:
                        figure('field_values_downstream')
                    plot(ch_dists[0], field_vals[profile_IDs[0]])
            else:
                if different_runs_different_plots:
                    figure('profile_' + unique_ID)
                else:
                    figure('profile')
                plot(ch_dists[0], mg2.at_node['topographic__elevation'][
                    profile_IDs[0]])
                if plot_field_downstream is not None:
                    if different_runs_different_plots:
                        figure('field_values_downstream_' + unique_ID)
                    else:
                        figure('field_values_downstream')
                    plot(ch_dists[0], field_vals[profile_IDs[0]])
        else:
            if different_runs_different_plots:
                draw_profile(mg2, figure_name=('profile_' + unique_ID))
            else:
                draw_profile(mg2)
            if plot_field_downstream is not None:
                profile_IDs = channel_nodes(
                    mg2, mg2.at_node['topographic__steepest_slope'],
                    mg2.at_node['drainage_area'], mg2.at_node[
                        'flow__receiver_node'])
                ch_dists = get_distances_upstream(
                    mg2, len(mg2.at_node['topographic__steepest_slope']),
                    profile_IDs, mg2.at_node['flow__link_to_receiver_node'])
                if different_runs_different_plots:
                    figure('field_values_downstream_' + unique_ID)
                else:
                    figure('field_values_downstream')
                plot(ch_dists[0], field_vals[profile_IDs[0]])
    if plot_field_downstream is not None:
        if different_runs_different_plots:
            figure('field_values_downstream_' + unique_ID)
        else:
            figure('field_values_downstream')
        xlabel('Distance downstream (m)')
        ylabel('Field value')


def get_float_of_folder_name(directory='.'):
    """Return float referred to at end of folder name, as a float.

    Folder is either this folder, or a path to another folder ending in a float
    """
    cwd = os.path.abspath(directory)
    _format = 0
    while 1:
        try:
            float_here = float(cwd[(-1 - _format):])
        except ValueError:
            break
        _format += 1
    assert _format != 0
    return float_here


def plot_sed_out(directory='.', step=1, dt=1., method='at_a_fixed_point',
                 values='qs', accel_for_normalization=None):
    """
    Plot a time series of sediment discharge from the largest channel in a sim.
    Step is the frequency of sampling of savefiles.
    dt is the timestep (for correct scaling of the x axis).

    method = {'at_a_fixed_point', 'max_point', 'total_out'}
    If 'at_a_fixed_point', we take the node with max water discharge in step 0.
    If 'max_point', we take the node with max water Q at each step.
    If 'total_out', we take the sum across all the nodes at the bottom edge
    (noting that this will blur the signal).

    values = {'qs', 'qs/Qw', 'Qw'}
    Supply the mugnitude of fault acceleration to 'accel_for_normalization' if
    values=='qs' and you want a normalization to final equilibrium to be
    applied.
    """
    assert method in {'at_a_fixed_point', 'max_point', 'total_out'}
    assert values in {'qs', 'qs/Qw', 'Qw'}
    mg2 = deepcopy(mg)
    z2 = mg2.add_zeros('node', 'topographic__elevation', noclobber=False)
    sedflux2 = mg2.add_zeros('node', 'channel_sediment__volumetric_flux',
                             noclobber=False)
    waterQ2 = mg2.add_zeros('node', 'surface_water__discharge',
                            noclobber=False)
    fr2 = FlowRouter(mg2)
    # search for topo saves, to establish the num files, codes etc:
    prefixed_z = [filename for filename in os.listdir(directory) if
                  filename.startswith('topographic__elevation')]
    # grab the unique ID, so we can differentiate plots (if desired)
    unique_ID = prefixed_z[0][-14:-4]
    # work on first save to establish num digits
    _format = 0
    while True:
        char = prefixed_z[0][-16 - _format]
        try:
            num = int(char)
        except ValueError:  # was a str
            break
        else:
            _format += 1
    # build the out_time list so we can plot the data:
    out_times = []
    for fname in prefixed_z:
        out_times.append(int(fname[::step][(-15 - _format):-15]))
    out_times = np.array(out_times, dtype=float)
    out_times *= dt
    # usefully, prefixed_z is returned in numeric order, because of our
    # numbering scheme. So we can just iterate though to pick up all the data
    sedqsaved = [filename for filename in os.listdir(directory) if
                 filename.startswith('channel_sediment__volumetric_flux')]
    waterQsaved = [filename for filename in os.listdir(directory) if
                   filename.startswith('surface_water__discharge')]
    first = True
    out_vals = []
    for (ztxt, stxt, wtxt) in zip(prefixed_z[::step], sedqsaved[::step],
                                  waterQsaved[::step]):
        z2[:] = np.loadtxt(directory + '/' + ztxt)
        sedflux2[:] = np.loadtxt(directory + '/' + stxt)
        waterQ2[:] = np.loadtxt(directory + '/' + wtxt)
        if values == 'qs':
            outarray = sedflux2
        elif values == 'Qw':
            outarray = waterQ2
        elif values == 'qs/Qw':
            outarray = sedflux2/waterQ2
        if method == 'at_a_fixed_point':
            if first:
                outnode = np.argmax(waterQ2)
                first = False
            out_vals.append(outarray[outnode])
        elif method == 'max_point':
            outnode = np.argmax(waterQ2)
            out_vals.append(outarray[outnode])
        elif method == 'total_out':
            out_vals.append(outarray[mg2.nodes_at_bottom_edge].sum())
    out_vals = np.array(out_vals)
    if values == 'qs' and accel_for_normalization is not None:
        print('adjusting qs for final uplift rate...')
        out_vals /= accel_for_normalization

    # so now we have the data for the two axes:
    plot(out_times, out_vals, '-')
    xlabel('Time since perturbation (y)')
    ylabel('value at outlet')

    return (out_times, out_vals)


def read_and_print_file_for_inputs(directory='.', print_file=False):
    total_dict = np.load(directory + '/expt_ID_paramdict.npy').item()
    if print_file:
        outfile = open(directory + '/expt_ID_params.txt', 'w')
        for key, value in total_dict.iteritems():
            outfile.write(str(key) + ': ' + str(value) + '\n')
        outfile.close()
    return total_dict


def plot_and_calc_concavities_final_run(directory='.'):
    """
    Plots S-A for final topo saved in folder "directory".

    Returns (k_s, concavity) for lin regression on log log trend.

    Concavity is defined as positive.
    """
    mg2 = deepcopy(mg)
    z2 = mg2.add_zeros('node', 'topographic__elevation', noclobber=False)
    fr2 = FlowRouter(mg2)
    # search for topo saves, to establish the num files, codes etc:
    prefixed_z = [filename for filename in os.listdir(directory) if
                  filename.startswith('topographic__elevation')]
    # grab the unique ID, so we can differentiate plots (if desired)
    unique_ID = prefixed_z[0][-14:-4]
    # work on first save to establish num digits
    _format = 0
    while True:
        char = prefixed_z[-1][-16 - _format]
        try:
            num = int(char)
        except ValueError:  # was a str
            break
        else:
            _format += 1

    # load the final topo:
    z2[:] = np.loadtxt(directory + '/' + prefixed_z[-1])
    # create a new flow map:
    fr2.run_one_step()
    profile_IDs = channel_nodes(
        mg2, mg2.at_node['topographic__steepest_slope'],
        mg2.at_node['drainage_area'], mg2.at_node['flow__receiver_node'])
    ch_dists = get_distances_upstream(
        mg2, len(mg2.at_node['topographic__steepest_slope']),
        profile_IDs, mg2.at_node['flow__link_to_receiver_node'])
    slopes = mg2.at_node['topographic__steepest_slope'][profile_IDs[0]]
    areas = mg2.at_node['drainage_area'][profile_IDs[0]]
    logslopes = np.log(slopes)
    logareas = np.log(areas)
    goodvals = np.logical_and(np.isfinite(logslopes), np.isfinite(logareas))
    concavity, log_k_s, _, _, _ = linregress(
        logareas[goodvals], logslopes[goodvals])
    k_s = np.exp(log_k_s)

    loglog(areas, slopes, 'x')
    loglog(areas, k_s * areas**concavity, '-')
    xlabel('Drainage area (m**2)')
    ylabel('Slope (m/m)')

    return k_s, -concavity
