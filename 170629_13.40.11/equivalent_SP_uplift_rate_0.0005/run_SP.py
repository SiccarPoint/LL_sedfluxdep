"""
Loads an existing landscape created as::

    sde = SedDepEroder(mg, K_sp=1.e-4, sed_dependency_type='almost_parabolic',
                       Qc='power_law', K_t=1.e-4)

Then perturbs it.

This is primarily a plotter for a single magnitude of uplift perturbation.
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

accel_factors = (5., 10., 20.)

#uplift_rates = (0.00001, 0.0001, 0.00025, 0.0005, 0.001)
uplift_rates = (0.0005, )

max_loops = 10000
multiplierforstab = 4

raster_params = {'shape': (50, 50), 'dx': 200.,
                 'initcond': 'initcondst5000.txt'}

inputs_sp = {'K_sp': 1.e-5, 'K_t': 1.e-5}
inputs_ld = {'linear_diffusivity': 1.e-2}

dt = 200.  # was 100.
out_interval = 25

color = 'gnuplot2'  # 'winter'

out_fields = [
        'topographic__elevation',
        'channel_sediment__volumetric_flux']

# build the structures:
mg = RasterModelGrid(raster_params['shape'], raster_params['dx'])
for edge in (mg.nodes_at_left_edge, mg.nodes_at_top_edge,
             mg.nodes_at_right_edge):
    mg.status_at_node[edge] = CLOSED_BOUNDARY

z = mg.add_field('node', 'topographic__elevation',
                 np.loadtxt(raster_params['initcond']))

sed = mg.add_zeros('node', 'channel_sediment__volumetric_flux', dtype=float)

fr = FlowRouter(mg)
eroder = FastscapeEroder(mg, **inputs_sp)
ld = LinearDiffuser(mg, **inputs_ld)


def build_master_dict(expt_ID):
    total_dict = inputs_sp.copy()
    total_dict.update(inputs_ld)
    total_dict.update(raster_params)
    total_dict['expt_ID'] = expt_ID
    total_dict['dt'] = dt
    total_dict['max_loops'] = max_loops
    total_dict['multiplierforstab'] = multiplierforstab
    total_dict['out_interval'] = out_interval
    total_dict['uplift_rates'] = uplift_rates
    total_dict['accel_factors'] = accel_factors
    return total_dict


def search_for_starting_file(dict_of_all_params, uplift_rate, directory='.',
                             enforce_equilibrium_time=False):
    """
    Returns a tuple of (path_to_existing_equilibrium_topo_file, run_time) or
    (None, None) if none exists. the file must have the same run conditions as
    this run.
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
                           'accel_factors']:
                    continue
                if key in ['dt', 'multiplierforstab', 'max_loops']:
                    if not enforce_equilibrium_time:  # ignore time to =brium
                        continue
                if dict_of_all_params[key] != found_params[key]:
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
        return (paths_of_valid_starting_files[0], equivalent_timestostab[0])
    else:
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
        if initfile is None:
            accel_factor = 1.
            # use time as a unique identifier:
            run_ID = int(time.time())
            # this is a baseline run, so extend the time for run:
            for i in xrange(multiplierforstab*max_loops):
                z_pre = z.copy()
                fr.route_flow()
                ld.run_one_step(dt)

                # flush any diffused loose sed:
                z[:] = np.minimum(z, z_pre)

                eroder.run_one_step(dt)

                # measure the erosion done here; scale it as the sde has scaled
                # it elsewhere (i.e., avg per node)
                sed[mg.core_nodes] = (z_pre[mg.core_nodes] -
                          z[mg.core_nodes])/mg.number_of_core_nodes

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
        for accel_factor in accel_factors:
            z[:] = np.loadtxt(initfile)
            # use time as a unique identifier:
            run_ID = int(time.time())
            path_to_data = (expt_ID + '/uplift_rate_' + str(uplift_rate) +
                            '/accel_' + str(accel_factor))
            os.mkdir(path_to_data)
            for i in xrange(max_loops):
                z_pre = z.copy()
                fr.route_flow()
                ld.run_one_step(dt)

                # flush any diffused loose sed:
                z[:] = np.minimum(z, z_pre)

                eroder.run_one_step(dt)

                # measure the erosion done here; scale it as the sde has scaled
                # it elsewhere (i.e., avg per node)
                sed[mg.core_nodes] = (z_pre[mg.core_nodes] -
                          z[mg.core_nodes])/mg.number_of_core_nodes

                z[mg.core_nodes] += accel_factor * uplift_rate * dt
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
    profile_IDs = channel_nodes(
        grid, grid.at_node['topographic__steepest_slope'],
        grid.at_node['drainage_area'], grid.at_node['flow__receiver_node'])
    ch_dists = get_distances_upstream(
        grid, len(grid.at_node['topographic__steepest_slope']),
        profile_IDs, grid.at_node['flow__link_to_receiver_node'])
    plot(ch_dists[0], grid.at_node['topographic__elevation'][profile_IDs[0]])
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


def extend_equilibrium_run(uplift_rates_to_extend):
    """******NOT TESTED******
    Call this function to extend runs for attempts at equilibrium that didn't
    quite make it. A new experiment is made, any useful existing data is ported
    over from the best available old run, then runs begin from the point the
    last experiment terminated.
    uplift_rates_to_extend is a list of rates to extend.
    Assumes properties are set in-line.
    Assumes a previous run has been allowed to complete, i.e., the file
    topographic__elevation_after_...' exists.
    """
    expt_ID, total_dict = make_expt_folder_and_store_master_dict()
    for uplift_rate in uplift_rates_to_extend:
        run_ID = int(time.time())
        initfile, equib_time = search_for_starting_file(
            total_dict, uplift_rate, directory='.')
        # initfile is the topo file, with full path. We want that path.
        if initfile is None:
            print('No existing run found for uplift rate ' + str(uplift_rate))
            continue  # no initfile
        init_path, init_fname = os.path.split(initfile)
        # grab that old paramdict again:
        found_params = np.load(init_path + '/../expt_ID_paramdict.npy').item()
        old_interval = found_params['out_interval']
        if out_interval < old_interval:
            print("U = " + str(uplift_rate) + ": New out_interval must " +
                  "exceed previous, which was " + str(old_interval))
            continue
        multiples = out_interval // old_interval
        if not out_interval % old_interval == 0:
            print("U = " + str(uplift_rate) + ": New out_interval must be " +
                  "an integer multiple of old out_interval, which was " +
                  str(old_interval))
            continue
        old_iters = found_params[max_loops]
        if old_iters >= max_loops:
            print("U = " + str(uplift_rate) + ": old run is longer than new!")
            continue
        # now, copy over that sweet sweet data:
        newdir = expt_ID + '/uplift_rate_' + str(uplift_rate)
        os.mkdir(newdir)
        old_numchars = len(str(old_iters)) + 1
        new_numchars = len(str(max_loops)) + 1
        for field in out_fields:
            for j in xrange(0, old_iters, multiples):
                old_num_zeros = old_numchars - len(str(i))
                new_num_zeros = new_numchars - len(str(i))
                fname_list = [f for f in os.listdir(init_path) if f.startswith(
                    field + '_' + '0'*old_num_zeros + str(j) + '_')]
                assert len(fname_list) == 1
                copyfile(init_path + '/' + fname_list[0],
                         newdir + '/' + field + '_' + '0'*new_num_zeros +
                         str(j) + '_' + str(run_ID) + '.txt')
        # add a note that we've copied these:
        outfile = open(newdir + '/copied_data_readme.txt', 'w')
        outfile.write('Some files in this folder were copied from an older ' +
                      'run.')
        outfile.write('The files were copied from ' + init_path)
        outfile.write('The maximum iteration that is a copy is ' +
                      str(old_iters))
        outfile.close()
        # now, note the current value of j is the last good file ID, so:
        mg.at_node['topographic__elevation'][:] = np.loadtxt(
            newdir + '/topographic__elevation_' + '0'*new_num_zeros + str(j) +
            '_' + str(run_ID) + '.txt')
        path_to_data = newdir
        for i in xrange(j, max_loops):
            fr.route_flow()
            eroder.run_one_step(dt)
            ld.run_one_step(dt)
            z[mg.core_nodes] += uplift_rate * dt
            print(i)
            if i % out_interval == 0 and i != j:
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


def extend_perturbed_runs(total_iters_to_reach=0):
    """Load all perturbed runs in current folder, and extend them.

    Function should be called from within an experiment folder
    (extend all perturbations for all starting uplift rates), an
    'uplift_rate_XXXX' folder (extend all perturbations for this rate) or an
    'accel_XXX' folder (extend this accel only).

    Does NOT create a new expt or run ID, just extends the old ones. Adds a
    text file annotating what has happened.
    """
    # look for the params to use. Also tells us where we are in the hierarchy
    level = 0  # 0: top, 1: uplift, 2: accel:
    cwd = os.getcwd()
    while True:
        try:
            paramdict = np.load('expt_ID_paramdict.npy').item()
        except IOError:
            os.chdir('..')
            level += 1
        else:
            break
    # now back to where we started in the dir str:
    os.chdir(cwd)
    if level == 2:  # in accel_ folder
        # get the accel that this is:
        accel_factors = [get_float_of_folder_name(), ]
        # get the U of the host folder:
        uplift_rates = [get_float_of_folder_name(directory=(cwd + '/..')), ]
        wd_stub = os.path.abspath(os.getcwd() + '/../..')
    elif level == 1:  # in uplift_ folder
        accel_fnames = [filename for filename in os.listdir('.') if
                        filename.startswith('accel_')]
        accel_factors = [get_float_of_folder_name(directory=(
            cwd + '/' + filename)) for filename in accel_fnames]
        uplift_rates = [get_float_of_folder_name(), ]
        wd_stub = os.path.abspath(os.getcwd() + '/..')
    elif level == 0:  # in top folder
        uplift_fnames = [filename for filename in os.listdir('.') if
                         filename.startswith('uplift_rate_')]
        uplift_rates = [get_float_of_folder_name(directory=(
            cwd + '/' + filename)) for filename in uplift_fnames]
        accel_factors = paramdict['accel_factors']
        wd_stub = os.path.abspath(os.getcwd())

    for uplift_rate in uplift_rates:
        for accel_factor in accel_factors:
            wd = (wd_stub + '/uplift_rate_' + str(uplift_rate) + '/accel_' +
                  str(accel_factor))
            # get the saved filenames that already exist in this folder:
            runnames = [filename for filename in os.listdir(wd) if
                        filename.startswith('topographic__elevation')]
            seddepthnames = [filename for filename in os.listdir(wd) if
                             filename.startswith('channel_sediment__depth')]
            # as elsewhere, the final entry is the last run, so --
            # establish the loop number of that run:
            run_ID = runnames[-1][-14:-4]  # is a str
            _format = 0
            while True:
                char = runnames[-1][-16 - _format]
                try:
                    num = int(char)
                except ValueError:  # was a str
                    break
                else:
                    _format += 1
            finaliter = int(runnames[-1][(-15 - _format):-15])
            finalsediter = int(seddepthnames[-1][(-15 - _format):-15])
            assert finaliter == finalsediter  # ...just in case

            # test we need to actually do more runs:
            if total_iters_to_reach < finaliter + paramdict['out_interval']:
                continue

            # check we aren't going to have a "zero problem"; correct if we do
            max_zeros = len(str(total_iters_to_reach))
            if max_zeros + 1 > _format:  # less won't be possible from continue
                extra_zeros = max_zeros + 1 - _format
                for allfile in os.listdir(wd):
                    if allfile[-14:-4] == run_ID:
                        os.rename(wd + '/' + allfile, (
                            wd + '/' + allfile[:(-15 - _format)] +
                            '0'*extra_zeros + allfile[(-15-_format):]))
                runnames = [filename for filename in os.listdir(wd) if
                            filename.startswith('topographic__elevation')]
                seddepthnames = [filename for filename in os.listdir(wd) if
                                 filename.startswith(
                                    'channel_sediment__depth')]
            if max_zeros + 1 < _format:
                max_zeros = _format - 1  # in case of any bonus 0s from old run

            # build the structures:
            mg = RasterModelGrid(paramdict['shape'], paramdict['dx'])
            for edge in (mg.nodes_at_left_edge, mg.nodes_at_top_edge,
                         mg.nodes_at_right_edge):
                mg.status_at_node[edge] = CLOSED_BOUNDARY

            z = mg.add_zeros('node', 'topographic__elevation')
            seddepth = mg.add_zeros('node', 'channel_sediment__depth')
            fr = FlowRouter(mg)
            eroder = SedDepEroder(mg, **paramdict)
            ld = LinearDiffuser(mg, **paramdict)

            # load the last available elev data:
            z[:] = np.loadtxt(wd + '/' + runnames[-1])
            seddepth[:] = np.loadtxt(wd + '/' + seddepthnames[-1])

            # save a note
            try:
                appendfile = open(wd + '/appended_run_readme.txt', 'a')
            except IOError:
                appendfile = open(wd + '/appended_run_readme.txt', 'w')
            appendfile.write(
                'This run was appended at timestamp ' + str(int(time.time())) +
                '.\n')
            appendfile.write(
                'New loops were added from iteration ' + str(finaliter) +
                ' and terminated at iteration ' + str(total_iters_to_reach) +
                '.\n\n')
            appendfile.close()

            # get runnin'
            print('Extending uplift ' + str(uplift_rate) + ' accel ' +
                  str(accel_factor) + ' from iter number ' + str(finaliter))
            dt = paramdict['dt']
            for i in xrange(finaliter + 1, total_iters_to_reach):
                fr.route_flow()
                eroder.run_one_step(dt)
                ld.run_one_step(dt)
                z[mg.core_nodes] += accel_factor * uplift_rate * dt
                print(i)
                if i % out_interval == 0:
                    zeros_to_add = max_zeros - len(str(i)) + 1
                    # note an OoM buffer! Just to be safe
                    if zeros_to_add < 0:
                        # ...just in case, though should never happen
                        print('Problem allocating zeros on savefiles')
                    ilabel = '0' * zeros_to_add + str(i)
                    identifier = ilabel + '_' + str(run_ID)
                    for field in out_fields:
                        np.savetxt(wd + '/' + field + '_' +
                                   identifier + '.txt', mg.at_node[field])


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
                 values='qs', accel_for_normalization=None, fixed_node=None):
    """
    Plot a time series of sediment discharge from the largest channel in a sim.
    Step is the frequency of sampling of savefiles.
    dt is the timestep (for correct scaling of the x axis).

    method = {'at_a_fixed_point', 'total_out'}
    If 'at_a_fixed_point', specify node with fixed_node.
    If 'total_out', we take the sum across all the nodes at the bottom edge
    (noting that this will blur the signal).

    values = {'qs', }
    Supply the mugnitude of fault acceleration to 'accel_for_normalization' if
    values=='qs' and you want a normalization to final equilibrium to be
    applied.
    """
    assert method in {'at_a_fixed_point', 'max_point', 'total_out'}
    assert values in {'qs', }
    mg2 = deepcopy(mg)
    z2 = mg2.add_zeros('node', 'topographic__elevation', noclobber=False)
    sedflux2 = mg2.add_zeros('node', 'channel_sediment__volumetric_flux',
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
    first = True
    out_vals = []
    for (ztxt, stxt) in zip(prefixed_z[::step], sedqsaved[::step]):
        z2[:] = np.loadtxt(directory + '/' + ztxt)
        sedflux2[:] = np.loadtxt(directory + '/' + stxt)
        if values == 'qs':
            outarray = sedflux2
        if method == 'at_a_fixed_point':
            if first:
                assert fixed_node is not None
                outnode = fixed_node
                first = False
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


# os.chdir('170622_17.54.54/uplift_rate_0.0001')
# os.chdir('accel_2.0')
# plot_sed_out(dt=200., accel_for_normalization=2.)
# os.chdir('../accel_5.0')
# plot_sed_out(dt=200., accel_for_normalization=5.)
# os.chdir('../accel_10.0')
# plot_sed_out(dt=200., accel_for_normalization=10.)
# os.chdir('../accel_20.0')
# plot_sed_out(dt=200., accel_for_normalization=20.)
# os.chdir('../../..')
# show()

# for root, dirs, files in os.walk(path):
#     files = [f for f in files if not f[0] == '.']
#     dirs[:] = [d for d in dirs if not d[0] == '.']
#     print files
