% Generate analysis-only controlled QuaDRiGa CSI for the mechanism study.
%
% This script varies one physical condition at a time while keeping the CSI
% dimensions, scenario, sampling intervals, array, and geometry distribution
% fixed.  It is not used for training or headline extrapolation results.
%
% Required environment variables:
%   CONTROLLED_QUADRIGA_ROOT  versioned output root
%   CONTROL_GROUP             speed | delay | angle
%   QUADRIGA_ROOT             extracted QuaDRiGa source root
% Optional:
%   SAMPLES_PER_LEVEL         defaults to 2000
%   GENERATOR_COMMIT          source-code commit recorded in config.mat
%   QUADRIGA_VERSION          release identifier recorded in config.mat

clc; clear; close all;

output_root = getenv('CONTROLLED_QUADRIGA_ROOT');
control_group = lower(getenv('CONTROL_GROUP'));
quadriga_root = getenv('QUADRIGA_ROOT');
generator_commit = getenv('GENERATOR_COMMIT');
quadriga_version = getenv('QUADRIGA_VERSION');
if isempty(output_root)
    error('CONTROLLED_QUADRIGA_ROOT is required');
end
if ~ismember(control_group, {'speed','delay','angle'})
    error('CONTROL_GROUP must be speed, delay, or angle');
end
if isempty(quadriga_root)
    error('QUADRIGA_ROOT is required');
end
addpath(genpath(quadriga_root));
if isempty(which('qd_simulation_parameters'))
    error('QuaDRiGa is unavailable under QUADRIGA_ROOT');
end
if isempty(generator_commit)
    generator_commit = 'unknown';
end
if isempty(quadriga_version)
    quadriga_version = 'unknown';
end

sample_text = getenv('SAMPLES_PER_LEVEL');
if isempty(sample_text)
    samples_per_level = 2000;
else
    samples_per_level = floor(str2double(sample_text));
end
if ~isfinite(samples_per_level) || samples_per_level < 1
    error('SAMPLES_PER_LEVEL must be a positive integer');
end

% Shared diagnostic configuration.  Each control uses an out-of-pretraining
% scale so that the tested offsets match the extrapolation setting.
scenario = '3GPP_38.901_UMi_NLOS';
fc = 3.5e9;
delta_f = 30e3;
delta_t = 1e-3;
BSlocation = [0; 0; 30];
UEcenter = [200; 0; 1.5];
rho_range = [20,50];
phi_range = [-60,60];
downtilt_deg = 7;
element_spacing_wavelength = 0.5;
array_pattern = '3gpp-mmw';
default_speed_kmh = 30;
speed_levels_kmh = [3,120];
delay_levels_s = [30,1000] * 1e-9;
angle_levels_deg = [2,60];
base_seed = 420000;
matlab_version = version;
quadriga_path = which('qd_simulation_parameters');

switch control_group
    case 'speed'
        requested_levels = speed_levels_kmh;
        requested_unit = 'km/h';
        control_id = 1;
        T = 64; K = 64; UPA = [4,4];
    case 'delay'
        requested_levels = delay_levels_s;
        requested_unit = 's';
        control_id = 2;
        T = 24; K = 512; UPA = [2,4];
    case 'angle'
        requested_levels = angle_levels_deg;
        requested_unit = 'deg';
        control_id = 3;
        T = 16; K = 64; UPA = [16,16];
end
Ant = prod(UPA);

group_dir = fullfile(output_root, control_group);
if ~exist(group_dir, 'dir')
    mkdir(group_dir);
end

for level_id = 1:numel(requested_levels)
    requested_value = requested_levels(level_id);
    level_dir = fullfile(group_dir, sprintf('L%d', level_id));
    if ~exist(level_dir, 'dir')
        mkdir(level_dir);
    end

    H_test = complex(zeros(samples_per_level, T, K, Ant, 'single'));
    speed_test = zeros(samples_per_level, 1, 'single');
    requested_test = repmat(single(requested_value), samples_per_level, 1);
    realized_delay_test = zeros(samples_per_level, 1, 'single');
    realized_asd_test = zeros(samples_per_level, 1, 'single');
    sample_seed_test = zeros(samples_per_level, 1, 'uint32');

    fprintf('Generating %s L%d, requested %.9g %s, %d samples\n', ...
        control_group, level_id, requested_value, requested_unit, ...
        samples_per_level);

    for sample_id = 1:samples_per_level
        % The same sample id receives the same seed at both levels.  This
        % pairs geometry and stochastic draws as far as the simulator permits.
        sample_seed = base_seed + 10000 * control_id + sample_id;
        rng(sample_seed, 'twister');
        sample_seed_test(sample_id) = uint32(sample_seed);

        if strcmp(control_group, 'speed')
            UESpeed = requested_value;
        else
            UESpeed = default_speed_kmh;
        end

        s = qd_simulation_parameters;
        s.center_frequency = fc;
        s.set_speed(UESpeed, delta_t);
        s.use_random_initial_phase = true;
        s.use_3GPP_baseline = 1;

        M_BS = UPA(1);
        N_BS = UPA(2);
        BSAntArray = qd_arrayant.generate(array_pattern, M_BS, N_BS, ...
            s.center_frequency, 1, downtilt_deg, ...
            element_spacing_wavelength, 1, 1, ...
            M_BS * element_spacing_wavelength, ...
            N_BS * element_spacing_wavelength);
        UEAntArray = qd_arrayant.generate(array_pattern, 1, 1, ...
            s.center_frequency, 1, downtilt_deg, ...
            element_spacing_wavelength, 1, 1, ...
            element_spacing_wavelength, element_spacing_wavelength);

        total_time = (T - 1) * delta_t;
        UETrackLength = UESpeed / 3.6 * total_time;
        rho = rho_range(1) + diff(rho_range) * rand();
        phi = phi_range(1) + diff(phi_range) * rand();
        UElocation = [-rho * cosd(phi); rho * sind(phi); 0] + UEcenter;
        UEtrack = qd_track.generate('linear', UETrackLength);
        UEtrack.name = sprintf('%s-L%d-S%d', control_group, level_id, sample_id);
        UEtrack.interpolate('distance', 1 / s.samples_per_meter, [], [], 1);

        layout = qd_layout(s);
        layout.no_tx = 1;
        layout.tx_array = BSAntArray;
        layout.tx_position = BSlocation;
        layout.no_rx = 1;
        layout.rx_array = UEAntArray;
        layout.rx_track = UEtrack;
        layout.rx_position = UElocation;
        layout.set_scenario(scenario);

        builder = layout.init_builder();
        builder.lsp_xcorr = eye(8);
        if strcmp(control_group, 'delay')
            scenpar = builder.scenpar;
            scenpar.DS_mu = log10(requested_value);
            scenpar.DS_sigma = 0;
            builder.scenpar = scenpar;
        elseif strcmp(control_group, 'angle')
            scenpar = builder.scenpar;
            scenpar.AS_D_mu = log10(requested_value);
            scenpar.AS_D_sigma = 0;
            builder.scenpar = scenpar;
        end
        builder.gen_parameters();
        channel = builder.get_channels();

        H_sample = channel.fr(K * delta_f, K);
        H_sample = permute(H_sample, [4,3,2,1]);
        if size(H_sample,1) ~= T
            error('Expected %d snapshots, got %d', T, size(H_sample,1));
        end
        H_test(sample_id,:,:,:) = single(H_sample);
        speed_test(sample_id) = single(UESpeed);
        realized_delay_test(sample_id) = single(mean(builder.ds(:)));
        realized_asd_test(sample_id) = single(mean(builder.asD(:)));

        if mod(sample_id, 25) == 0
            fprintf('%s L%d: %d/%d\n', control_group, level_id, ...
                sample_id, samples_per_level);
        end
    end

    data_path = fullfile(level_dir, 'test_data.mat');
    save(data_path, 'H_test', 'speed_test', 'requested_test', ...
        'realized_delay_test', 'realized_asd_test', 'sample_seed_test', '-v7.3');
    config_path = fullfile(level_dir, 'config.mat');
    save(config_path, 'control_group', 'requested_value', 'requested_unit', ...
        'fc', 'delta_f', 'delta_t', 'T', 'K', 'UPA', 'scenario', ...
        'BSlocation', 'UEcenter', 'rho_range', 'phi_range', ...
        'downtilt_deg', 'element_spacing_wavelength', 'array_pattern', ...
        'default_speed_kmh', 'samples_per_level', 'base_seed', ...
        'generator_commit', 'quadriga_version', 'quadriga_path', ...
        'matlab_version');
    clear H_test speed_test requested_test realized_delay_test;
    clear realized_asd_test sample_seed_test;
end

disp('Controlled QuaDRiGa diagnostic group completed successfully.');
