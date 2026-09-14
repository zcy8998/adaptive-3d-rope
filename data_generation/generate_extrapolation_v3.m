% Generate only the new extrapolation scales for the revision.
%
% Required environment variables:
%   EXTRAPOLATION_V3_ROOT  versioned output root
%   SCALE_GROUP            antenna256 | freq512 | freq1024
% Optional:
%   GENERATOR_COMMIT       source-code commit recorded in config.mat
%   QUADRIGA_ROOT          extracted QuaDRiGa source root
%   QUADRIGA_VERSION       release identifier recorded in config.mat
%   SAMPLES_PER_DATASET    defaults to 2000 (use only for smoke tests)

clc; clear; close all;

output_root = getenv('EXTRAPOLATION_V3_ROOT');
scale_group = getenv('SCALE_GROUP');
generator_commit = getenv('GENERATOR_COMMIT');
quadriga_root = getenv('QUADRIGA_ROOT');
quadriga_version = getenv('QUADRIGA_VERSION');
if isempty(output_root)
    error('EXTRAPOLATION_V3_ROOT is required');
end
if isempty(scale_group)
    error('SCALE_GROUP is required');
end
if isempty(generator_commit)
    generator_commit = 'unknown';
end
if ~isempty(quadriga_root)
    addpath(genpath(quadriga_root));
end
if isempty(which('qd_simulation_parameters'))
    error('QuaDRiGa is unavailable; set QUADRIGA_ROOT to its source root');
end
if isempty(quadriga_version)
    quadriga_version = 'unknown';
end

% Scenario-specific distributions are identical to the existing benchmark.
scenes = {
    {3.5,  30,  1.0,  '3GPP_38.901_UMi_NLOS',   [10,30]};
    {2.1,  15,  1.0,  '3GPP_38.901_UMa_LOS',    [30,60]};
    {28.0, 120, 0.25, '3GPP_38.901_Indoor_LOS', [0,5]};
    {0.9,  15,  1.0,  '3GPP_38.901_RMa_NLOS',   [60,120]};
};

switch scale_group
    case 'antenna256'
        suite = 'array';
        first_dataset_id = 25;
        K_target = 64;
        T_target = 16;
        UPA_target = [16,16];
        scale_code = 256;
    case 'freq512'
        suite = 'freq';
        first_dataset_id = 21;
        K_target = 512;
        T_target = 24;
        UPA_target = [2,4];
        scale_code = 512;
    case 'freq1024'
        suite = 'freq';
        first_dataset_id = 25;
        K_target = 1024;
        T_target = 24;
        UPA_target = [2,4];
        scale_code = 1024;
    otherwise
        error('Unsupported SCALE_GROUP: %s', scale_group);
end

samples_per_dataset_text = getenv('SAMPLES_PER_DATASET');
if isempty(samples_per_dataset_text)
    samples_per_dataset = 2000;
else
    samples_per_dataset = str2double(samples_per_dataset_text);
    if ~isfinite(samples_per_dataset) || samples_per_dataset < 1
        error('SAMPLES_PER_DATASET must be a positive integer');
    end
    samples_per_dataset = floor(samples_per_dataset);
end
BSlocation = [0; 0; 30];
UEcenter = [200; 0; 1.5];
rho_range = [20,50];
phi_range = [-60,60];
downtilt_deg = 7;
element_spacing_wavelength = 0.5;
array_pattern = '3gpp-mmw';
matlab_version = version;
quadriga_path = which('qd_simulation_parameters');

suite_dir = fullfile(output_root, suite);
if ~exist(suite_dir, 'dir')
    mkdir(suite_dir);
end

for scene_id = 1:length(scenes)
    entry = scenes{scene_id};
    fc = entry{1} * 1e9;
    K = K_target;
    delta_f = entry{2} * 1e3;
    T = T_target;
    delta_t = entry{3} * 1e-3;
    UPA = UPA_target;
    scenario = entry{4};
    speed_range = entry{5};
    Ant = prod(UPA);
    dataset_id = first_dataset_id + scene_id - 1;
    rng_seed = 420000 + 1000 * scale_code + scene_id;
    rng(rng_seed, 'twister');

    dataset_dir = fullfile(suite_dir, sprintf('D%d', dataset_id));
    if ~exist(dataset_dir, 'dir')
        mkdir(dataset_dir);
    end
    H_test = complex(zeros(samples_per_dataset, T, K, Ant, 'single'));
    speed_test = zeros(samples_per_dataset, 1, 'single');

    fprintf('Generating %s D%d (%s), %d samples\n', ...
        scale_group, dataset_id, scenario, samples_per_dataset);
    for sample_id = 1:samples_per_dataset
        UESpeed = speed_range(1) + diff(speed_range) * rand();
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
        UEtrack.name = num2str(sample_id);
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
        [channel, ~] = layout.get_channels();

        bandwidth = K * delta_f;
        H_sample = channel.fr(bandwidth, K);
        H_sample = permute(H_sample, [4,3,2,1]);
        H_test(sample_id,:,:,:) = single(H_sample);
        speed_test(sample_id) = single(UESpeed);
        if mod(sample_id, 50) == 0
            fprintf('%s D%d: %d/%d\n', scale_group, dataset_id, ...
                sample_id, samples_per_dataset);
        end
    end

    test_path = fullfile(dataset_dir, 'test_data.mat');
    save(test_path, 'H_test', 'speed_test', '-v7.3');
    config_path = fullfile(dataset_dir, 'config.mat');
    save(config_path, 'fc', 'K', 'delta_f', 'T', 'delta_t', 'UPA', ...
        'scenario', 'speed_range', 'BSlocation', 'UEcenter', 'rho_range', ...
        'phi_range', 'downtilt_deg', 'element_spacing_wavelength', ...
        'array_pattern', 'samples_per_dataset', 'rng_seed', ...
        'matlab_version', 'quadriga_version', 'quadriga_path', ...
        'generator_commit');
    clear H_test speed_test;
end

disp('Requested extrapolation scale group completed successfully.');
