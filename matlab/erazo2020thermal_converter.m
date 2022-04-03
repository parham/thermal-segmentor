
% This script provides a converter to convert the csv files in the dataset
% to thermal images.
%
% References:
% Erazo-Aux, J., Loaiza-Correa, H., Restrepo-Giron, A. D., Ibarra-Castanedo, C., & 
% Maldague, X. (2020). Thermal imaging dataset from composite material academic 
% samples inspected by pulsed thermography. Data in brief, 32, 106313.

% directory path
folder = '/home/phm/Datasets/clemente_data/GFRP-008_facq-145Hz_s-Front_Img-2000';
% create the output folder
out_folder = fullfile(folder,'output');
mkdir(out_folder);
% list of csv files
files = dir(fullfile(folder, '*.csv'));

for index = 1:length(files)
    f = fullfile(folder,files(index).name);
    data = csvread(f);
    res = single(rescale(data));
    % Write the file
    [~, f, ~] = fileparts(files(index).name);
    imwrite(res, fullfile(out_folder, sprintf('%s.png', f)));
end