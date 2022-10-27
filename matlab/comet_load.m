
clear;
clc;

fileName = '/home/phm/Downloads/photos (1)/experiments/experiment 3/ds_3_phm_loss.json'; % filename in JSON extension
fid = fopen(fileName); % Opening the file
raw = fread(fid, inf); % Reading the contents
str = char(raw'); % Transformation
fclose(fid); % Closing the file
data = jsondecode(str); % Using the jsondecode function to parse JSON from string
metric = data.y;
