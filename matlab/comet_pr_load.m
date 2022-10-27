
clear;
clc;

fileName = '/home/phm/Downloads/photos (1)/experiments/deeplab-pr-curve-class-2.json'; % filename in JSON extension
fid = fopen(fileName); % Opening the file
raw = fread(fid, inf); % Reading the contents
str = char(raw'); % Transformation
fclose(fid); % Closing the file
data = jsondecode(str); % Using the jsondecode function to parse JSON from string
x = data.x; y = data.y;
plot(x,y)
