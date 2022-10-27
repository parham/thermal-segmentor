
clear;
clc;

img_dir = '/home/phm/Dataset/thermal-segmentation/Plate_Simulation/Curve-Steel/Image';
img_files = dir(fullfile(img_dir, '*.png'));

steps = length(img_files);
h = waitbar(0,'Please wait...');
prog = [];
dis = [];
for step = 1:steps
    fpath = fullfile(img_dir, img_files(step).name);
    img = imread(fpath);
    [X_no_dither,map] = rgb2ind(img,5,'nodither');
    color_clss = sort(mean(map, 2));
    color_clss = color_clss(1:end-3);
%     d = std(color_clss);
    d = max(color_clss) - min(color_clss);
    dis = cat(1, dis, d);
    prog = cat(1, prog, color_clss');
    waitbar(step / steps)
end
close(h)