% before running, please change follow root by yourself
data_root = '/data';
% set groundturth root
root_gt = data_root + '/h36m/gt/test/';
% set coarse 3d predict root
root_coarse3d = data_root + '/h36m/ours_3d/mask3d-400000/';
% set refine 3d predict root
root_refine = '/home/cyan/code/tensorflow/z_refine/humaneva/1213/';
% set 2dpose root, if use_hg2d=True use hourglass predict 2d
% if use_hg2d=False, use our 2d prediction: root_2d = data_root + '/h36m/ours_2d/bilstm2d-p1-800000/';
use_hg2d = True;
root_2d = data_root + '/h36m/hg2dh36m/';
% set img root
im_root = data_root + '/human3.6m/';


close all;
clear;clc;
addpaths;

eval=zeros(15);
m=[];

% load skel information
skel = load('skel_16');
skel = skel.skel;

s = '%s%f';
for i=1:80
   s=strcat(s,'%f');
end
ss = '%s%f';
for i=1:31
   ss=strcat(ss,'%f');
end

max_min = csvread(data_root + '/h36m/16point_mean_limb_scaled_max_min.csv');

for folder_i = 1:15
    action_id = num2str(folder_i);
    fliename = fopen(strcat(root_folder,'test',action_id,'.txt'));
    file = strcat('mpjp_',action_id,'.txt'); % refine 3dpose mpjp eval
    fid = fopen(fullfile(root_refine, file),'r');
    a = cell2mat(textscan(fid,'%f%f%f%f%f%f'));
    [min_a, I] = min(a,[],2);
    sub_a = a(:,1) - min_a;
    idex = find(sub_a>20);
    m(folder_i,:) = mean(a);
    disp([action_id,' eval = ',num2str(m(folder_i,1)),' ', num2str(m(folder_i,2)),' ',num2str(m(folder_i,3)),' ',num2str(m(folder_i,4)),' ', num2str(m(folder_i,5)),' ',num2str(m(folder_i,6))]);
    
    gt2d3d = textscan(fliename,s,'delimiter',[' ',',']);
    im_name = gt2d3d{1,1};
    gt2d3d{1,1}=[];
    gt2d3d = cell2mat(gt2d3d);
    gt_tmp = gt2d3d(:,33:80);
    gt_tmp = gt_tmp .* repmat(max_min(1,:)-max_min(2,:),[size(gt_tmp,1),1]) + repmat(max_min(2,:),[size(gt_tmp,1),1]);
    
    if use_hg2d
    	hg2d_tmp = textscan(fopen([root_2d,'test',action_id,'_square2d.txt'],'r'),ss,'delimiter',[' ',',']);
    	hg2d_tmp{1,1}=[];
    	hg2d_tmp = cell2mat(hg2d_tmp);
    else
    	pred2d_tmp = csvread([root_2d,'test',action_id,'_norm.csv']);
    	hg2d_tmp = pred2d_tmp(:,2:end);

    coarse3d_tmp = csvread([root_coarse3d, action_id, '_unnorm.csv']);
    coarse3d_tmp = coarse3d_tmp(:,2:end);

    % show every frame img
    for j=1:length(idex)
        idex(j)
        im = imread([im_root,im_name{idex(j)}]);
        refine_tmp = csvread([root_refine,'refine3d', '_', action_id,'.txt']);
        refine = refine_tmp(2*idex(j),:);
        refine = reshape(refine,3,16);
        coarse3d = coarse3d_tmp(idex(j),:);
        coarse3d = reshape(coarse3d,3,16);
        gt = gt_tmp(idex(j),:);
        gt = reshape(gt,3,16);
        hg2d = hg2d_tmp(idex(j),:);
        hg2d = hg2d .* max(size(im));
        hg2d = reshape(hg2d,2,16);
        
        clf;
        figure(1),
        subplot(1,4,1),imshow(im);
        hold on
        vis2Dskel(hg2d,skel);title('im & 2d');
        hold off
        subplot(1,4,2),vis3Dskel(gt*0.003,skel, 'viewpoint',[15 15]);title('gt');
        subplot(1,4,3),vis3Dskel(coarse3d*0.003,skel, 'viewpoint',[15 15]);title('coarse3d');
        subplot(1,4,4),vis3Dskel(refine*0.003,skel, 'viewpoint',[15 15]);title('refine3d');
        pause();     
    end  
end

% figure,vis3Dskel(rpsm*0.003,skel, 'viewpoint',[15 15]);title('rpsm');
% saveas(gcf,strcat('/home/cyan/code/tensorflow/z_refine/vis_result/duibi3/action',action_id,'_',int2str(idex(j)),'_rpsm'),'epsc');
% figure,vis3Dskel(refine*0.003,skel, 'viewpoint',[15 15]);title('refine');
% saveas(gcf,strcat('/home/cyan/code/tensorflow/z_refine/vis_result/duibi3/action',action_id,'_',int2str(idex(j)),'_refine'),'epsc');
% figure,imshow(im);
% hold on
% vis2Dskel(hg2d,skel);title('im & 2d');
% hold off
% saveas(gcf,strcat('/home/cyan/code/tensorflow/z_refine/vis_result/duibi3/action',action_id,'_',int2str(idex(j)),'_im'),'epsc');
% figure,vis3Dskel(gt*0.003,skel, 'viewpoint',[15 15]);title('gt');
% saveas(gcf,strcat('/home/cyan/code/tensorflow/z_refine/vis_result/duibi3/action',action_id,'_',int2str(idex(j)),'_gt'),'epsc');
